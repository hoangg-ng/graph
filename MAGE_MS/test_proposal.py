import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from proposal_model import (
    ARMS,
    ExpertPool,
    ProposalMAGE,
    parameter_count,
)


def make_args(nodes=7, model_dim=16, experts=4, topk=2, graph_dim=5):
    return SimpleNamespace(
        model_dim=model_dim,
        recur_num=experts,
        topk=topk,
        node_num=nodes,
        feature_dim=1,
        graph_dim=graph_dim,
        input_dim=3,
        output_dim=1,
        seq_len=96,
        horizon=12,
        minute=288,
        day=7,
    )


def make_input(batch=2, nodes=7):
    data = torch.randn(batch, 96, nodes, 1)
    time = torch.arange(96).remainder(288).float() / 288
    data = torch.cat(
        (
            data,
            time[None, :, None, None].expand(batch, -1, nodes, -1),
            torch.zeros(batch, 96, nodes, 1),
        ),
        dim=-1,
    )
    return data


class ProposalModelTests(unittest.TestCase):
    def test_pdm_outputs_shapes_and_proposed_model_has_gradients(self):
        model = ProposalMAGE(
            make_args(), "proposed", pdm_width=8, pdm_depth=1, dropout=0
        )
        x = make_input()
        encoded = model.temporal(x)
        self.assertEqual(len(encoded), 4)
        self.assertTrue(all(value.shape == (2, 7, 16) for value in encoded))

        model.train()
        prediction, counts = model(x, bias=torch.zeros(3, 4))
        self.assertEqual(prediction.shape, (2, 12, 7, 1))
        self.assertEqual(counts.shape, (3, 4))
        prediction.square().mean().backward()
        self.assertIsNotNone(model.blocks[0].pools[0].lambda_logits.grad)
        self.assertTrue(torch.isfinite(model.temporal.embedding.weight.grad).all())

    def test_factorized_power_matches_explicit_dense_power(self):
        torch.manual_seed(4)
        pool = ExpertPool(num_experts=3, num_nodes=5, graph_dim=2)
        route = torch.softmax(torch.randn(2, 5, 3), dim=-1)
        x = torch.randn(2, 5, 4)

        e1 = torch.softmax(pool.E1[0], dim=-1)
        e2 = torch.softmax(pool.E2[0], dim=-1)
        experts = torch.einsum("end,edj->enj", e1, e2)
        dense = torch.einsum("bne,enj->bnj", route, experts)
        self.assertTrue(torch.allclose(dense.sum(dim=-1), torch.ones(2, 5), atol=1e-6))

        expected = x
        for _ in range(4):
            expected = torch.bmm(dense, expected)
        actual = pool.positive_power(x, route, power=4)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_negative_operator_uses_per_expert_sigmoid_coefficients(self):
        pool = ExpertPool(num_experts=3, num_nodes=4, graph_dim=2)
        with torch.no_grad():
            pool.lambda_logits.copy_(torch.tensor([-4.0, 0.0, 4.0]))
        route = torch.full((1, 4, 3), 1 / 3)
        x = torch.randn(1, 4, 2)
        actual = pool.apply(x, route, sign=1)
        common = torch.sigmoid(pool.lambda_logits).mean() * pool.apply(x, route, sign=0)
        self.assertFalse(torch.allclose(actual, common))
        actual.square().sum().backward()
        self.assertTrue(torch.all(pool.lambda_logits.grad.abs() > 0))

    def test_separate_signed_graph_parameters(self):
        pool = ExpertPool(num_experts=3, num_nodes=5, graph_dim=2)
        self.assertEqual(pool.E1.shape, (2, 3, 5, 2))
        self.assertEqual(pool.E2.shape, (2, 3, 2, 5))

        route = torch.softmax(torch.randn(2, 5, 3), dim=-1)
        x = torch.randn(2, 5, 4)
        lam = torch.sigmoid(pool.lambda_logits)
        e1_minus = torch.softmax(pool.E1[1], dim=-1)
        e2_minus = torch.softmax(pool.E2[1], dim=-1)
        dense_minus = torch.einsum(
            "bne,end,edj->bnj",
            route * lam.view(1, 1, -1),
            e1_minus,
            e2_minus,
        )

        expected = torch.bmm(dense_minus, x)
        actual = pool.apply(x, route, sign=1)
        torch.testing.assert_close(actual, expected)

    def test_schedules_have_matching_power_budget_and_graph_params(self):
        shared = ProposalMAGE(make_args(), "shared", pdm_width=8, pdm_depth=1)
        proposed = ProposalMAGE(make_args(), "proposed", pdm_width=8, pdm_depth=1)
        reversed_model = ProposalMAGE(make_args(), "reversed", pdm_width=8, pdm_depth=1)
        independent = ProposalMAGE(make_args(), "independent", pdm_width=8, pdm_depth=1)
        uniform = ProposalMAGE(make_args(), "uniform", pdm_width=8, pdm_depth=1)

        self.assertEqual(sum((1, 2, 4, 8)), sum((8, 4, 2, 1)))
        self.assertEqual(proposed.propagation_count, reversed_model.propagation_count)
        self.assertEqual(proposed.positive_propagation_count, 45)
        self.assertEqual(proposed.propagation_count, 57)
        self.assertEqual(parameter_count(proposed), parameter_count(reversed_model))
        self.assertEqual(proposed.graph_parameter_count, shared.graph_parameter_count)
        self.assertGreater(independent.graph_parameter_count, shared.graph_parameter_count)
        self.assertEqual(uniform.graph_parameter_count, 0)
        self.assertEqual(len(independent.blocks[0].pools), 4)
        self.assertEqual(len(uniform.blocks[0].pools), 0)
        self.assertIsNone(uniform.blocks[0].router)

    def test_forward_uses_the_declared_number_of_graph_applications(self):
        model = ProposalMAGE(
            make_args(), "proposed", pdm_width=8, pdm_depth=1, dropout=0
        ).eval()
        original_apply = ExpertPool.apply
        calls = 0

        def counted_apply(pool, x, route, sign=0):
            nonlocal calls
            calls += 1
            return original_apply(pool, x, route, sign)

        with patch.object(ExpertPool, "apply", counted_apply):
            with torch.no_grad():
                model(make_input(batch=1))
        self.assertEqual(calls, model.propagation_count)

    def test_every_core_arm_produces_the_expected_output(self):
        x = make_input(batch=1)
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = ProposalMAGE(
                    make_args(), arm, pdm_width=8, pdm_depth=1, dropout=0
                ).eval()
                with torch.no_grad():
                    prediction = model(x)
                self.assertEqual(prediction.shape, (1, 12, 7, 1))


if __name__ == "__main__":
    unittest.main()

"""Additive implementation of the MAGE proposal's five core model variants."""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.abspath(__file__))
for source_dir in ("official", "official_repro"):
    candidate = os.path.join(ROOT, source_dir)
    if os.path.isfile(os.path.join(candidate, "src", "models", "MAGE.py")):
        sys.path.insert(0, candidate)
        break

import models as mage_models  # noqa: E402


SCALES = (1, 2, 4, 8)
SCHEDULES = {
    "shared": (1, 1, 1, 1),
    "proposed": (1, 2, 4, 8),
    "reversed": (8, 4, 2, 1),
    "independent": (1, 1, 1, 1),
    "uniform": (0, 0, 0, 0),
}
EXTRA_SCHEDULES = {
    "no_spatial": (0, 0, 0, 0),
    "grouped": (1,),
    "grouped_no_pdm": (1,),
    "grouped_no_spatial": (0,),
    "grouped_uniform": (0,),
    "proposed_no_pdm": SCHEDULES["proposed"],
}
ARM_SCHEDULES = {**SCHEDULES, **EXTRA_SCHEDULES}
ARMS = tuple(ARM_SCHEDULES)
GROUPED_ARMS = ("grouped", "grouped_no_pdm", "grouped_no_spatial", "grouped_uniform")
NO_GRAPH_ARMS = ("uniform", "no_spatial", "grouped_no_spatial", "grouped_uniform")


def _pool_time(x, factor):
    batch, length, nodes, channels = x.shape
    if length % factor:
        x = x[:, length % factor :]
    return x.reshape(batch, length // factor, factor, nodes, channels).mean(dim=2)


class _MovingAverage(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width < 1:
            raise ValueError("moving-average width must be positive")
        self.width = width

    def forward(self, x):
        left = (self.width - 1) // 2
        right = self.width - 1 - left
        padded = torch.cat(
            (
                x[:, :1].expand(-1, left, -1),
                x,
                x[:, -1:].expand(-1, right, -1),
            ),
            dim=1,
        )
        return F.avg_pool1d(padded.transpose(1, 2), self.width, stride=1).transpose(1, 2)


def _temporal_mlp(source_length, target_length):
    return nn.Sequential(
        nn.Linear(source_length, target_length),
        nn.GELU(),
        nn.Linear(target_length, target_length),
    )


class _PDMBlock(nn.Module):
    """TimeMixer-style decomposition and cross-resolution temporal mixing."""

    def __init__(self, lengths, moving_average):
        super().__init__()
        self.decompose = _MovingAverage(moving_average)
        self.seasonal_up = nn.ModuleList(
            _temporal_mlp(lengths[i - 1], lengths[i]) for i in range(1, len(lengths))
        )
        self.trend_down = nn.ModuleList(
            _temporal_mlp(lengths[i + 1], lengths[i]) for i in range(len(lengths) - 1)
        )

    def forward(self, scales):
        trend = [self.decompose(x) for x in scales]
        seasonal = [x - t for x, t in zip(scales, trend)]

        for i in range(1, len(seasonal)):
            mixed = self.seasonal_up[i - 1](seasonal[i - 1].transpose(1, 2)).transpose(1, 2)
            seasonal[i] = seasonal[i] + mixed
        for i in range(len(trend) - 2, -1, -1):
            mixed = self.trend_down[i](trend[i + 1].transpose(1, 2)).transpose(1, 2)
            trend[i] = trend[i] + mixed
        return [s + t for s, t in zip(seasonal, trend)]


class PDMScales(nn.Module):
    """Return one time-encoded (batch, nodes, model_dim) tensor per resolution."""

    def __init__(
        self,
        seq_len,
        flow_dim,
        model_dim,
        scales=SCALES,
        width=16,
        depth=2,
        moving_average=5,
    ):
        super().__init__()
        if seq_len < max(scales):
            raise ValueError("seq_len must provide at least one sample at every scale")
        if depth < 1:
            raise ValueError("PDM depth must be positive")
        self.scales = tuple(scales)
        self.lengths = tuple(seq_len // scale for scale in self.scales)
        self.flow_dim = flow_dim
        self.embedding = nn.Linear(flow_dim, width)
        self.blocks = nn.ModuleList(
            _PDMBlock(self.lengths, moving_average) for _ in range(depth)
        )
        self.projections = nn.ModuleList(
            nn.Linear(length * width, model_dim) for length in self.lengths
        )

    def forward(self, x):
        batch, length, nodes, channels = x.shape
        if length < max(self.scales) or channels < self.flow_dim:
            raise ValueError("input does not satisfy the configured PDM dimensions")
        flow = x[..., : self.flow_dim]
        scales = []
        for factor in self.scales:
            pooled = _pool_time(flow, factor)
            embedded = self.embedding(
                pooled.permute(0, 2, 1, 3).reshape(batch * nodes, -1, self.flow_dim)
            )
            scales.append(embedded)
        for block in self.blocks:
            scales = block(scales)
        return [
            projection(value.reshape(batch, nodes, -1))
            for projection, value in zip(self.projections, scales)
        ]


class LinearScaleEncoder(nn.Module):
    """Encode pooled flow independently at each temporal scale."""

    def __init__(self, seq_len, flow_dim, model_dim, scales=SCALES):
        super().__init__()
        self.scales = tuple(scales)
        self.flow_dim = flow_dim
        self.encoders = nn.ModuleList(
            nn.Linear((seq_len // scale) * flow_dim, model_dim)
            for scale in self.scales
        )

    def forward(self, x):
        flow = x[..., : self.flow_dim]
        batch, _, nodes, _ = flow.shape
        outputs = []
        for scale, encoder in zip(self.scales, self.encoders):
            pooled = _pool_time(flow, scale)
            flattened = pooled.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
            outputs.append(encoder(flattened))
        return outputs


class ExpertPool(nn.Module):
    """Factorized low-rank positive and expert-weighted negative graph operators."""

    def __init__(self, num_experts, num_nodes, graph_dim):
        super().__init__()
        self.num_experts = num_experts
        self.E1 = nn.Parameter(torch.rand(2, num_experts, num_nodes, graph_dim))
        self.E2 = nn.Parameter(torch.rand(2, num_experts, graph_dim, num_nodes))
        self.lambda_logits = nn.Parameter(torch.zeros(num_experts))

    def apply(self, x, route, sign=0):
        if sign not in (0, 1):
            raise ValueError("sign must be 0 (positive) or 1 (negative)")
        e1 = torch.softmax(self.E1[sign], dim=-1)
        e2 = torch.softmax(self.E2[sign], dim=-1)
        weights = route
        if sign == 1:
            weights = weights * torch.sigmoid(self.lambda_logits).view(1, 1, -1)

        batch, nodes, features = x.shape
        hidden = torch.einsum("edn,bnf->bedf", e2, x)
        node_to_expert = (
            weights.unsqueeze(-1)
            * e1.permute(1, 0, 2).unsqueeze(0)
        ).reshape(batch, nodes, -1)
        return torch.bmm(node_to_expert, hidden.reshape(batch, -1, features))

    def positive_power(self, x, route, power):
        if power < 1:
            raise ValueError("positive graph power must be at least one")
        for _ in range(power):
            x = self.apply(x, route, sign=0)
        return x


class ProposalBlock(nn.Module):
    def __init__(
        self,
        arm,
        model_dim,
        num_nodes,
        num_experts,
        topk,
        graph_dim,
        dropout,
    ):
        super().__init__()
        self.arm = arm
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = mage_models.M.SwiGLU_FFN(
            model_dim, model_dim, dropout=dropout
        )
        if arm in NO_GRAPH_ARMS:
            self.router = None
            self.pools = nn.ModuleList()
        else:
            self.router = nn.Linear(model_dim, num_experts)
            pool_count = len(SCALES) if arm == "independent" else 1
            self.pools = nn.ModuleList(
                ExpertPool(num_experts, num_nodes, graph_dim) for _ in range(pool_count)
            )
        self.topk = topk
        self.num_experts = num_experts

    def _route(self, fine, bias):
        scores = torch.sigmoid(self.router(self.norm1(fine)))
        topk = torch.topk(scores + bias, self.topk, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk, True)
        selected = torch.where(mask, scores, torch.zeros_like(scores))
        route = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return route, topk

    def forward(self, streams, bias):
        if self.arm in NO_GRAPH_ARMS:
            topk = None
            route = None
        else:
            route, topk = self._route(streams[0], bias)

        outputs = []
        for scale_index, hidden in enumerate(streams):
            normalized = self.norm1(hidden)
            if self.arm in ("uniform", "grouped_uniform"):
                mixed = normalized.mean(dim=1, keepdim=True).expand_as(normalized)
            elif self.arm in ("no_spatial", "grouped_no_spatial"):
                mixed = torch.zeros_like(normalized)
            else:
                pool = self.pools[scale_index] if self.arm == "independent" else self.pools[0]
                power = ARM_SCHEDULES[self.arm][scale_index]
                positive = pool.positive_power(normalized, route, power)
                negative = pool.apply(normalized, route, sign=1)
                mixed = positive - negative
            residual = hidden + mixed
            outputs.append(residual + self.ffn(self.norm2(residual)))
        if topk is None:
            counts = torch.zeros(
                self.num_experts, dtype=torch.long, device=streams[0].device
            )
        else:
            counts = torch.bincount(topk.reshape(-1), minlength=self.num_experts)
        return outputs, counts


class ProposalMAGE(nn.Module):
    """MAGE proposal model; leaves the repository's original MAGE implementations untouched."""

    def __init__(
        self,
        model_args,
        arm="proposed",
        pdm_width=16,
        pdm_depth=2,
        moving_average=5,
        dropout=0.3,
    ):
        super().__init__()
        if arm not in ARM_SCHEDULES:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.node_num = model_args.node_num
        self.seq_len = model_args.seq_len
        self.horizon = model_args.horizon
        self.output_dim = model_args.output_dim
        model_dim = model_args.model_dim
        num_experts = model_args.recur_num
        topk = model_args.topk
        graph_dim = model_args.graph_dim
        input_dim = model_args.input_dim
        flow_dim = model_args.feature_dim

        prompt_args = SimpleNamespace(
            feature_dim=flow_dim,
            node_num=self.node_num,
            model_dim=model_dim,
            second=getattr(model_args, "second", 0),
            minute=getattr(model_args, "minute", 288),
            hour=getattr(model_args, "hour", 0),
            day=getattr(model_args, "day", 7),
            week=getattr(model_args, "week", 0),
            weekday=getattr(model_args, "weekday", 0),
            month=getattr(model_args, "month", 0),
            quarter=getattr(model_args, "quarter", 0),
            year=getattr(model_args, "year", 0),
        )
        self.prompt = mage_models.M.Prompt_Pool(prompt_args)
        if arm in ("proposed_no_pdm", "grouped_no_pdm"):
            self.temporal = LinearScaleEncoder(
                self.seq_len, flow_dim, model_dim
            )
        else:
            self.temporal = PDMScales(
                self.seq_len,
                flow_dim,
                model_dim,
                width=pdm_width,
                depth=pdm_depth,
                moving_average=moving_average,
            )
        self.group_projection = (
            nn.Linear(len(SCALES) * model_dim, model_dim) if arm in GROUPED_ARMS else None
        )
        stream_count = 1 if arm in GROUPED_ARMS else len(SCALES)
        self.blocks = nn.ModuleList(
            ProposalBlock(
                arm,
                model_dim,
                self.node_num,
                num_experts,
                topk,
                graph_dim,
                dropout,
            )
            for _ in range(3)
        )
        self.heads = nn.ModuleList(
            nn.Linear(model_dim, self.horizon * self.output_dim)
            for _ in range(stream_count)
        )
        self.num_experts = num_experts

    @property
    def graph_parameter_count(self):
        return sum(
            parameter.numel()
            for block in self.blocks
            for module in (block.router, block.pools)
            if module is not None
            for parameter in module.parameters()
        )

    @property
    def propagation_count(self):
        return self.positive_propagation_count + self.negative_propagation_count

    @property
    def positive_propagation_count(self):
        return 3 * sum(ARM_SCHEDULES[self.arm])

    @property
    def negative_propagation_count(self):
        schedule = ARM_SCHEDULES[self.arm]
        return 0 if self.arm in NO_GRAPH_ARMS else 3 * len(schedule)

    @property
    def applications_by_scale(self):
        schedule = ARM_SCHEDULES[self.arm]
        if self.arm in GROUPED_ARMS:
            return {"grouped": 3 * schedule[0], "negative": self.negative_propagation_count}
        result = {
            str(scale): 3 * int(power)
            for scale, power in zip(SCALES, schedule)
        }
        if self.arm not in NO_GRAPH_ARMS:
            result["negative"] = self.negative_propagation_count
        return result

    def forward(self, x, bias=None):
        if bias is None:
            bias = torch.zeros(
                3, self.num_experts, device=x.device, dtype=x.dtype
            )
        prompt = self.prompt(x)
        encoded = self.temporal(x)
        if self.arm in GROUPED_ARMS:
            fused = self.group_projection(torch.cat(encoded, dim=-1))
            streams = [fused + prompt]
        else:
            streams = [value + prompt for value in encoded]
        counts = []
        for block, block_bias in zip(self.blocks, bias):
            streams, block_counts = block(streams, block_bias)
            counts.append(block_counts)
        prediction = sum(
            head(hidden) for head, hidden in zip(self.heads, streams)
        )
        prediction = prediction.reshape(
            x.shape[0], self.node_num, self.horizon, self.output_dim
        ).transpose(1, 2)
        if self.training:
            return prediction, torch.stack(counts)
        return prediction


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())

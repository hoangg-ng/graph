#!/usr/bin/env python3
"""Install isolated Grouped ablation arms and a safe scheduler in MAGE_MS.

Targets the Jiwon-work source snapshot checked on 2026-11-07.
Run from the MAGE_MS directory of a clean Git worktree.
No training is started and no existing evaluation results are touched.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def apply_edits(content: str, edits, name: str) -> str:
    for number, (old, new, expected) in enumerate(edits, 1):
        count = content.count(old)
        if count != expected:
            raise RuntimeError(
                f"{name}: edit {number} expects {expected} matches, found {count}. "
                "Stop rather than modifying a changed source version."
            )
        content = content.replace(old, new)
    return content


PROPOSAL_EDITS = [
    (
        '    "grouped": (1,),\n',
        '    "grouped": (1,),\n'
        '    "grouped_no_pdm": (1,),\n'
        '    "grouped_no_spatial": (0,),\n'
        '    "grouped_uniform": (0,),\n',
        1,
    ),
    (
        'ARMS = tuple(ARM_SCHEDULES)\n',
        'ARMS = tuple(ARM_SCHEDULES)\n'
        'GROUPED_ARMS = ("grouped", "grouped_no_pdm", "grouped_no_spatial", "grouped_uniform")\n'
        'NO_GRAPH_ARMS = ("uniform", "no_spatial", "grouped_no_spatial", "grouped_uniform")\n',
        1,
    ),
    ('if arm in ("uniform", "no_spatial"):', 'if arm in NO_GRAPH_ARMS:', 1),
    ('self.arm in ("uniform", "no_spatial")', 'self.arm in NO_GRAPH_ARMS', 2),
    ('if self.arm == "uniform":', 'if self.arm in ("uniform", "grouped_uniform"):', 1),
    ('elif self.arm == "no_spatial":', 'elif self.arm in ("no_spatial", "grouped_no_spatial"):', 1),
    ('if arm == "proposed_no_pdm":', 'if arm in ("proposed_no_pdm", "grouped_no_pdm"):', 1),
    ('if arm == "grouped"', 'if arm in GROUPED_ARMS', 2),
    ('if self.arm == "grouped":', 'if self.arm in GROUPED_ARMS:', 2),
    ('if self.arm not in ("uniform", "no_spatial"):', 'if self.arm not in NO_GRAPH_ARMS:', 1),
    (
        'return {"grouped": 3, "negative": self.negative_propagation_count}',
        'return {"grouped": 3 * schedule[0], "negative": self.negative_propagation_count}',
        1,
    ),
]

SCHEDULER_EDITS = [
    (
        'FOLLOWUP_ARMS = ("mage_features", "no_spatial", "grouped", "proposed_no_pdm")',
        'FOLLOWUP_ARMS = ("grouped", "grouped_no_pdm", "grouped_no_spatial", "grouped_uniform")',
        1,
    ),
    ('"study": "followup96",', '"study": "ablation96",\n        "validation_only": bool(args.validation_only),', 1),
    ('"followup_manifest.json"', '"ablation_manifest.json"', 1),
    ('            "recipe_digest",', '            "validation_only",\n            "recipe_digest",', 1),
    (
        '    return options\n',
        '    if args.validation_only:\n'
        '        options.append("--validation-only")\n'
        '    return options\n',
        1,
    ),
    (
        '    parser.add_argument("--datasets",',
        '    parser.add_argument("--validation-only", action="store_true")\n'
        '    parser.add_argument("--datasets",',
        1,
    ),
    (
        'job: run_eval.expected_signature(args, job, code_digest, data_digests)',
        'job: train_eval.run_identity(\n'
        '            train_eval.make_parser().parse_args(\n'
        '                _worker_args(args, job, _result_path(args, job))\n'
        '            ),\n'
        '            source_hash=code_digest,\n'
        '            dataset_hash=data_digests[job.dataset],\n'
        '        )["signature"]',
        1,
    ),
    ('default=ROOT / "results/eval_followup96"', 'default=ROOT / "results/ablation_val96"', 1),
    ('default=ROOT / "logs/eval_followup96"', 'default=ROOT / "logs/ablation_val96"', 1),
]

TEST_SOURCE = dedent('''\
    """CPU checks for the Grouped ablation arms."""
    import unittest
    from unittest.mock import patch

    import torch

    from proposal_model import ExpertPool, LinearScaleEncoder, ProposalMAGE
    from test_proposal import make_args, make_input


    COUNTS = {
        "grouped": (3, 3),
        "grouped_no_pdm": (3, 3),
        "grouped_no_spatial": (0, 0),
        "grouped_uniform": (0, 0),
    }


    class GroupedAblationTests(unittest.TestCase):
        def test_output_and_graph_counts(self):
            x = make_input(batch=2)
            for arm, (positive, negative) in COUNTS.items():
                with self.subTest(arm=arm):
                    torch.manual_seed(17)
                    model = ProposalMAGE(
                        make_args(), arm, pdm_width=8, pdm_depth=1, dropout=0
                    ).eval()
                    with torch.inference_mode():
                        output = model(x)
                    self.assertEqual(tuple(output.shape), (2, 12, 7, 1))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(len(model.heads), 1)
                    self.assertIsNotNone(model.group_projection)
                    self.assertEqual(len(model.temporal(x)), 4)
                    self.assertEqual(model.positive_propagation_count, positive)
                    self.assertEqual(model.negative_propagation_count, negative)
                    self.assertEqual(model.propagation_count, positive + negative)

        def test_pdm_removal_and_graph_removal(self):
            no_pdm = ProposalMAGE(make_args(), "grouped_no_pdm", pdm_width=8, pdm_depth=1)
            self.assertIsInstance(no_pdm.temporal, LinearScaleEncoder)
            self.assertEqual(len(no_pdm.temporal.encoders), 4)
            self.assertEqual(len(no_pdm.blocks[0].pools), 1)

            for arm in ("grouped_no_spatial", "grouped_uniform"):
                with self.subTest(arm=arm):
                    model = ProposalMAGE(make_args(), arm, pdm_width=8, pdm_depth=1).eval()
                    self.assertTrue(all(block.router is None for block in model.blocks))
                    self.assertTrue(all(len(block.pools) == 0 for block in model.blocks))
                    self.assertEqual(model.graph_parameter_count, 0)
                    with patch.object(ExpertPool, "apply", side_effect=AssertionError("graph called")):
                        with torch.inference_mode():
                            model(make_input(batch=1))


    if __name__ == "__main__":
        unittest.main()
''')

REPORT_SOURCE = dedent('''\
    """Report matched Grouped ablation validation or test MAE with provenance checks."""
    from __future__ import annotations

    import argparse
    import json
    from pathlib import Path
    from statistics import mean, stdev

    ARMS = ("grouped", "grouped_no_pdm", "grouped_no_spatial", "grouped_uniform")
    DATASETS = ("PEMS04", "PEMS08")
    SKIP_CONFIG = {"dataset", "arm", "seed", "device"}


    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-dir", type=Path, required=True)
        parser.add_argument("--metric", choices=("best_val_mae", "test_mae"), default="best_val_mae")
        parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
        parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
        args = parser.parse_args()
        is_validation = args.metric == "best_val_mae"
        required_protocol = (
            "96to12_v1_validation_only" if is_validation
            else "96to12_v1_window_local_features"
        )
        lines = [f"# Grouped ablation ({args.metric})", "",
                 "Exploratory unless all planned five seeds and the full test protocol are completed.", ""]
        for dataset in DATASETS:
            measurements = {}
            provenance = set()
            for arm in args.arms:
                measurements[arm] = {}
                for seed in args.seeds:
                    path = args.results_dir / f"{dataset}_{arm}_s{seed}.json"
                    if not path.is_file():
                        continue
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("completed") is not True or record.get("protocol") != required_protocol:
                        raise ValueError(f"Invalid or wrong protocol: {path}")
                    if (record.get("dataset"), record.get("arm"), record.get("seed")) != (dataset, arm, seed):
                        raise ValueError(f"Wrong dataset/arm/seed in {path}")
                    cfg = record.get("config", {})
                    if bool(cfg.get("validation_only")) != is_validation:
                        raise ValueError(f"Validation flag mismatch: {path}")
                    config_key = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in cfg.items() if k not in SKIP_CONFIG))
                    provenance.add((record.get("code_digest"), record.get("data_digest"), config_key))
                    value = record.get(args.metric)
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Missing {args.metric}: {path}")
                    measurements[arm][seed] = float(value)
            if len(provenance) > 1:
                raise ValueError(f"{dataset}: different code, data, or training config; cannot compare")
            lines += [f"## {dataset}", "", "| Arm | Completed | MAE mean +/- SD | Paired improvement vs Grouped | Paired wins |",
                      "|---|---:|---:|---:|---:|"]
            baseline = measurements.get("grouped", {})
            for arm, vals in measurements.items():
                ordered = sorted(vals.values())
                maetxt = f"{mean(ordered):.4f} +/- {stdev(ordered):.4f}" if len(ordered) > 1 else (f"{ordered[0]:.4f}" if ordered else "n/a")
                common = set(baseline) & set(vals)
                if arm != "grouped" and common:
                    baseline_mean = mean(baseline[s] for s in common)
                    candidate_mean = mean(vals[s] for s in common)
                    pct = 100 * (baseline_mean - candidate_mean) / baseline_mean
                    wins = sum(vals[s] < baseline[s] for s in common)
                    comp, wintext = f"{pct:+.2f}%", f"{wins}/{len(common)}"
                else:
                    comp, wintext = "-", "-"
                lines.append(f"| {arm} | {len(vals)}/{len(args.seeds)} | {maetxt} | {comp} | {wintext} |")
            lines.append("")
        report_path = args.results_dir / f"ablation_{args.metric}_summary.md"
        report_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        print("\\n".join(lines))
        print(f"Written: {report_path}")


    if __name__ == "__main__":
        main()
''')


def main():
    source = Path("proposal_model.py")
    scheduler_source = Path("run_followup_eval.py")
    if not source.is_file() or not scheduler_source.is_file():
        raise SystemExit("Run from the MAGE_MS directory containing proposal_model.py and run_followup_eval.py")
    outputs = [Path("run_ablation_eval.py"), Path("test_grouped_ablation.py"), Path("report_grouped_ablation.py")]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise SystemExit("Refusing to overwrite: " + ", ".join(existing))

    patched_proposal = apply_edits(source.read_text(encoding="utf-8"), PROPOSAL_EDITS, source.name)
    patched_scheduler = apply_edits(scheduler_source.read_text(encoding="utf-8"), SCHEDULER_EDITS, scheduler_source.name)

    # Validate all four code strings before writing anything.
    import ast
    ast.parse(patched_proposal, filename=str(source))
    ast.parse(patched_scheduler, filename=str(outputs[0]))
    ast.parse(TEST_SOURCE, filename=str(outputs[1]))
    ast.parse(REPORT_SOURCE, filename=str(outputs[2]))

    source.write_text(patched_proposal, encoding="utf-8")
    for dest, content in zip(outputs, (patched_scheduler, TEST_SOURCE, REPORT_SOURCE)):
        dest.write_text(content, encoding="utf-8")
    for p in [source, *outputs]:
        print("WROTE", p)
    print("Run: python -m unittest test_grouped_ablation test_proposal test_complexity")


if __name__ == "__main__":
    main()

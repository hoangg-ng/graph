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
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Written: {report_path}")


if __name__ == "__main__":
    main()

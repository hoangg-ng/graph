"""Make screenshot-style MAGE-vs-Proposed tables from matched training results.

Read-only. Require matching code, data, 96->12 protocol, and training recipe.
Use only completed train_eval.py result JSONs; never use tune/smoke logs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics

ARMS = (
    ("mage", "Baseline: MAGE", "original spatial mixer"),
    ("shared", "Shared", "(1, 1, 1, 1)"),
    ("proposed", "Proposed", "(1, 2, 4, 8)"),
    ("reversed", "Reversed", "(8, 4, 2, 1)"),
    ("independent", "Independent", "one operator per scale"),
    ("uniform", "Uniform", "global mean"),
)
EXPECTED_SEEDS = tuple(range(5))
PROTOCOL = "96to12_v1_window_local_features"
# Exclude dataset, arm, seed, and execution environment from cross-arm comparison.
COMMON_CONFIG_KEYS = (
    "seq_len", "horizon", "epochs", "patience", "lr", "wd", "bs",
    "lb", "lb_lr", "dropout", "model_dim", "experts", "topk",
    "graph_dim", "pdm_width", "pdm_depth", "moving_average", "tf32",
    "validation_only",
)


@dataclass(frozen=True)
class Record:
    dataset: str
    arm: str
    seed: int
    mae: float
    rmse: float
    mape: float
    epochs_run: int
    code_digest: str
    data_digest: str
    config: dict


def read_records(folder: Path, dataset: str):
    records = {}
    missing = []
    for arm, _, _ in ARMS:
        for seed in EXPECTED_SEEDS:
            path = folder / f"{dataset}_{arm}_s{seed}.json"
            if not path.exists():
                missing.append(path.name)
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read {path}: {exc}") from exc
            if raw.get("completed") is not True:
                raise ValueError(f"Not a completed run: {path}")
            if raw.get("protocol") != PROTOCOL:
                raise ValueError(f"Unexpected data/evaluation protocol: {path}")
            if (raw.get("dataset"), raw.get("arm"), raw.get("seed")) != (dataset, arm, seed):
                raise ValueError(f"Filename and result metadata disagree: {path}")
            if not raw.get("code_digest") or not raw.get("data_digest"):
                raise ValueError(f"Missing source/data provenance: {path}")
            cfg = raw.get("config")
            if not isinstance(cfg, dict):
                raise ValueError(f"Missing training configuration: {path}")
            if cfg.get("validation_only") is not False:
                raise ValueError(f"Validation-only trial cannot be a test MAE: {path}")
            records[(arm, seed)] = Record(
                dataset=dataset, arm=arm, seed=seed,
                mae=float(raw["test_mae"]),
                rmse=float(raw["test_rmse"]),
                mape=float(raw["test_mape"]),
                epochs_run=int(raw["epochs_run"]),
                code_digest=raw["code_digest"],
                data_digest=raw["data_digest"],
                config=cfg,
            )
    return records, missing


def validate_matched(records, dataset):
    if not records:
        return
    codes = {r.code_digest for r in records.values()}
    data = {r.data_digest for r in records.values()}
    configs = {
        tuple((key, r.config.get(key)) for key in COMMON_CONFIG_KEYS)
        for r in records.values()
    }
    problems = []
    if len(codes) != 1:
        problems.append("different source revisions")
    if len(data) != 1:
        problems.append("different prepared datasets or splits")
    if len(configs) != 1:
        problems.append("different training hyperparameters")
    if problems:
        raise ValueError(f"{dataset}: NOT COMPARABLE: {', '.join(problems)}. "
                         "Do not average these runs. Keep each protocol in its own results directory.")
    config = next(iter(records.values())).config
    if config.get("seq_len") != 96 or config.get("horizon") != 12:
        raise ValueError(f"{dataset}: expected exactly 96 -> 12")


def mean_sd(values):
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def format_mean_sd(values):
    mean, sd = mean_sd(values)
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:.3f} (1 run)"
    return f"{mean:.3f} +/- {sd:.3f}"


def render_dataset(folder, dataset):
    records, missing = read_records(folder, dataset)
    validate_matched(records, dataset)
    lines = [
        f"## Results for each variant ({dataset}, 96 -> 12, masked test MAE)",
        "",
        "| Variant | Spatial mixing schedule | Runs | Test MAE (mean +/- SD) |",
        "|---|---|---:|---:|",
    ]
    csv_rows = []
    raw_rows = []
    for arm, name, schedule in ARMS:
        subset = [records[(arm, seed)] for seed in EXPECTED_SEEDS if (arm, seed) in records]
        values = [r.mae for r in subset]
        maemean, maesd = mean_sd(values)
        count = len(values)
        completed = count == len(EXPECTED_SEEDS)
        status = f"{count}/5" + ("" if completed else " (partial)")
        display = format_mean_sd(values)
        lines.append(f"| {name} | {schedule} | {status} | {display} |")
        csv_rows.append({
            "dataset": dataset, "arm": arm, "variant": name,
            "schedule": schedule, "n": count, "expected": 5,
            "complete": completed, "mae_mean": maemean, "mae_sd": maesd,
            "rmse_mean": mean_sd([r.rmse for r in subset])[0],
            "mape_mean": mean_sd([r.mape for r in subset])[0],
            "epochs_run_mean": mean_sd([r.epochs_run for r in subset])[0],
        })
        for r in subset:
            raw_rows.append({
                "dataset": dataset, "arm": r.arm, "seed": r.seed,
                "test_mae": r.mae, "test_rmse": r.rmse, "test_mape": r.mape,
                "epochs_run": r.epochs_run,
            })
    lines.extend(["", "### Paired comparison with MAGE", "",
                  "| Comparison | Paired seeds | Candidate wins | Mean MAE improvement vs MAGE |",
                  "|---|---:|---:|---:|"])
    for arm, name, _ in ARMS[1:]:
        seed_pairs = [(records[("mage", s)], records[(arm, s)])
                      for s in EXPECTED_SEEDS if ("mage", s) in records and (arm, s) in records]
        n = len(seed_pairs)
        wins = sum(candidate.mae < base.mae for base, candidate in seed_pairs)
        ref_mean = mean_sd([base.mae for base, _ in seed_pairs])[0]
        can_mean = mean_sd([candidate.mae for _, candidate in seed_pairs])[0]
        gain = (100 * (ref_mean - can_mean) / ref_mean) if ref_mean else None
        gain_str = f"{gain:+.2f}%" if gain is not None else "-"
        lines.append(f"| {name} vs MAGE | {n}/5 | {wins}/{n} | {gain_str} |")
    lines.extend(["", f"Missing runs: **{len(missing)}**. ",
                  "A positive percentage indicates lower MAE for the candidate. ",
                  "Rows with fewer than five seeds are preliminary, not final five-seed results.",
                  ""])  # End-of-file newline included by caller.
    return "\n".join(lines), csv_rows, raw_rows, missing


def csv_write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "eval_proposal96")
    p.add_argument("--datasets", nargs="+", choices=("PEMS04", "PEMS08"), default=("PEMS04", "PEMS08"))
    p.add_argument("--out-dir", type=Path, help="Default: results directory")
    args = p.parse_args()
    out = args.out_dir or args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    all_sections = ["# Matched MAGE versus multiscale spatial mixing evaluation", "",
                    "All methods use identical 96 -> 12 windows, chronological data split, and five seeds (0..4).",
                    "An independent MAGE reimplementation with original backbone and linear temporal encoder is the baseline;",
                    "it is not the original paper's 12 -> 12 benchmark result.", ""]
    all_aggregates, all_raw, all_missing = [], [], []
    for ds in args.datasets:
        section, agg, raw, missing = render_dataset(args.results_dir, ds)
        all_sections.extend([section, ""])
        all_aggregates.extend(agg)
        all_raw.extend(raw)
        all_missing.extend(missing)
        print("\n" + section)
    (out / "matched_mae_table.md").write_text("\n".join(all_sections), encoding="utf-8")
    csv_write(out / "matched_mae_summary.csv", all_aggregates,
              ("dataset", "arm", "variant", "schedule", "n", "expected", "complete", "mae_mean", "mae_sd",
               "rmse_mean", "mape_mean", "epochs_run_mean"))
    csv_write(out / "matched_mae_per_seed.csv", all_raw,
              ("dataset", "arm", "seed", "test_mae", "test_rmse", "test_mape", "epochs_run"))
    (out / "matched_mae_missing.txt").write_text("\n".join(all_missing) + ("\n" if all_missing else ""), encoding="utf-8")
    print(f"\nWrote {out / 'matched_mae_table.md'}")
    print(f"Wrote {out / 'matched_mae_summary.csv'}")
    print(f"Wrote {out / 'matched_mae_per_seed.csv'}")
    print(f"Missing: {len(all_missing)} runs")
    if all_missing:
        print("WARNING: Some rows are incomplete. Do not report partial five-seed means as final results.")


if __name__ == "__main__":
    main()

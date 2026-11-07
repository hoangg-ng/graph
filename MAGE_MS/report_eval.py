"""Aggregate core MAGE + TimeMixer experiments and apply the prespecified decision rule.

Only results from train_eval.py with completed=true and matching data/code
fingerprints are treated as comparable.  Read-only; never re-trains models.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

ARMS = ("shared", "proposed", "reversed", "independent", "uniform", "mage", "mage_features")
PRIMARY = ("shared", "proposed", "reversed")
FIELDS = (
    "dataset", "arm", "seeds", "n", "mae_mean", "mae_sd", "rmse_mean", "mape_mean",
    "mae_15m", "mae_30m", "mae_60m", "unmasked_mae", "unmasked_rmse",
    "excluded_fraction_pct", "params", "graph_params", "train_s_per_epoch",
    "peak_mem_mb", "inference_ms_per_sample",
)


def expected_jobs(profile="full"):
    from run_eval import planned_jobs
    return planned_jobs(profile)


def _mean(xs):
    x = [float(v) for v in xs if v is not None]
    return statistics.mean(x) if x else None


def _sd(xs):
    x = [float(v) for v in xs if v is not None]
    return statistics.stdev(x) if len(x) > 1 else 0.0 if x else None


def _fmt(val, digits=3):
    return f"{val:.{digits}f}" if val is not None else "—"


def load_results(root: Path, profile: str):
    results = {}
    issues, missing = [], []
    for job in expected_jobs(profile):
        path = root / f"{job.name}.json"
        if not path.is_file():
            missing.append(job.name)
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            issues.append(f"Could not read {path}: {e}")
            continue
        if not data.get("completed"):
            issues.append(f"Unverified result: {path}")
            continue
        if (data.get("dataset"), data.get("arm"), data.get("seed")) != (
            job.dataset, job.arm, job.seed
        ):
            issues.append(f"Misidentified result: {path}")
            continue
        if data.get("protocol") != "96to12_v1_window_local_features":
            issues.append(f"Unexpected protocol: {path}: {data.get('protocol')}")
            continue
        results[(job.dataset, job.arm, job.seed)] = data
    return results, missing, issues


def analyze_fingerprints(results):
    issues = []
    codes = {v.get("code_digest") for v in results.values()}
    if len(codes) > 1 or None in codes:
        issues.append("Multiple source-code revisions found; results must not be pooled as one study")
    for dataset in ("PEMS04", "PEMS08"):
        local = [r for r in results.values() if r["dataset"] == dataset]
        data_sha = {r.get("data_digest") for r in local}
        if len(data_sha) > 1 or None in data_sha:
            issues.append(f"Different prepared data/split files across {dataset} arms")
    # Compare common training hyperparameters; the arm, dataset and seed are
    # intentionally excluded, since they are experimental variables.
    common_keys = (
        "seq_len", "horizon", "epochs", "patience", "lr", "wd", "bs", "lb", "lb_lr",
        "dropout", "model_dim", "experts", "topk", "graph_dim", "pdm_width",
        "pdm_depth", "moving_average", "tf32",
    )
    configs = {
        tuple((key, r.get("config", {}).get(key)) for key in common_keys)
        for r in results.values()
    }
    if len(configs) > 1:
        issues.append("Different training recipes detected across experiment arms")
    return issues


def rows_for_table(results):
    table = []
    for dataset in ("PEMS04", "PEMS08"):
        for arm in ARMS:
            grouped = sorted(
                (r for (ds, a, _), r in results.items() if ds == dataset and a == arm),
                key=lambda r: r["seed"],
            )
            if not grouped:
                continue
            row = {
                "dataset": dataset, "arm": arm,
                "seeds": ",".join(str(r["seed"]) for r in grouped), "n": len(grouped),
                "mae_mean": _mean(r.get("test_mae") for r in grouped),
                "mae_sd": _sd(r.get("test_mae") for r in grouped),
                "rmse_mean": _mean(r.get("test_rmse") for r in grouped),
                "mape_mean": _mean(r.get("test_mape") for r in grouped),
                "mae_15m": _mean(r.get("horizon_15m", [None])[0] for r in grouped),
                "mae_30m": _mean(r.get("horizon_30m", [None])[0] for r in grouped),
                "mae_60m": _mean(r.get("horizon_60m", [None])[0] for r in grouped),
                "unmasked_mae": _mean(r.get("test_unmasked_mae") for r in grouped),
                "unmasked_rmse": _mean(r.get("test_unmasked_rmse") for r in grouped),
                "excluded_fraction_pct": 100 * _mean(r.get("test_excluded_fraction") for r in grouped),
                "params": round(_mean(r.get("params") for r in grouped)),
                "graph_params": _mean(r.get("graph_params") for r in grouped),
                "train_s_per_epoch": _mean(r.get("train_s_per_epoch") for r in grouped),
                "peak_mem_mb": _mean(r.get("peak_mem_mb") for r in grouped),
                "inference_ms_per_sample": _mean(r.get("test_inference_ms_per_sample") for r in grouped),
            }
            table.append(row)
    return table


def paired(results, ds, ref_arm, alternative="proposed", seeds=range(5)):
    detail = []
    for seed in seeds:
        first = results.get((ds, ref_arm, seed))
        second = results.get((ds, alternative, seed))
        if first is None or second is None:
            continue
        a, b = first["test_mae"], second["test_mae"]
        detail.append({
            "dataset": ds, "reference": ref_arm, "candidate": alternative,
            "seed": seed, "ref_mae": a, "candidate_mae": b,
            "delta_mae": b - a, "improvement_pct": 100 * (a - b) / a if a else None,
            "candidate_wins": bool(b < a),
        })
    ref_vals = [v["ref_mae"] for v in detail]
    candidate_vals = [v["candidate_mae"] for v in detail]
    ref_avg, candidate_avg = _mean(ref_vals), _mean(candidate_vals)
    return {
        "n": len(detail), "details": detail, "paired_wins": sum(v["candidate_wins"] for v in detail),
        "ref_mean": ref_avg, "candidate_mean": candidate_avg,
        "relative_mean_improvement_pct": 100 * (ref_avg - candidate_avg) / ref_avg
        if ref_avg else None,
    }


def ordering_decision(results, validity_issues):
    against_shared = paired(results, "PEMS04", "shared")
    against_reversed = paired(results, "PEMS04", "reversed")
    missing = any(x["n"] != 5 for x in (against_shared, against_reversed))
    if validity_issues:
        outcome = "INVALID_COMPARISON"
    elif missing:
        outcome = "INCOMPLETE"
    else:
        support = all(
            x["paired_wins"] >= 4 and x["relative_mean_improvement_pct"] >= 1.0
            for x in (against_shared, against_reversed)
        )
        outcome = "SUPPORTED" if support else "NOT_SUPPORTED_BY_PRESPECIFIED_RULE"
    return outcome, against_shared, against_reversed


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_report(results, missing, issues, profile):
    table = rows_for_table(results)
    issues.extend(analyze_fingerprints(results))
    outcome, shared, reversed_result = ordering_decision(results, issues)
    lines = [
        "# MAGE + TimeMixer: experiment results", "",
        f"Profile: `{profile}`. Completed result files: **{len(results)}**. "
        f"Missing runs: **{len(missing)}**. Validation warnings: **{len(issues)}**.", "",
        "## Masked test MAE and resources", "",
        "| Dataset | Arm | Seeds | MAE (mean ± sd) | RMSE | 15/30/60min MAE | Params | Time/epoch (s) | GPU peak (MiB) |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in table:
        h = " / ".join(_fmt(r[k]) for k in ("mae_15m", "mae_30m", "mae_60m"))
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {r['n']} | "
            f"{_fmt(r['mae_mean'])} ± {_fmt(r['mae_sd'])} | {_fmt(r['rmse_mean'])} | {h} | "
            f"{r['params']:,} | {_fmt(r['train_s_per_epoch'],1)} | {_fmt(r['peak_mem_mb'],0)} |"
        )
    lines.extend(["", "## Main hypothesis (PEMS04, five paired seeds)", "",
                  "Prespecified support requires **≥4/5 wins** and **≥1% lower mean MAE** "
                  "against **both Shared and Reversed**.", "",
                  f"Decision: **{outcome}**", "",
                  "| Comparison | Pairs | Proposed wins | Relative mean MAE improvement |",
                  "|---|---:|---:|---:|"])
    for name, r in (("Proposed vs Shared", shared), ("Proposed vs Reversed", reversed_result)):
        lines.append(f"| {name} | {r['n']}/5 | {r['paired_wins']}/5 | "
                     f"{_fmt(r['relative_mean_improvement_pct'],2)}% |")
    lines.extend(["", "## Integrity checks", ""])
    if issues:
        for issue in issues:
            lines.append(f"- **WARNING:** {issue}")
    else:
        lines.append("All collected results match one source version, one dataset split per dataset, "
                     "and one controlled training recipe.")
    if missing:
        lines.extend(["", "## Not yet completed", ""])
        lines.extend(f"- `{name}`" for name in missing)
    lines.extend(["", "## Scope caveat", "",
                  "The 43-run launcher covers the five core arms, common-protocol MAGE, "
                  "and MAGE with engineered features. It does **not** by itself run the external "
                  "official TimeMixer or STID implementations, or MAGE's original 12→12 protocol.", ""])
    return "\n".join(lines), table, (shared, reversed_result), outcome, issues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "eval_proposal96")
    p.add_argument("--profile", choices=("smoke", "pilot", "core", "full"), default="full")
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()
    output = args.out_dir or args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    results, missing, issues = load_results(args.results_dir, args.profile)
    report, table, paired_results, decision, issues = make_report(results, missing, issues, args.profile)
    (output / "summary.md").write_text(report, encoding="utf-8")
    write_csv(output / "summary.csv", FIELDS, table)
    details = [entry for result in paired_results for entry in result["details"]]
    write_csv(output / "paired_comparisons.csv", (
        "dataset", "reference", "candidate", "seed", "ref_mae", "candidate_mae",
        "delta_mae", "improvement_pct", "candidate_wins"), details)
    write_csv(output / "missing_jobs.csv", ("job",), [{"job": v} for v in missing])
    (output / "decision.json").write_text(json.dumps({
        "decision": decision, "completed": len(results), "missing": missing,
        "integrity_issues": issues, "paired": [
            {key: val for key, val in x.items() if key != "details"} for x in paired_results
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(results)} complete, {len(missing)} missing; decision={decision}")
    print(f"Report: {output / 'summary.md'}")
    for item in issues:
        print("WARNING:", item)


if __name__ == "__main__":
    main()

"""Strict, source-backed validation/test and cost comparisons for Grouped/Slim.

Validation mode reads best_val_mae only; it never substitutes test metrics.
Cost fields require saved-checkpoint profiles of the matching data/code version.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from report_matched_eval import COMMON_CONFIG_KEYS
from run_unified_slim_eval import ARMS, OPTIONAL_ARMS

ROOT = Path(__file__).resolve().parent
BASE_COMMIT = "b5d4d818ff1e0673396ec1c74dc59cb32c2f7e55"
SLIM_COMMIT = "76a7d6717282a68840d18451038ac632c8ad8fc4"


def stats(values):
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None) if values else (None, None)


def pct(candidate, baseline):
    return 100 * (candidate - baseline) / baseline if candidate is not None and baseline else None


def read_runs(directory, datasets, arms, seeds, validation_only):
    records, missing = {}, []
    protocol = "96to12_v1_validation_only" if validation_only else "96to12_v1_window_local_features"
    metric = "best_val_mae" if validation_only else "test_mae"
    for dataset in datasets:
        for arm in arms:
            for seed in seeds:
                path = directory / f"{dataset}_{arm}_s{seed}.json"
                if not path.exists():
                    missing.append(path.name)
                    continue
                row = json.loads(path.read_text())
                if row.get("completed") is not True:
                    missing.append(path.name)
                    continue
                if ((row.get("dataset"), row.get("arm"), row.get("seed")) != (dataset, arm, seed)
                        or type(row.get("seed")) is not int):
                    raise ValueError(f"Filename/identity mismatch: {path}")
                if row.get("protocol") != protocol:
                    raise ValueError(f"Wrong evaluation protocol: {path}")
                config = row.get("config", {})
                if any(k not in config for k in COMMON_CONFIG_KEYS):
                    raise ValueError(f"Missing common configuration: {path}")
                if (config.get("dataset"), config.get("arm"), config.get("seed")) != (dataset, arm, seed):
                    raise ValueError(f"Configuration identity mismatch: {path}")
                if config["validation_only"] is not validation_only or config["seq_len"] != 96 or config["horizon"] != 12:
                    raise ValueError(f"Inconsistent 96-to-12 evaluation settings: {path}")
                for field in (metric, "params"):
                    value = row.get(field)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid {field}: {path}")
                if not row.get("data_digest") or not row.get("code_digest"):
                    raise ValueError(f"Missing provenance: {path}")
                if not validation_only:
                    horizons = row.get("per_horizon", [])
                    if len(horizons) != 12 or any(not isinstance(values, list) or len(values) != 3 for values in horizons):
                        raise ValueError(f"Missing horizon metrics: {path}")
                    values = [row.get("test_rmse"), row.get("test_mape")] + [h[0] for h in horizons]
                    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                        raise ValueError(f"Invalid test/horizon metrics: {path}")
                records[(dataset, arm, seed)] = row
    if len({r["code_digest"] for r in records.values()}) > 1:
        raise ValueError("Mixed source versions; use one new-code run directory")
    configurations = {json.dumps({k: r["config"][k] for k in COMMON_CONFIG_KEYS}, sort_keys=True)
                      for r in records.values()}
    if len(configurations) > 1:
        raise ValueError("Different training/evaluation settings")
    for dataset in datasets:
        if len({r["data_digest"] for k, r in records.items() if k[0] == dataset}) > 1:
            raise ValueError(f"Mixed prepared data/splits: {dataset}")
    return records, missing


def read_profiles(directory, records):
    profiles = {}
    if directory is None:
        return profiles
    for path in sorted(directory.glob("profile_*.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "completed":
            continue
        key = row.get("dataset"), row.get("arm"), row.get("seed")
        if key not in records:
            continue
        result = records[key]
        if row.get("data_digest") != result["data_digest"]:
            raise ValueError(f"Profile/MAE data mismatch: {path}")
        if row.get("checkpoint_code_digest") != result["code_digest"] or row.get("profile_code_digest") != result["code_digest"]:
            raise ValueError(f"Profile/MAE source mismatch: {path}")
        if not result.get("signature") or row.get("checkpoint_signature") != result["signature"]:
            raise ValueError(f"Profile uses a different checkpoint configuration: {path}")
        if key in profiles:
            raise ValueError(f"Duplicate cost profile: {path}")
        profiles[key] = row
    return profiles


def read_validation_metrics(directory, records):
    metrics = {}
    if directory is None:
        return metrics
    for key, result in records.items():
        dataset, arm, seed = key
        path = directory / f"validation_{dataset}_{arm}_s{seed}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if (row.get("completed") is not True
                or row.get("protocol") != "96to12_v1_validation_best_checkpoint_metrics"
                or (row.get("dataset"), row.get("arm"), row.get("seed")) != key):
            raise ValueError(f"Invalid validation metric file: {path}")
        if any(row.get(k) != result.get(k) for k in ("signature", "data_digest", "code_digest")):
            raise ValueError(f"Validation metric provenance mismatch: {path}")
        horizons = row.get("per_horizon", [])
        if len(horizons) != 12 or any(len(h) != 3 for h in horizons):
            raise ValueError(f"Missing validation horizon metrics: {path}")
        values = [row.get(k) for k in ("val_mae", "val_rmse", "val_mape")] + [v for h in horizons for v in h]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid validation metrics: {path}")
        if not math.isclose(row["val_mae"], result["best_val_mae"], rel_tol=1e-5, abs_tol=1e-4):
            raise ValueError(f"Validation metrics do not match best checkpoint: {path}")
        metrics[key] = row
    return metrics


def summarize(records, profiles, datasets, arms, seeds, validation_only, validation_metrics=None):
    validation_metrics = validation_metrics or {}
    metric = "best_val_mae" if validation_only else "test_mae"
    rows, comparisons = [], []
    cost_fields = {
        "counted_gflops": "gflops_counted_per_forward",
        "latency_ms_batch1": "latency_ms_batch1_median",
        "latency_ms_batch64": "latency_ms_batch64_median",
        "throughput_batch64": "throughput_samples_per_s_batch64",
        "peak_mib_batch64": "inference_peak_allocated_mib_batch64",
        "graph_applications": "total_graph_applications",
        "slim_mean_applications": "slim_mean_applications",
    }
    for dataset in datasets:
        current = {}
        for arm in arms:
            runs = [records[(dataset, arm, s)] for s in seeds if (dataset, arm, s) in records]
            costs = [profiles[(dataset, arm, s)] for s in seeds if (dataset, arm, s) in profiles]
            val_rows = [validation_metrics[(dataset, arm, s)] for s in seeds if (dataset, arm, s) in validation_metrics]
            params = {r["params"] for r in runs}
            if len(params) > 1:
                raise ValueError(f"Changing parameter count across seeds: {dataset}/{arm}")
            row = {
                "dataset": dataset, "arm": arm, "metric": metric,
                "completed_seeds": len(runs), "expected_seeds": len(seeds),
                "status": "exploratory_validation" if validation_only else ("complete_5seed" if len(runs) == 5 else "preliminary"),
                "mae_mean": stats([r[metric] for r in runs])[0],
                "mae_sd": stats([r[metric] for r in runs])[1],
                "params": next(iter(params), None), "profile_seed_count": len(costs),
                "validation_metric_seed_count": len(val_rows),
                "rmse_mean": stats([r["val_rmse"] for r in val_rows])[0] if validation_only else stats([r["test_rmse"] for r in runs])[0],
                "mape_mean": stats([r["val_mape"] for r in val_rows])[0] if validation_only else stats([r["test_mape"] for r in runs])[0],
            }
            for minutes, horizon in ((15, 2), (30, 5), (60, 11)):
                row[f"mae_{minutes}m"] = stats([r["per_horizon"][horizon][0] for r in (val_rows if validation_only else runs)])[0]
            for name, source in cost_fields.items():
                row[name] = stats([r[source] for r in costs if isinstance(r.get(source), (int, float))])[0]
            pairs = [(records[(dataset, arm, s)][metric], records[(dataset, "grouped", s)][metric])
                     for s in seeds if (dataset, arm, s) in records and (dataset, "grouped", s) in records]
            row["paired_seed_count"] = len(pairs)
            row["paired_wins_vs_grouped"] = sum(a < b for a, b in pairs) if pairs and arm != "grouped" else None
            row["mae_delta_vs_grouped"] = stats([a - b for a, b in pairs])[0] if arm != "grouped" else None
            row["mae_change_pct_vs_grouped"] = pct(stats([a for a, _ in pairs])[0], stats([b for _, b in pairs])[0]) if pairs and arm != "grouped" else None
            current[arm] = row
        baseline = current.get("grouped", {})
        for arm, row in current.items():
            row["parameter_reduction_pct"] = -pct(row["params"], baseline.get("params")) if row["params"] is not None and baseline.get("params") else None
            row["latency_change_pct"] = pct(row["latency_ms_batch1"], baseline.get("latency_ms_batch1"))
            row["memory_change_pct"] = pct(row["peak_mib_batch64"], baseline.get("peak_mib_batch64"))
            rows.append(row)
        differences = [records[(dataset, "grouped_slim", s)][metric] - records[(dataset, "slim_then_grouped", s)][metric]
                       for s in seeds if (dataset, "grouped_slim", s) in records and (dataset, "slim_then_grouped", s) in records]
        comparisons.append({"dataset": dataset, "paired_seeds": len(differences),
            "A_minus_B_mae_mean": stats(differences)[0], "A_minus_B_mae_sd": stats(differences)[1],
            "A_wins": sum(d < 0 for d in differences) if differences else None,
            "interpretation": "Descriptive paired differences; no statistical significance claim"})
    return rows, comparisons


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(results_dir, out_dir, profile_dir=None, datasets=("PEMS04", "PEMS08"),
                 arms=ARMS, seeds=(0, 1), validation_only=True, overwrite=False, validation_metrics_dir=None):
    names = ("unified_slim_summary.md", "unified_slim_summary.csv", "unified_slim_provenance.json", "unified_slim_missing.txt")
    outputs = [out_dir / name for name in names]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Report already exists in {out_dir}")
    records, missing = read_runs(results_dir, datasets, arms, seeds, validation_only)
    profiles = read_profiles(profile_dir, records)
    validation_metrics = read_validation_metrics(validation_metrics_dir, records) if validation_only else {}
    rows, comparisons = summarize(records, profiles, datasets, arms, seeds, validation_only, validation_metrics)
    lines = ["# Grouped / Slim fusion-order comparison", "",
        f"Mode: {'validation only (best_val_mae)' if validation_only else 'test evaluation'}. "
        f"Completed {len(records)}/{len(datasets)*len(arms)*len(seeds)} runs. All pilot findings are exploratory.", "",
        "MAE change is candidate minus Grouped; negative is lower. Standard deviations are sample SD. "
        "RMSE/MAPE are means of per-run metrics. Missing metrics remain n/a. In validation mode, RMSE/MAPE and horizon errors come only from "
        "separate best-checkpoint validation metrics; these are not test results.", "",
        "| Dataset | Model | Seeds | MAE mean ± SD | RMSE | MAPE | Params | Counted GFLOPs | Latency B=1 ms | Peak B=64 MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['completed_seeds']}/{row['expected_seeds']} | "
            f"{fmt(row['mae_mean'])} ± {fmt(row['mae_sd'])} | {fmt(row['rmse_mean'])} | {fmt(row['mape_mean'])} | "
            f"{row['params'] if row['params'] is not None else 'n/a'} | {fmt(row['counted_gflops'])} | "
            f"{fmt(row['latency_ms_batch1'], 3)} | {fmt(row['peak_mib_batch64'], 1)} |")
    lines += ["", "## Paired and resource comparisons", "",
        "| Dataset | Model | ΔMAE | ΔMAE % | Wins / pairs | Param reduction % | Δlatency % | Δmemory % | MAE 15/30/60m |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        wins = f"{row['paired_wins_vs_grouped']}/{row['paired_seed_count']}" if row['paired_wins_vs_grouped'] is not None else "n/a"
        horizons = "/".join(fmt(row[f'mae_{m}m']) for m in (15, 30, 60))
        lines.append(f"| {row['dataset']} | {row['arm']} | {fmt(row['mae_delta_vs_grouped'])} | "
            f"{fmt(row['mae_change_pct_vs_grouped'], 2)} | {wins} | {fmt(row['parameter_reduction_pct'], 2)} | "
            f"{fmt(row['latency_change_pct'], 2)} | {fmt(row['memory_change_pct'], 2)} | {horizons} |")
    lines += ["", "## A versus B", ""]
    for row in comparisons:
        lines.append(f"- {row['dataset']}: paired A−B MAE {fmt(row['A_minus_B_mae_mean'])} ± "
                     f"{fmt(row['A_minus_B_mae_sd'])}; A wins {row['A_wins'] if row['A_wins'] is not None else 'n/a'}/{row['paired_seeds']}.")
    available = [r["A_minus_B_mae_mean"] for r in comparisons if r["A_minus_B_mae_mean"] is not None]
    if len(datasets) < 2 or len(available) != len(datasets):
        consistency = "Not assessed: paired results are needed from both datasets"
    elif all(value < 0 for value in available):
        consistency = "A has lower available paired mean MAE on both datasets (descriptive)"
    elif all(value > 0 for value in available):
        consistency = "B has lower available paired mean MAE on both datasets (descriptive)"
    else:
        consistency = "No common strict direction across datasets; signs differ or a mean is tied"
    lines += ["", f"Cross-dataset consistency: {consistency}."]
    lines += ["", "## Interpretation and measurement limits", "",
        "A fuses before spatial processing; B reuses the same three Slim blocks across four streams. "
        "A/B have equal parameters, but B executes Slim/FFN 12 times versus A's 3. Prompt position and "
        "nonlinear processing also differ. Results cannot identify a causal effect of fusion position alone.", "",
        "Cost values come only from matching saved-checkpoint profiles, not random weights. The CSV records "
        "profile_seed_count. FLOPs count registered formulas only; latency/memory depend on the device/runtime. "
        "Peak B=64 memory is inference allocation, not training peak. No significant improvement or cross-dataset "
        "consistency is asserted from this descriptive pilot; both datasets must be examined.", ""]
    provenance = {
        "base_commit": BASE_COMMIT, "slim_reference_commit": SLIM_COMMIT,
        "validation_only": validation_only, "seeds": list(seeds), "missing": missing,
        "supplemental_validation_metric_count": len(validation_metrics),
        "code_digests": sorted({r["code_digest"] for r in records.values()}),
        "data_digests": {d: sorted({r['data_digest'] for k, r in records.items() if k[0] == d}) for d in datasets},
        "comparisons": comparisons,
        "cross_dataset_consistency": consistency,
        "profile_environments": [{k: r.get(k) for k in ("dataset", "arm", "seed", "gpu_model", "torch_version", "cuda_version", "precision", "tf32_enabled", "warmup", "repeats", "profile_batch_sizes", "flops_count_method")} for r in profiles.values()],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text("\n".join(lines), encoding="utf-8")
    with outputs[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset", "arm"])
        writer.writeheader()
        writer.writerows(rows)
    outputs[2].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs[3].write_text("".join(name + "\n" for name in missing), encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--validation-metrics-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("PEMS04", "PEMS08"), default=["PEMS04", "PEMS08"])
    parser.add_argument("--arms", nargs="+", choices=OPTIONAL_ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else ([0, 1] if args.validation_only else list(range(5)))
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in range(5) for s in seeds):
        parser.error("Select unique seeds in 0..4")
    for path in build_report(args.results_dir, args.out_dir, args.profile_dir, args.datasets,
                             args.arms, seeds, args.validation_only, args.overwrite, args.validation_metrics_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

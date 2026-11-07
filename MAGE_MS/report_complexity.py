"""Combine saved profile measurements with matched evaluation MAE results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SUMMARY_FIELDS = (
    "dataset",
    "arm",
    "test_mae_mean",
    "test_mae_sd",
    "mae_seed_count",
    "profile_seed_count",
    "params_total",
    "params_trainable",
    "params_non_trainable",
    "graph_params",
    "positive_graph_applications",
    "negative_graph_applications",
    "total_graph_applications",
    "graph_applications_comparability",
    "gflops_counted_per_forward",
    "latency_ms_batch1_median",
    "latency_ms_batch1_p95",
    "latency_ms_batch64_median",
    "latency_ms_batch64_p95",
    "feature_preparation_ms_batch1_median",
    "feature_preparation_ms_batch1_p95",
    "feature_preparation_ms_batch64_median",
    "feature_preparation_ms_batch64_p95",
    "throughput_samples_per_s_batch64",
    "model_memory_mib",
    "inference_peak_allocated_mib_batch1",
    "inference_peak_allocated_mib_batch64",
    "parameter_memory_mib_fp32",
)
PROFILE_MEAN_FIELDS = {
    "gflops_counted_per_forward",
    "latency_ms_batch1_median",
    "latency_ms_batch1_p95",
    "latency_ms_batch64_median",
    "latency_ms_batch64_p95",
    "feature_preparation_ms_batch1_median",
    "feature_preparation_ms_batch1_p95",
    "feature_preparation_ms_batch64_median",
    "feature_preparation_ms_batch64_p95",
    "throughput_samples_per_s_batch64",
    "model_memory_mib",
    "inference_peak_allocated_mib_batch1",
    "inference_peak_allocated_mib_batch64",
    "parameter_memory_mib_fp32",
}


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _profile_rows(profile_dir: Path):
    rows = []
    for path in sorted(profile_dir.glob("profile_*.json")):
        row = _load_json(path)
        row["_profile_file"] = str(path)
        rows.append(row)
    return rows


def _mae_rows(mae_dir: Path):
    rows = []
    for path in sorted(mae_dir.glob("*_s*.json")):
        try:
            row = _load_json(path)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot read MAE result {path}: {error}") from error
        if row.get("completed") and row.get("dataset") and row.get("arm"):
            row["_mae_file"] = str(path)
            rows.append(row)
    return rows


def _mean(values):
    valid = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return statistics.mean(valid) if valid else None


def _stdev(values):
    valid = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return statistics.stdev(valid) if len(valid) > 1 else None


def _aggregate(mae_rows, profile_rows):
    keys = sorted(
        {(row["dataset"], row["arm"]) for row in mae_rows}
        | {(row["dataset"], row["arm"]) for row in profile_rows}
    )
    summaries = []
    for dataset, arm in keys:
        maes = [
            row.get("test_mae")
            for row in mae_rows
            if row["dataset"] == dataset and row["arm"] == arm
        ]
        profiles = [
            row
            for row in profile_rows
            if row.get("dataset") == dataset
            and row.get("arm") == arm
            and row.get("status") == "completed"
        ]
        first = profiles[0] if profiles else {}
        summary = {
            "dataset": dataset,
            "arm": arm,
            "test_mae_mean": _mean(maes),
            "test_mae_sd": _stdev(maes),
            "mae_seed_count": sum(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in maes
            ),
            "profile_seed_count": len(profiles),
        }
        for field in SUMMARY_FIELDS[6:]:
            if field in PROFILE_MEAN_FIELDS:
                values = [
                    profile.get(field)
                    for profile in profiles
                    if isinstance(profile.get(field), (int, float))
                ]
                value = _mean(values) if values else None
            else:
                value = first.get(field)
            summary[field] = value
        summaries.append(summary)
    return summaries


def _per_run(mae_rows, profile_rows):
    profiles = {
        (row.get("dataset"), row.get("arm"), row.get("seed")): row
        for row in profile_rows
    }
    result = []
    for mae in mae_rows:
        profile = profiles.get((mae.get("dataset"), mae.get("arm"), mae.get("seed")), {})
        result.append({
            "dataset": mae.get("dataset"),
            "arm": mae.get("arm"),
            "seed": mae.get("seed"),
            "test_mae": mae.get("test_mae"),
            "profile_status": profile.get("status", "not_profiled"),
            "profile_error": profile.get("error"),
            "params_total": profile.get("params_total"),
            "params_trainable": profile.get("params_trainable"),
            "params_non_trainable": profile.get("params_non_trainable"),
            "graph_params": profile.get("graph_params"),
            "positive_graph_applications": profile.get("positive_graph_applications"),
            "negative_graph_applications": profile.get("negative_graph_applications"),
            "total_graph_applications": profile.get("total_graph_applications"),
            "graph_applications_comparability": profile.get(
                "graph_applications_comparability",
                "not_comparable_to_proposal_factorized_calls"
                if profile.get("total_graph_applications") is None
                else "proposal_factorized_operator_calls",
            ),
            "gflops_counted_per_forward": profile.get("gflops_counted_per_forward"),
            "latency_ms_batch1_median": profile.get("latency_ms_batch1_median"),
            "latency_ms_batch1_p95": profile.get("latency_ms_batch1_p95"),
            "latency_ms_batch64_median": profile.get("latency_ms_batch64_median"),
            "latency_ms_batch64_p95": profile.get("latency_ms_batch64_p95"),
            "feature_preparation_ms_batch1_median": profile.get("feature_preparation_ms_batch1_median"),
            "feature_preparation_ms_batch1_p95": profile.get("feature_preparation_ms_batch1_p95"),
            "feature_preparation_ms_batch64_median": profile.get("feature_preparation_ms_batch64_median"),
            "feature_preparation_ms_batch64_p95": profile.get("feature_preparation_ms_batch64_p95"),
            "throughput_samples_per_s_batch64": profile.get("throughput_samples_per_s_batch64"),
            "model_memory_mib": profile.get("model_memory_mib"),
            "inference_peak_allocated_mib_batch1": profile.get("inference_peak_allocated_mib_batch1"),
            "inference_peak_allocated_mib_batch64": profile.get("inference_peak_allocated_mib_batch64"),
            "parameter_memory_mib_fp32": profile.get("parameter_memory_mib_fp32"),
            "checkpoint_code_digest": profile.get("checkpoint_code_digest"),
            "profile_code_digest": profile.get("profile_code_digest"),
            "checkpoint_code_digest_match": profile.get("checkpoint_code_digest_match"),
        })
    return result


def _comparison_lines(dataset, by_arm):
    comparisons = (
        ("proposed", "shared", "Proposed vs Shared"),
        ("proposed", "reversed", "Proposed vs Reversed"),
        ("proposed", "mage", "Proposed vs MAGE"),
        ("independent", "shared", "Independent vs Shared"),
        ("uniform", "proposed", "Uniform vs Proposed"),
    )
    lines = []
    for left, right, label in comparisons:
        a, b = by_arm.get(left), by_arm.get(right)
        if not a or not b:
            continue
        mae_delta = (
            None
            if a["test_mae_mean"] is None or b["test_mae_mean"] is None
            else a["test_mae_mean"] - b["test_mae_mean"]
        )
        flops_delta = (
            None
            if a["gflops_counted_per_forward"] is None or b["gflops_counted_per_forward"] is None
            else a["gflops_counted_per_forward"] - b["gflops_counted_per_forward"]
        )
        params_delta = (
            None
            if a["params_total"] is None or b["params_total"] is None
            else a["params_total"] - b["params_total"]
        )
        lines.append(
            f"- **{label}:** ΔMAE={_fmt(mae_delta)}, "
            f"Δcounted GFLOPs={_fmt(flops_delta)}, Δparameters={_fmt(params_delta)} "
            f"(first minus second; negative is lower)."
        )
    return lines


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_report(mae_dir: Path, profile_dir: Path, overwrite: bool = False):
    mae_rows = _mae_rows(mae_dir)
    profile_rows = _profile_rows(profile_dir)
    if not profile_rows:
        raise FileNotFoundError(f"No profile_*.json files found in {profile_dir}")
    summaries = _aggregate(mae_rows, profile_rows)
    per_run = _per_run(mae_rows, profile_rows)

    summary_csv = profile_dir / "model_complexity_summary.csv"
    per_run_csv = profile_dir / "model_complexity_per_run.csv"
    markdown = profile_dir / "model_complexity_table.md"
    summary_json = profile_dir / "model_complexity_summary.json"
    outputs = (summary_csv, per_run_csv, markdown, summary_json)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite report outputs: {', '.join(existing)}")

    _write_csv(summary_csv, SUMMARY_FIELDS, summaries)
    _write_csv(per_run_csv, tuple(per_run[0]) if per_run else ("dataset", "arm", "seed"), per_run)
    with markdown.open("w", encoding="utf-8") as handle:
        handle.write("# Model complexity and matched MAE\n\n")
        handle.write(
            "MAE is the mean ± sample standard deviation over completed seed runs. "
            "Complexity fields come from saved-checkpoint profiles; if only seed 0 was profiled, "
            "they represent seed 0. Counted FLOPs include only formulas registered by the stated "
            "PyTorch FlopCounterMode and are not guaranteed to cover every operator. "
            "Latency and GPU memory are hardware/runtime-specific.\n\n"
        )
        handle.write(
            "| Dataset | Model | Test MAE (mean ± SD) | Params (M) | Counted GFLOPs | "
            "Graph apps (+/−) | Latency B=1 (ms) | Throughput B=64 (samples/s) | "
            "Peak GPU memory B=1/64 (MiB) |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in summaries:
            mae = "n/a" if row["test_mae_mean"] is None else (
                f'{row["test_mae_mean"]:.4f} ± '
                f'{_fmt(row["test_mae_sd"], 4) if row["test_mae_sd"] is not None else "n/a"}'
            )
            if row["positive_graph_applications"] is None:
                apps = "not comparable (MAGE path)"
            else:
                apps = f'{row["positive_graph_applications"]}/{row["negative_graph_applications"]}'
            memory = (
                f'{_fmt(row["inference_peak_allocated_mib_batch1"], 1)}/'
                f'{_fmt(row["inference_peak_allocated_mib_batch64"], 1)}'
            )
            handle.write(
                f'| {row["dataset"]} | {row["arm"]} | {mae} | '
                f'{_fmt(None if row["params_total"] is None else row["params_total"] / 1e6, 3)} | '
                f'{_fmt(row["gflops_counted_per_forward"], 4)} | {apps} | '
                f'{_fmt(row["latency_ms_batch1_median"], 3)} | '
                f'{_fmt(row["throughput_samples_per_s_batch64"], 1)} | {memory} |\n'
            )
        handle.write("\n## Paired model comparisons\n\n")
        for dataset in sorted({row["dataset"] for row in summaries}):
            handle.write(f"### {dataset}\n\n")
            by_arm = {row["arm"]: row for row in summaries if row["dataset"] == dataset}
            lines = _comparison_lines(dataset, by_arm)
            handle.write("\n".join(lines) + "\n\n" if lines else "No complete pairs available.\n\n")
        completed_profiles = [row for row in profile_rows if row.get("status") == "completed"]
        environments = [
            {
                "dataset": row.get("dataset"),
                "arm": row.get("arm"),
                "device": row.get("device"),
                "gpu_model": row.get("gpu_model"),
                "torch_version": row.get("torch_version"),
                "cuda_version": row.get("cuda_version"),
                "precision": row.get("precision"),
                "tf32_enabled": row.get("tf32_enabled"),
                "warmup": row.get("warmup"),
                "repeats": row.get("repeats"),
                "flops_count_method": row.get("flops_count_method"),
                "flops_coverage_note": row.get("flops_coverage_note"),
                "checkpoint_code_digest": row.get("checkpoint_code_digest"),
                "profile_code_digest": row.get("profile_code_digest"),
                "checkpoint_code_digest_match": row.get("checkpoint_code_digest_match"),
            }
            for row in completed_profiles
        ]
        handle.write("## Measurement environment\n\n")
        handle.write("Each completed profile records its GPU, PyTorch/CUDA versions, FP32/TF32 setting, batches, warm-up/repeat counts, FLOP method, and checkpoint/profile code-digest match status. See `model_complexity_per_run.csv` for run-level values.\n\n")
    payload = {
        "mae_results_dir": str(mae_dir),
        "profile_dir": str(profile_dir),
        "profile_seed_policy": "Complexity metrics are taken from the available profiled checkpoint(s); they are not averaged across unprofiled MAE seeds.",
        "summaries": summaries,
        "per_run": per_run,
        "profile_environment": environments,
        "profile_statuses": [
            {key: row.get(key) for key in ("dataset", "arm", "seed", "status", "error")}
            for row in profile_rows
        ],
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mae-results-dir", type=Path, default=ROOT / "results/eval_proposal96")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "results/complexity_v1")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing report files")
    args = parser.parse_args()
    outputs = build_report(args.mae_results_dir, args.profile_dir, args.overwrite)
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

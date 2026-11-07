"""Profile saved MAGE proposal checkpoints without training or test targets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

import train_eval
from proposal_model import ARMS

ROOT = Path(__file__).resolve().parent
DEFAULT_ARMS = ("mage", "shared", "proposed", "reversed", "independent", "uniform")
PROFILE_ARMS = DEFAULT_ARMS + ("mage_features",) + tuple(
    arm for arm in ARMS if arm not in DEFAULT_ARMS
)


class IncompatibleCheckpointError(RuntimeError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: Path, value: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing profile: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _checkpoint_args(checkpoint: dict, dataset: str, arm: str, data_dir: Path, device: torch.device):
    if not isinstance(checkpoint.get("args"), dict):
        raise IncompatibleCheckpointError("checkpoint is missing its saved args dictionary")
    values = dict(checkpoint["args"])
    if values.get("dataset") != dataset or values.get("arm") != arm:
        raise IncompatibleCheckpointError(
            f"checkpoint identity is {values.get('dataset')}/{values.get('arm')}, "
            f"not {dataset}/{arm}"
        )
    if values.get("seq_len") != 96 or values.get("horizon") != 12:
        raise IncompatibleCheckpointError("checkpoint does not use the 96-to-12 protocol")
    values.update(
        dataset=dataset,
        arm=arm,
        data_dir=str(data_dir),
        device=str(device),
    )
    return SimpleNamespace(**values)


def _load_profile_inputs(args, dataset: str, arm: str, data_dir: Path, batch_size: int):
    load_args = SimpleNamespace(**vars(args))
    load_args.device = "cpu"
    if arm == "mage_features":
        load_args.arm = "mage"
    data, _, _, _, indices = train_eval.legacy._load_data(load_args, torch.device("cpu"))
    origins = indices["val"][:batch_size]
    if len(origins) != batch_size:
        raise ValueError(f"{dataset} validation split has fewer than {batch_size} origins")
    offsets = torch.arange(-(args.seq_len - 1), 1, device=origins.device)
    x = data[origins[:, None] + offsets]
    if arm == "mage_features":
        x = train_eval.make_window_features(x)
    return data, indices, x


def _applications(arm: str, model) -> tuple[int | None, int | None, int | None, dict | None]:
    if not hasattr(model, "positive_propagation_count"):
        return None, None, None, None
    positive = int(model.positive_propagation_count)
    negative = int(model.negative_propagation_count)
    by_scale = {}
    by_scale.update(model.applications_by_scale)
    return positive, negative, positive + negative, by_scale


def _model_memory_bytes(model: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in model.parameters()) + sum(
        t.numel() * t.element_size() for t in model.buffers()
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latencies(model, x, bias, warmup: int, repeats: int, device: torch.device):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, bias=bias)
        _sync(device)
        measurements = []
        if device.type == "cuda":
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x, bias=bias)
                end.record()
                end.synchronize()
                measurements.append(float(start.elapsed_time(end)))
        else:
            for _ in range(repeats):
                start = time.perf_counter()
                model(x, bias=bias)
                measurements.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(statistics.median(measurements)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "throughput_samples_per_s": float(x.shape[0] / (statistics.median(measurements) / 1000)),
    }


def _feature_latencies(x, warmup: int, repeats: int, device: torch.device):
    measurements = []
    with torch.inference_mode():
        for _ in range(warmup):
            train_eval.make_window_features(x)
        _sync(device)
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                train_eval.make_window_features(x)
                end.record()
                end.synchronize()
                measurements.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                train_eval.make_window_features(x)
                measurements.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(statistics.median(measurements)),
        "p95_ms": float(np.percentile(measurements, 95)),
    }


def _inference_memory(model, x, bias, device: torch.device):
    if device.type != "cuda":
        return {
            "input_memory_mib": None,
            "inference_peak_allocated_mib": None,
            "inference_additional_peak_mib": None,
        }
    _sync(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(x, bias=bias)
        if isinstance(output, tuple):
            output = output[0]
    _sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    result = {
        "input_memory_mib": float(x.numel() * x.element_size() / 2**20),
        "inference_peak_allocated_mib": float(peak / 2**20),
        "inference_additional_peak_mib": float(max(0, peak - baseline) / 2**20),
    }
    del output
    return result


def _count_flops(model, x, bias, device: torch.device):
    method = (
        "torch.utils.flop_counter.FlopCounterMode under torch.no_grad "
        "(inference_mode disables dispatch counting in the installed PyTorch)"
    )
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as error:
        return None, method, f"FlopCounterMode unavailable: {error}"
    try:
        model.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                output = model(x, bias=bias)
                if isinstance(output, tuple):
                    output = output[0]
            _sync(device)
        value = int(counter.get_total_flops())
        if value <= 0:
            raise RuntimeError("counter returned no counted operations")
        return value, method, None
    except Exception as error:
        return None, method, f"{type(error).__name__}: {error}"


def _validate_output(model, x, bias, expected_nodes: int, horizon: int) -> None:
    model.eval()
    with torch.inference_mode():
        output = model(x, bias=bias)
    if isinstance(output, tuple):
        output = output[0]
    expected = (x.shape[0], horizon, expected_nodes, 1)
    if tuple(output.shape) != expected:
        raise ValueError(f"model output shape {tuple(output.shape)} != {expected}")
    if not torch.isfinite(output).all():
        raise ValueError("model produced non-finite output values")


def profile_one(
    dataset: str,
    arm: str,
    seed: int,
    checkpoint_path: Path,
    data_dir: Path,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    current_data_digest = train_eval.data_digest(dataset, str(data_dir))
    if checkpoint.get("data_digest") != current_data_digest:
        raise IncompatibleCheckpointError(
            f"checkpoint data digest {checkpoint.get('data_digest')} does not match "
            f"prepared data {current_data_digest}"
        )
    args = _checkpoint_args(checkpoint, dataset, arm, data_dir, device)
    data, indices, raw_inputs = _load_profile_inputs(
        args, dataset, arm, data_dir, max(batch_sizes)
    )
    input_dim = 8 if arm == "mage_features" else int(data.shape[-1])
    model, graph_params, _, _ = train_eval.legacy._build_model(
        arm, args, int(data.shape[1]), input_dim
    )
    try:
        model.load_state_dict(checkpoint["state"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise IncompatibleCheckpointError(f"strict state restore failed: {error}") from error
    model = model.to(device).eval()
    if "bias" not in checkpoint:
        raise IncompatibleCheckpointError("checkpoint is missing the trained routing bias")
    bias = checkpoint["bias"].to(device)

    val_count = len(indices["val"])
    if val_count < max(batch_sizes):
        raise ValueError(f"{dataset} validation split has fewer than {max(batch_sizes)} origins")
    feature_timings = {}
    def prepare_input(batch_size: int):
        raw_x = raw_inputs[:batch_size].to(device)
        if arm == "mage_features":
            feature_timings[str(batch_size)] = _feature_latencies(
                raw_x, warmup, repeats, device
            )
            return train_eval.make_window_features(raw_x)
        return raw_x

    count_x = prepare_input(1)
    _validate_output(model, count_x, bias, int(data.shape[1]), args.horizon)
    count_batch_size = count_x.shape[0]
    params_total = sum(parameter.numel() for parameter in model.parameters())
    params_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    params_non_trainable = params_total - params_trainable
    positive, negative, total, by_scale = _applications(arm, model)

    flops, flop_method, flop_error = _count_flops(model, count_x, bias, device)
    del count_x
    row = {
        "status": "completed",
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_code_digest": checkpoint.get("code_digest"),
        "checkpoint_signature": checkpoint.get("signature"),
        "data_digest": checkpoint.get("data_digest"),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "params_non_trainable": int(params_non_trainable),
        "graph_params": None if graph_params is None else int(graph_params),
        "parameter_memory_mib_fp32": float(params_total * 4 / 2**20),
        "model_memory_mib": float(_model_memory_bytes(model) / 2**20),
        "positive_graph_applications": positive,
        "negative_graph_applications": negative,
        "total_graph_applications": total,
        "graph_applications_comparability": (
            "not_comparable_to_proposal_factorized_calls"
            if total is None
            else "proposal_factorized_operator_calls"
        ),
        "applications_by_scale": by_scale,
        "flops_counted_per_forward": flops,
        "gflops_counted_per_forward": None if flops is None else float(flops / 1e9),
        "flops_counted_per_sample": flops,
        "flops_count_method": flop_method,
        "flops_coverage_note": (
            "PyTorch FlopCounterMode counts only registered operator formulas; "
            "unregistered operators are not included. It counts the complete executed forward path."
        ),
        "flops_error": flop_error,
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "FP32",
        "tf32_enabled": False,
        "warmup": warmup,
        "repeats": repeats,
        "validation_inputs_only": True,
        "timings": {},
        "feature_preparation_timings": feature_timings,
        "memory_by_batch": {},
    }
    if flops is not None:
        row["flops_counted_per_sample"] = flops / count_batch_size
    for batch_size in batch_sizes:
        x = prepare_input(batch_size)
        _validate_output(model, x, bias, int(data.shape[1]), args.horizon)
        key = str(batch_size)
        row["timings"][key] = _latencies(model, x, bias, warmup, repeats, device)
        row["memory_by_batch"][key] = _inference_memory(model, x, bias, device)
        del x
    for batch_size, key in ((1, "batch1"), (64, "batch64")):
        if str(batch_size) in row["timings"]:
            timing = row["timings"][str(batch_size)]
            row[f"latency_ms_{key}_median"] = timing["median_ms"]
            row[f"latency_ms_{key}_p95"] = timing["p95_ms"]
            if batch_size == 64:
                row["throughput_samples_per_s_batch64"] = timing["throughput_samples_per_s"]
        if str(batch_size) in row["memory_by_batch"]:
            memory = row["memory_by_batch"][str(batch_size)]
            row[f"input_memory_mib_{key}"] = memory["input_memory_mib"]
            row[f"inference_peak_allocated_mib_{key}"] = memory["inference_peak_allocated_mib"]
            row[f"inference_additional_peak_mib_{key}"] = memory["inference_additional_peak_mib"]
        if str(batch_size) in row["feature_preparation_timings"]:
            feature_timing = row["feature_preparation_timings"][str(batch_size)]
            row[f"feature_preparation_ms_{key}_median"] = feature_timing["median_ms"]
            row[f"feature_preparation_ms_{key}_p95"] = feature_timing["p95_ms"]
    row["inference_peak_allocated_mib"] = row["memory_by_batch"].get("1", {}).get(
        "inference_peak_allocated_mib"
    )
    return row


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("PEMS04", "PEMS08"), default=["PEMS04", "PEMS08"])
    parser.add_argument("--arms", nargs="+", choices=PROFILE_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/eval_proposal96")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_proposal96")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/complexity_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Replace profile JSON files already present in --out-dir")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if any(size < 1 for size in args.batch_sizes) or args.warmup < 0 or args.repeats < 1:
        parser.error("batch sizes and repeats must be positive; warmup cannot be negative")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA requested but unavailable: {device}")
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results_dir = _resolve(args.results_dir)
    data_dir = _resolve(args.data_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for arm in args.arms:
            for seed in args.seeds:
                name = f"profile_{dataset}_{arm}_s{seed}.json"
                output_path = out_dir / name
                try:
                    row = profile_one(
                        dataset,
                        arm,
                        seed,
                        results_dir / f"{dataset}_{arm}_s{seed}.pt",
                        data_dir,
                        device,
                        args.batch_sizes,
                        args.warmup,
                        args.repeats,
                    )
                except FileNotFoundError as error:
                    row = {"status": "missing", "dataset": dataset, "arm": arm, "seed": seed, "error": str(error)}
                except IncompatibleCheckpointError as error:
                    row = {"status": "incompatible", "dataset": dataset, "arm": arm, "seed": seed, "error": str(error)}
                except Exception as error:
                    row = {
                        "status": "error",
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                row.update({
                    "profile_device": str(device),
                    "profile_torch_version": torch.__version__,
                    "profile_cuda_version": torch.version.cuda,
                    "profile_warmup": args.warmup,
                    "profile_repeats": args.repeats,
                    "profile_batch_sizes": args.batch_sizes,
                    "profile_code_digest": train_eval.code_digest(),
                })
                if row.get("status") == "completed":
                    row["checkpoint_code_digest_match"] = (
                        row.get("checkpoint_code_digest") == row["profile_code_digest"]
                    )
                try:
                    _write_json(output_path, row, args.overwrite)
                except FileExistsError as error:
                    print(f"EXISTS {error}; pass --overwrite only if replacement is intended", flush=True)
                    continue
                print(f"{row['status'].upper()} {dataset}/{arm}/s{seed}: {output_path}", flush=True)


if __name__ == "__main__":
    main()

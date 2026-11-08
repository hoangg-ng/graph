"""Evaluate saved best checkpoints on validation only; never train or test.

Supplemental metrics are written separately from the original training results.
"""
import argparse
import json
import math
from pathlib import Path

import torch

import profile_models
import train_eval
from run_unified_slim_eval import OPTIONAL_ARMS


def evaluate_one(result_path, data_dir, device):
    result = json.loads(result_path.read_text())
    if (result.get("completed") is not True or result.get("protocol") != "96to12_v1_validation_only"
            or result.get("config", {}).get("validation_only") is not True):
        raise ValueError(f"Expected completed validation-only run: {result_path}")
    dataset, arm, seed = (result[k] for k in ("dataset", "arm", "seed"))
    if (result_path.name != f"{dataset}_{arm}_s{seed}.json" or type(seed) is not int
            or seed not in range(5) or arm not in OPTIONAL_ARMS):
        raise ValueError("Filename/result identity is not a supported unified run")
    checkpoint = torch.load(result_path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    if checkpoint.get("signature") != result.get("signature"):
        raise ValueError("Checkpoint/result signature mismatch")
    code, digest = train_eval.code_digest(), train_eval.data_digest(dataset, str(data_dir))
    if result["code_digest"] != code or checkpoint.get("code_digest") != code:
        raise ValueError("Checkpoint/result/current source mismatch")
    if result["data_digest"] != digest or checkpoint.get("data_digest") != digest:
        raise ValueError("Checkpoint/result/current data mismatch")
    args = profile_models._checkpoint_args(checkpoint, dataset, arm, data_dir, device)
    if args.tf32 != 0:
        raise ValueError("This comparison uses FP32 with TF32 disabled")
    data, flow, mean, std, indices = train_eval.legacy._load_data(args, device)
    args.flow_mean, args.flow_std = mean, std
    model, *_ = train_eval.legacy._build_model(arm, args, data.shape[1], data.shape[2])
    model.load_state_dict(checkpoint["state"], strict=True)
    model.to(device).eval()
    # This is the only evaluation split used by this entry point.
    prediction, target = train_eval._evaluate(model, checkpoint["bias"].to(device),
                                               data, flow, indices, args, "val")
    metrics = train_eval._metrics(prediction, target)
    horizons = [train_eval._metrics(prediction[:, h], target[:, h]) for h in range(12)]
    if any(v is None or not math.isfinite(v) for row in [metrics, *horizons] for v in row):
        raise ValueError("Non-finite validation metrics")
    if not math.isclose(metrics[0], result["best_val_mae"], rel_tol=1e-5, abs_tol=1e-4):
        raise ValueError("Restored validation MAE disagrees with selected best checkpoint")
    return {
        "completed": True, "protocol": "96to12_v1_validation_best_checkpoint_metrics",
        "dataset": dataset, "arm": arm, "seed": seed, "signature": result["signature"],
        "code_digest": code, "data_digest": digest, "selection_metric": "best_val_mae",
        "val_mae": metrics[0], "val_rmse": metrics[1], "val_mape": metrics[2],
        "per_horizon": horizons, "device": str(device), "tf32": False,
        "description": "Validation split only; no optimization and no test evaluation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("PEMS04", "PEMS08"), default=["PEMS04", "PEMS08"])
    parser.add_argument("--arms", nargs="+", choices=OPTIONAL_ARMS, default=list(OPTIONAL_ARMS[:4]))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or any(s not in range(5) for s in args.seeds)
            or len(set(args.datasets)) != len(args.datasets) or len(set(args.arms)) != len(args.arms)):
        parser.error("Use unique datasets, arms, and seeds 0..4")
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    planned = [(d, a, s) for d in args.datasets for a in args.arms for s in args.seeds]
    outputs = [args.out_dir / f"validation_{d}_{a}_s{s}.json" for d, a, s in planned]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Supplemental outputs exist; use another directory or explicit --overwrite")
    for (dataset, arm, seed), output in zip(planned, outputs):
        result = evaluate_one(args.results_dir / f"{dataset}_{arm}_s{seed}.json", args.data_dir, torch.device(args.device))
        profile_models._write_json(output, result, overwrite=args.overwrite)
        print(f"VALIDATION ONLY {dataset}/{arm}/s{seed}: {result['val_mae']:.4f}, "
              f"RMSE={result['val_rmse']:.4f}, MAPE={result['val_mape']:.4f}", flush=True)


if __name__ == "__main__":
    main()

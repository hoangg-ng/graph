"""Reproducible, resumable evaluation worker for the MAGE + TimeMixer proposal.

Install next to proposal_model.py, train_proposal.py, prep_proposal.py, models.py.
This is additive: existing research code is not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

import train_proposal as legacy
from proposal_model import ARMS

ROOT = Path(__file__).resolve().parent
BASELINES = ("mage", "mage_features")
CODE_FILES = ("train_eval.py", "train_proposal.py", "proposal_model.py", "models.py", "prep_proposal.py")
CONFIG_KEYS = (
    "dataset", "arm", "seed", "seq_len", "horizon", "epochs", "patience",
    "lr", "wd", "bs", "lb", "lb_lr", "dropout", "model_dim", "experts",
    "topk", "graph_dim", "pdm_width", "pdm_depth", "moving_average", "tf32", "device", "validation_only",
)


def file_hash(path: Path, hash_object=None) -> str:
    h = hash_object if hash_object is not None else hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def code_digest() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source: {path}")
        h.update(name.encode())
        file_hash(path, h)
    return h.hexdigest()


def data_dir_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def data_digest(dataset: str, data_dir: str) -> str:
    directory = data_dir_path(data_dir) / dataset
    h = hashlib.sha256()
    for name in ("his.npz", "idx_train.npy", "idx_val.npy", "idx_test.npy", "metadata.json"):
        p = directory / name
        if not p.is_file():
            if name == "metadata.json":
                continue
            raise FileNotFoundError(f"Prepared data missing: {p}; run prep_proposal.py first")
        h.update(name.encode())
        file_hash(p, h)
    return h.hexdigest()


def run_identity(args, source_hash=None, dataset_hash=None) -> dict:
    source_hash = source_hash or code_digest()
    dataset_hash = dataset_hash or data_digest(args.dataset, args.data_dir)
    config = {name: getattr(args, name) for name in CONFIG_KEYS}
    payload = {"config": config, "code_digest": source_hash, "data_digest": dataset_hash}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "signature": identity}


def make_window_features(x: torch.Tensor) -> torch.Tensor:
    """Make 8 channels from exactly the supplied 96-step history, with no earlier observations.

    Output order: normalized flow, trailing 3/12 means, first difference,
    sin/cos time of day, time of day, day of week. Calendar stays in last 2 positions
    as required by MAGE's Prompt_Pool(feature_dim=6).
    """
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected (B,T,N,3) flow+TOD+DOW; got {tuple(x.shape)}")
    flow = x[..., 0]
    b, t, n = flow.shape
    cumulative = torch.cat((torch.zeros_like(flow[:, :1]), torch.cumsum(flow, dim=1)), dim=1)
    positions = torch.arange(1, t + 1, device=x.device)
    averages = []
    for width in (3, 12):
        starts = (positions - width).clamp_min(0)
        denominators = (positions - starts).view(1, t, 1)
        averages.append((cumulative[:, positions] - cumulative[:, starts]) / denominators)
    delta = torch.cat((torch.zeros_like(flow[:, :1]), flow[:, 1:] - flow[:, :-1]), dim=1)
    angle = 2 * torch.pi * x[..., 1]
    return torch.stack(
        (flow, averages[0], averages[1], delta, angle.sin(), angle.cos(), x[..., 1], x[..., 2]),
        dim=-1,
    )


def make_batch(data, flow, origins, args):
    x, y = legacy._batch(data, flow, origins, args.seq_len, args.horizon)
    if args.arm == "mage_features":
        x = make_window_features(x)
    return x, y


def _atomic_save(obj, location: Path):
    location.parent.mkdir(parents=True, exist_ok=True)
    temp = location.with_name(f"{location.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, temp)
        os.replace(temp, location)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(obj, location: Path):
    location.parent.mkdir(parents=True, exist_ok=True)
    temp = location.with_name(f"{location.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, location)
    finally:
        temp.unlink(missing_ok=True)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in s["cuda"]])


def _metrics(pred, target):
    try:
        return [float(v.item()) for v in legacy.masked_metrics(pred, target)]
    except ValueError:
        return [None, None, None]


def _evaluate(model, bias, data, flow, indices, args, split):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for origins in indices[split].split(args.bs):
            x, y = make_batch(data, flow, origins, args)
            output = model(x, bias=bias)
            if isinstance(output, tuple):
                output = output[0]
            predictions.append((output[..., 0] * args.flow_std + args.flow_mean).cpu())
            targets.append(y.cpu())
    return torch.cat(predictions), torch.cat(targets)


def _cpu_or_gpu_context(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested GPU is not available: {device}")
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = False
    return device


def run(args):
    if args.seq_len != 96 or args.horizon != 12:
        raise ValueError("This study uses only the shared 96 -> 12 protocol")
    if args.arm not in ARMS + BASELINES:
        raise ValueError(f"Unknown arm: {args.arm}")
    device = _cpu_or_gpu_context(args)
    identity = run_identity(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    resume_path = Path(str(out) + ".resume.pt")
    best_path = out.with_suffix(".pt")

    if out.is_file():
        with out.open(encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("completed") and existing.get("signature") == identity["signature"]:
            print(f"Already completed: {out}", flush=True)
            return existing
        if not args.overwrite:
            raise RuntimeError(
                f"Result already exists with a different/older protocol: {out}. "
                "Choose another --out or explicitly pass --overwrite."
            )
    if resume_path.exists() and args.reset_resume:
        resume_path.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # legacy._load_data would create engineered features on the entire time series.
    # Load only the 3 raw prepared channels for ALL arms; create engineered features
    # inside each individual historical window in make_batch instead.
    load_args = argparse.Namespace(**vars(args))
    if args.arm == "mage_features":
        load_args.arm = "mage"
    data, flow, mean, std, indices = legacy._load_data(load_args, device)
    args.flow_mean, args.flow_std = mean, std
    if data.shape[-1] != 3:
        raise RuntimeError("Prepared data must contain 3 channels before feature engineering")

    model, graph_params, applications, positive_apps = legacy._build_model(
        args.arm, args, data.shape[1], 8 if args.arm == "mage_features" else 3
    )
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1, 30, 50, 70, 90, 130, 170, 210, 250, 290, 300], gamma=0.5
    )
    bias = torch.zeros(3, args.experts, device=device)
    start_epoch, best_val, best_state, best_bias, wait = 0, float("inf"), None, None, 0
    log, epoch_seconds = [], []

    if resume_path.exists():
        snapshot = torch.load(resume_path, map_location="cpu", weights_only=False)
        if snapshot.get("signature") != identity["signature"]:
            raise RuntimeError(
                f"Incompatible resume snapshot: {resume_path}. "
                "Use a separate output directory or --reset-resume."
            )
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        bias = snapshot["bias"].to(device)
        start_epoch = snapshot["epoch"]
        best_val, best_state, best_bias, wait = (
            snapshot["best_val"], snapshot["best_state"], snapshot["best_bias"], snapshot["wait"]
        )
        log, epoch_seconds = snapshot["log"], snapshot["epoch_seconds"]
        _restore_rng(snapshot["rng"])
        print(f"Resuming {args.dataset}/{args.arm}/s{args.seed} at epoch {start_epoch + 1}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"START dataset={args.dataset} arm={args.arm} seed={args.seed} "
        f"params={params} graph_params={graph_params} device={device} "
        f"graph_applications={applications} signature={identity['signature'][:12]}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.monotonic()
        losses = []
        permutation = indices["train"][torch.randperm(len(indices["train"]), device=device)]
        for origins in permutation.split(args.bs):
            x, y = make_batch(data, flow, origins, args)
            if not (torch.isfinite(y) & (y > 0)).any():
                continue
            pred, counts = model(x, bias=bias)
            loss = legacy.masked_metrics(pred[..., 0] * std + mean, y)[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            legacy._bias_counts(bias, counts, args)
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError("Training split has no valid positive targets")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - t0
        epoch_seconds.append(elapsed)
        scheduler.step()
        val_prediction, val_target = _evaluate(model, bias, data, flow, indices, args, "val")
        val_mae = _metrics(val_prediction, val_target)[0]
        if val_mae is None or not np.isfinite(val_mae):
            raise ValueError("Validation MAE not finite")
        log.append({"epoch": epoch + 1, "train_mae": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val, wait = val_mae, 0
            best_state, best_bias = _cpu_state(model), bias.detach().cpu().clone()
        else:
            wait += 1
        if epoch == start_epoch or epoch % 10 == 0 or wait == 0:
            print(
                f"epoch={epoch+1}/{args.epochs} train_mae={np.mean(losses):.3f} "
                f"val_mae={val_mae:.3f} best={best_val:.3f} "
                f"sec={elapsed:.1f} wait={wait}", flush=True,
            )
        if (epoch + 1) % args.save_every == 0 or wait >= args.patience or epoch + 1 == args.epochs:
            _atomic_save({
                "signature": identity["signature"], "epoch": epoch + 1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "bias": bias.detach().clone(),
                "best_val": best_val, "best_state": best_state, "best_bias": best_bias,
                "wait": wait, "log": log, "epoch_seconds": epoch_seconds, "rng": _rng_state(),
            }, resume_path)
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch+1}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint from completed epochs")
    model.load_state_dict(best_state)
    bias = best_bias.to(device)
    _atomic_save({
        "state": best_state, "bias": best_bias, "args": vars(args),
        **identity, "best_val_mae": best_val,
    }, best_path)
    if args.validation_only:
        # Hyperparameter selection may use validation data, never test labels.
        # Do not call _evaluate(..., "test") from this path.
        result = {
            "completed": True, **identity, "dataset": args.dataset,
            "arm": args.arm, "seed": args.seed,
            "recipe_source": args.recipe_source,
            "recipe_digest": args.recipe_digest,
            "recipe_source_code_digest": args.recipe_source_code_digest,
            "protocol": "96to12_v1_validation_only",
            "best_val_mae": best_val, "epochs_run": len(log),
            "params": params, "graph_params": graph_params,
            "train_s_per_epoch": float(np.mean(epoch_seconds)),
            "peak_mem_mb": torch.cuda.max_memory_allocated(device) / (2**20)
            if device.type == "cuda" else None,
            "args": vars(args), "log": log,
        }
        _atomic_json(result, out)
        resume_path.unlink(missing_ok=True)
        print(f"VALIDATION_ONLY {args.dataset} {args.arm} seed={args.seed} "
              f"best_val_mae={best_val:.4f}", flush=True)
        return result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.monotonic()
    pred, target = _evaluate(model, bias, data, flow, indices, args, "test")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_s = time.monotonic() - t0
    masked = _metrics(pred, target)
    per_horizon = [_metrics(pred[:, h], target[:, h]) for h in range(args.horizon)]
    good = torch.isfinite(target)
    unmasked_err = (pred[good] - target[good]) if good.any() else None
    unmasked_mae = unmasked_err.abs().mean().item() if unmasked_err is not None else None
    unmasked_rmse = unmasked_err.square().mean().sqrt().item() if unmasked_err is not None else None
    masked_fraction = 1.0 - float((good & (target > 0)).float().mean().item())
    result = {
        "completed": True, **identity, "dataset": args.dataset,
        "arm": args.arm, "seed": args.seed, "protocol": "96to12_v1_window_local_features",
        "recipe_source": args.recipe_source,
        "recipe_digest": args.recipe_digest,
        "recipe_source_code_digest": args.recipe_source_code_digest,
        "test_mae": masked[0], "test_rmse": masked[1], "test_mape": masked[2],
        "per_horizon": per_horizon,
        "horizon_15m": per_horizon[2], "horizon_30m": per_horizon[5],
        "horizon_60m": per_horizon[11],
        "test_unmasked_mae": unmasked_mae, "test_unmasked_rmse": unmasked_rmse,
        "test_excluded_fraction": masked_fraction,
        "best_val_mae": best_val, "epochs_run": len(log),
        "params": params, "graph_params": graph_params,
        "graph_applications_per_sample": applications,
        "positive_powers_per_sample": positive_apps,
        "train_s_per_epoch": float(np.mean(epoch_seconds)),
        "test_inference_s": inference_s,
        "test_inference_ms_per_sample": 1000 * inference_s / len(indices["test"]),
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / (2**20)
        if device.type == "cuda" else None,
        "device": args.device, "threads": args.threads,
        "args": vars(args), "log": log,
    }
    _atomic_json(result, out)
    resume_path.unlink(missing_ok=True)
    print(
        f"COMPLETE {args.dataset} {args.arm} seed={args.seed} "
        f"MAE={masked[0]:.4f} RMSE={masked[1]:.4f} MAPE={masked[2]:.2f} "
        f"peak={result['peak_mem_mb']}", flush=True,
    )
    return result


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("PEMS04", "PEMS08"), default="PEMS04")
    p.add_argument("--arm", choices=ARMS + BASELINES, default="proposed")
    p.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lb", type=int, choices=(0, 1), default=1)
    p.add_argument("--lb-lr", "--lb_lr", dest="lb_lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--model-dim", dest="model_dim", type=int, default=128)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--graph-dim", dest="graph_dim", type=int, default=32)
    p.add_argument("--pdm-width", dest="pdm_width", type=int, default=16)
    p.add_argument("--pdm-depth", dest="pdm_depth", type=int, default=2)
    p.add_argument("--moving-average", dest="moving_average", type=int, default=5)
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", default="data_proposal96")
    p.add_argument("--out", required=True)
    p.add_argument("--recipe-source", default=None)
    p.add_argument("--recipe-digest", default=None)
    p.add_argument("--recipe-source-code-digest", default=None)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--tf32", type=int, choices=(0, 1), default=0)
    p.add_argument("--save-every", dest="save_every", type=int, default=10)
    p.add_argument("--reset-resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validation-only", action="store_true",
                   help="Tune on validation only; never access test targets")
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    run(args)


if __name__ == "__main__":
    main()

"""Train one proposal arm or MAGE reference run with the shared evaluation protocol."""

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from proposal_model import ARMS, ProposalMAGE, parameter_count
from prep_proposal import build_mage_feature_data

import models


ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINES = ("mage", "mage_features")


def masked_metrics(prediction, target):
    valid = torch.isfinite(target) & (target > 0)
    if not valid.any():
        raise ValueError("No valid targets in batch")
    error = prediction[valid] - target[valid]
    mae = error.abs().mean()
    rmse = error.square().mean().sqrt()
    mape = (error.abs() / target[valid].abs().clamp_min(1e-6)).mean() * 100
    return mae, rmse, mape


def _metric_values(prediction, target):
    try:
        return [float(value.item()) for value in masked_metrics(prediction, target)]
    except ValueError:
        return [None, None, None]


def _build_model(arm, args, nodes, input_dim):
    model_args = SimpleNamespace(
        model_dim=args.model_dim,
        recur_num=args.experts,
        topk=args.topk,
        node_num=nodes,
        feature_dim=6 if arm == "mage_features" else 1,
        graph_dim=args.graph_dim,
        input_dim=input_dim,
        output_dim=1,
        seq_len=args.seq_len,
        horizon=args.horizon,
        second=0,
        minute=288,
        hour=0,
        day=7,
        week=0,
        weekday=0,
        month=0,
        quarter=0,
        year=0,
        bs=args.bs,
    )
    if arm in ARMS:
        model = ProposalMAGE(
            model_args,
            arm=arm,
            pdm_width=args.pdm_width,
            pdm_depth=args.pdm_depth,
            moving_average=args.moving_average,
            dropout=args.dropout,
        )
        graph_params = model.graph_parameter_count
        propagation_count = model.propagation_count
        positive_propagation_count = model.positive_propagation_count
    else:
        models.use_linear_contraction(True)
        model = models.MAGEX(
            model_args,
            encoder="linear",
            node_num=nodes,
            input_dim=input_dim,
            output_dim=1,
            seq_len=args.seq_len,
            horizon=args.horizon,
        )
        graph_params = None
        propagation_count = None
        positive_propagation_count = None
    return model, graph_params, propagation_count, positive_propagation_count


def _load_data(args, device):
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)
    dataset_dir = os.path.join(data_dir, args.dataset)
    with np.load(os.path.join(dataset_dir, "his.npz")) as archive:
        data_array = np.asarray(archive["data"], dtype=np.float32)
        mean, std = float(archive["mean"]), float(archive["std"])
    if data_array.ndim != 3 or data_array.shape[-1] != 3:
        raise ValueError("proposal data must contain normalized flow, time-of-day, and day-of-week")
    normalized_data = build_mage_feature_data(data_array) if args.arm == "mage_features" else data_array
    data = torch.from_numpy(normalized_data).to(device)
    flow = data[..., 0] * std + mean
    flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
    indices = {
        name: torch.from_numpy(
            np.load(os.path.join(dataset_dir, f"idx_{name}.npy"))
        ).to(device)
        for name in ("train", "val", "test")
    }
    return data, flow, mean, std, indices


def _batch(data, flow, origins, seq_len, horizon):
    offsets_x = torch.arange(-(seq_len - 1), 1, device=data.device)
    offsets_y = torch.arange(1, horizon + 1, device=data.device)
    return data[origins[:, None] + offsets_x], flow[origins[:, None] + offsets_y]


def _bias_counts(bias, counts, args):
    if args.lb and args.arm != "uniform":
        bias -= args.lb_lr * torch.sign(
            counts.float() - counts.float().mean(dim=-1, keepdim=True)
        )


def run(args):
    if args.seq_len != 96 or args.horizon != 12:
        raise ValueError("proposal runs use the common 96 -> 12 protocol")
    if args.arm not in ARMS + BASELINES:
        raise ValueError(f"unknown arm {args.arm!r}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, flow, mean, std, indices = _load_data(args, device)
    nodes, input_dim = data.shape[1], data.shape[2]
    model, graph_params, propagation_count, positive_propagation_count = _build_model(
        args.arm, args, nodes, input_dim
    )
    model = model.to(device)
    total_params = parameter_count(model)
    print(
        f"{args.dataset} {args.arm} seed={args.seed} params={total_params} "
        f"graph_params={graph_params} graph_applications_per_sample={propagation_count} "
        f"positive_powers_per_sample={positive_propagation_count}",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1, 30, 50, 70, 90, 130, 170, 210, 250, 290, 300],
        gamma=0.5,
    )
    bias = torch.zeros(3, args.experts, device=device)

    def predict(split):
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for origins in indices[split].split(args.bs):
                x, y = _batch(data, flow, origins, args.seq_len, args.horizon)
                prediction = model(x, bias=bias)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                predictions.append(prediction[..., 0] * std + mean)
                targets.append(y)
        return torch.cat(predictions), torch.cat(targets)

    best_val, best_state, best_bias = float("inf"), None, None
    wait, epoch_times, log = 0, [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        losses = []
        permutation = indices["train"][
            torch.randperm(len(indices["train"]), device=device)
        ]
        for origins in permutation.split(args.bs):
            x, y = _batch(data, flow, origins, args.seq_len, args.horizon)
            if not (torch.isfinite(y) & (y > 0)).any():
                continue
            prediction, counts = model(x, bias=bias)
            loss = masked_metrics(prediction[..., 0] * std + mean, y)[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            _bias_counts(bias, counts, args)
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError("training split contains no batches with valid positive targets")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_times.append(time.time() - start)
        scheduler.step()
        val_prediction, val_target = predict("val")
        val_mae = float(masked_metrics(val_prediction, val_target)[0].item())
        log.append(
            {"epoch": epoch + 1, "train_mae": float(np.mean(losses)), "val_mae": val_mae}
        )
        if val_mae < best_val:
            best_val, wait = val_mae, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_bias = bias.detach().clone()
        else:
            wait += 1
        if epoch % 10 == 0 or wait == 0:
            print(
                f"epoch={epoch + 1} train={np.mean(losses):.3f} "
                f"val={val_mae:.3f} best={best_val:.3f} "
                f"time={epoch_times[-1]:.1f}s",
                flush=True,
            )
        if wait >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    bias = best_bias
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_start = time.time()
    prediction, target = predict("test")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_inference_s = time.time() - inference_start
    masked = _metric_values(prediction, target)
    per_horizon = [_metric_values(prediction[:, step], target[:, step]) for step in range(args.horizon)]
    finite = torch.isfinite(target)
    positive = finite & (target > 0)
    if finite.any():
        unmasked_error = prediction[finite] - target[finite]
        unmasked_mae = float(unmasked_error.abs().mean().item())
        unmasked_rmse = float(unmasked_error.square().mean().sqrt().item())
    else:
        unmasked_mae = unmasked_rmse = None

    result = {
        "dataset": args.dataset,
        "arm": args.arm,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": len(log),
        "best_val_mae": best_val,
        "test_mae": masked[0],
        "test_rmse": masked[1],
        "test_mape": masked[2],
        "per_horizon": per_horizon,
        "test_unmasked_mae": unmasked_mae,
        "test_unmasked_rmse": unmasked_rmse,
        "test_excluded_fraction": 1.0 - float(positive.float().mean().item()),
        "params": total_params,
        "graph_params": graph_params,
        "graph_applications_per_sample": propagation_count,
        "positive_powers_per_sample": positive_propagation_count,
        "train_s_per_epoch": float(np.mean(epoch_times)),
        "test_inference_s": test_inference_s,
        "test_inference_ms_per_sample": 1000 * test_inference_s / len(indices["test"]),
        "peak_mem_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "args": vars(args),
        "log": log,
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    checkpoint = os.path.join(
        os.path.dirname(os.path.abspath(out)),
        "ckpt",
        os.path.basename(out).replace(".json", ".pt"),
    )
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save({"state": best_state, "bias": best_bias, "args": vars(args)}, checkpoint)
    with open(out, "w", encoding="utf-8") as target_file:
        json.dump(result, target_file, indent=2)
    print(
        f"TEST MAE={masked[0]:.3f} RMSE={masked[1]:.3f} "
        f"MAPE={masked[2]:.2f} epochs={len(log)}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS08"), default="PEMS04")
    parser.add_argument("--arm", choices=ARMS + BASELINES, default="proposed")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lb", type=int, default=1)
    parser.add_argument("--lb_lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--graph_dim", type=int, default=32)
    parser.add_argument("--pdm_width", type=int, default=16)
    parser.add_argument("--pdm_depth", type=int, default=2)
    parser.add_argument("--moving_average", type=int, default=5)
    parser.add_argument("--data_dir", default="data_proposal96")
    parser.add_argument("--out", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

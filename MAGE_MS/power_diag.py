"""Measure whether the learned positive spatial operator powers are distinguishable.

Relative Frobenius distances for k in {1,2,4,8}, 3 layers, fixed test origins.
For diagnostics only, materializes N-by-N matrices (PEMS04: N=307; PEMS08: N=170).
Training/inference use the existing factorized O(N) implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_proposal as legacy

ROOT = Path(__file__).resolve().parent
PAIRS = ((1, 2), (2, 4), (4, 8), (1, 8))


def positive_matrix(pool, route):
    e1 = torch.softmax(pool.E1[0], dim=-1)
    e2 = torch.softmax(pool.E2[0], dim=-1)
    return torch.einsum("bne,end,edj->bnj", route, e1, e2)


def distances(matrix):
    powers = {1: matrix}
    powers[2] = powers[1] @ powers[1]
    powers[4] = powers[2] @ powers[2]
    powers[8] = powers[4] @ powers[4]
    ratios = {}
    for a, b in PAIRS:
        numerator = torch.linalg.matrix_norm(powers[a] - powers[b], ord="fro", dim=(-2, -1))
        denominator = torch.linalg.matrix_norm(powers[a], ord="fro", dim=(-2, -1))
        ratios[f"D({a},{b})"] = float((numerator / denominator.clamp_min(1e-12)).mean())
    ratios["max_row_sum_error"] = float((matrix.sum(-1) - 1).abs().max())
    return ratios


def _mean_rows(rows):
    keys = [key for key in rows[0] if key.startswith("D(") or key == "max_row_sum_error"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def diagnose(args):
    source = args.results_dir / f"{args.dataset}_{args.arm}_s{args.seed}.pt"
    if not source.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model_args = SimpleNamespace(**checkpoint["args"])
    model_args.dataset = args.dataset
    model_args.arm = args.arm
    model_args.device = args.device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    data, flow, _, _, indices = legacy._load_data(model_args, device)
    model, _, _, _ = legacy._build_model(args.arm, model_args, data.shape[1], data.shape[2])
    model.load_state_dict(checkpoint["state"])
    model.to(device).eval()
    bias = checkpoint["bias"].to(device)
    test_origins = indices["test"]
    positions = np.linspace(0, len(test_origins) - 1, min(args.samples, len(test_origins)), dtype=int)
    chosen = test_origins[torch.from_numpy(np.unique(positions)).to(device)]
    data_rows = []
    with torch.inference_mode():
        for sample, origin in enumerate(chosen):
            x, _ = legacy._batch(data, flow, origin.view(1), model_args.seq_len, model_args.horizon)
            prompt = model.prompt(x)
            streams = [value + prompt for value in model.temporal(x)]
            for layer, (block, bias_layer) in enumerate(zip(model.blocks, bias), 1):
                route, _ = block._route(streams[0], bias_layer)
                if len(block.pools) != 1:
                    raise ValueError("The power-distance test requires a single shared expert pool")
                matrix = positive_matrix(block.pools[0], route)
                data_rows.append({"sample": sample, "origin": int(origin), "layer": layer,
                                  **distances(matrix)})
                streams, _ = block(streams, bias_layer)
    by_layer = {}
    for layer in (1, 2, 3):
        subset = [row for row in data_rows if row["layer"] == layer]
        by_layer[str(layer)] = _mean_rows(subset)
    result = {
        "dataset": args.dataset, "arm": args.arm, "seed": args.seed,
        "samples": len(chosen), "checkpoint": str(source),
        "description": "Relative Frobenius distance between powers of S_plus, conditioned on test inputs",
        "normalized_error_definition": (
            "D(a,b) = mean over samples of ||S_plus^a - S_plus^b||_F / "
            "max(||S_plus^a||_F, 1e-12); operator diagnostics only, not inference FLOPs."
        ),
        "by_layer": by_layer,
        "mean_across_layers": _mean_rows(data_rows),
        "raw": data_rows,
    }
    output = args.out or (
        ROOT / "results" / "complexity_v1" / "diagnostics"
        / f"power_distances_{args.dataset}_{args.arm}_s{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    for layer, values in by_layer.items():
        print("Layer", layer, values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("PEMS04", "PEMS08"), default="PEMS04")
    p.add_argument("--arm", choices=("shared", "proposed", "reversed"), default="proposed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "eval_proposal96")
    p.add_argument(
        "--out",
        type=Path,
        help="Output JSON path (default: results/complexity_v1/diagnostics/...)",
    )
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    diagnose(args)


if __name__ == "__main__":
    main()

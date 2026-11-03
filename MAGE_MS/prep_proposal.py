"""Prepare flow-only PEMS data for the additive proposal experiments."""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
START_DATES = {"04": "2018-01-01", "08": "2016-07-01"}
SEQ_LEN = 96
HORIZON = 12


def split_indices(length, seq_len=SEQ_LEN, horizon=HORIZON):
    if seq_len < 1 or horizon < 1 or length <= seq_len + horizon:
        raise ValueError("time series is too short for the requested windows")
    boundary1, boundary2 = int(length * 0.6), int(length * 0.8)
    origins = np.arange(seq_len - 1, length - horizon)
    train = origins[origins + horizon < boundary1]
    val = origins[(origins + 1 >= boundary1) & (origins + horizon < boundary2)]
    test = origins[(origins + 1 >= boundary2) & (origins + horizon < length)]
    if not len(train) or not len(val) or not len(test):
        raise ValueError("chronological split produced an empty partition")
    return train, val, test


def calendar_channels(length, start_date):
    timestamps = pd.date_range(start_date, periods=length, freq="5min")
    time_of_day = (
        (timestamps - timestamps.normalize()) / pd.Timedelta("1D")
    ).to_numpy(dtype=np.float32)
    day_of_week = (timestamps.dayofweek.to_numpy(dtype=np.float32) / 7.0)
    return time_of_day, day_of_week


def build_mage_feature_data(data):
    """Append causal moving averages, first difference, and TOD sine/cosine features."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError("expected data with shape (time, nodes, flow+time-of-day+day-of-week)")
    flow = data[..., 0]
    extras = []
    for window in (3, 12):
        cumulative = np.concatenate(
            (np.zeros_like(flow[:1]), np.cumsum(flow, axis=0, dtype=np.float64)), axis=0
        )
        ends = np.arange(1, len(flow) + 1)
        starts = np.maximum(0, ends - window)
        average = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
        extras.append(average.astype(np.float32)[..., None])
    difference = np.zeros_like(flow)
    difference[1:] = flow[1:] - flow[:-1]
    angle = (2 * np.pi * data[..., 1]).astype(np.float32)
    extras.extend((difference[..., None], np.sin(angle)[..., None], np.cos(angle)[..., None]))
    return np.concatenate((flow[..., None], *extras, data[..., 1:3]), axis=-1)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(dataset, raw_path, output_dir, seq_len=SEQ_LEN, horizon=HORIZON):
    dataset = str(dataset).zfill(2)
    if dataset not in START_DATES:
        raise ValueError(f"unsupported dataset {dataset!r}; expected 04 or 08")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"PEMS{dataset} source file not found: {raw_path}")
    with np.load(raw_path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise KeyError(f"{raw_path} does not contain a 'data' array")
        raw = np.asarray(archive["data"], dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] < 1:
        raise ValueError(f"expected source array shaped (time, nodes, features); got {raw.shape}")
    flow = raw[..., :1]
    if not np.isfinite(flow).all():
        raise ValueError("flow source contains non-finite values; clean it before preparation")

    length, nodes, _ = raw.shape
    train_idx, val_idx, test_idx = split_indices(length, seq_len, horizon)
    boundary1, boundary2 = int(length * 0.6), int(length * 0.8)
    mean = float(flow[:boundary1].mean(dtype=np.float64))
    std = float(flow[:boundary1].std(dtype=np.float64))
    std = max(std, 1e-6)
    normalized_flow = (flow - mean) / std
    tod, dow = calendar_channels(length, START_DATES[dataset])
    data = np.concatenate(
        (
            normalized_flow,
            np.broadcast_to(tod[:, None, None], (length, nodes, 1)),
            np.broadcast_to(dow[:, None, None], (length, nodes, 1)),
        ),
        axis=-1,
    ).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "his.npz"),
        data=data,
        mean=np.float32(mean),
        std=np.float32(std),
    )
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        np.save(os.path.join(output_dir, f"idx_{name}.npy"), indices)
    metadata = {
        "dataset": f"PEMS{dataset}",
        "source_file": os.path.basename(raw_path),
        "source_sha256": _sha256(raw_path),
        "source_shape": list(raw.shape),
        "prepared_shape": list(data.shape),
        "input_channels": ["normalized_flow", "time_of_day", "day_of_week"],
        "seq_len": seq_len,
        "horizon": horizon,
        "time_step_minutes": 5,
        "split_ratio": [0.6, 0.2, 0.2],
        "split_boundaries": {
            "train_end_exclusive": boundary1,
            "validation_end_exclusive": boundary2,
            "test_end_exclusive": length,
        },
        "origin_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "flow_mean_train": mean,
        "flow_std_train": std,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("04", "08"), default=("04", "08"))
    parser.add_argument("--out", default="data_proposal96")
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--raw_root", default=os.path.join(ROOT, "raw", "ASTGNN", "data"))
    args = parser.parse_args()
    for dataset in args.datasets:
        raw_path = os.path.join(args.raw_root, f"PEMS{dataset}", f"PEMS{dataset}.npz")
        output_dir = os.path.join(ROOT, args.out, f"PEMS{dataset}")
        metadata = prepare_dataset(
            dataset, raw_path, output_dir, args.seq_len, args.horizon
        )
        print(
            f"PEMS{dataset}: shape={metadata['prepared_shape']} "
            f"origins={metadata['origin_counts']} sha256={metadata['source_sha256']}"
        )


if __name__ == "__main__":
    main()

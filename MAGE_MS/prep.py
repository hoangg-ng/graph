"""Convert ASTGNN's PEMS0X.npz into MAGE/LargeST format.

Mirrors official/data/generate_data_for_training.py: values + time-of-day (fraction of day)
+ day-of-week (/7), chronological 6:2:2 split over sample indices, z-score from the training
span only. Differences: LargeST normalises channel 0 only; we z-score every value channel
(PEMS04/08 carry flow, occupancy, speed on very different scales). Labels are channel 0 (flow).
Splits are built for the largest window we use (seq_len=48) so every input length is evaluated
on exactly the same target timestamps.
"""
import os
import sys
import numpy as np
import pandas as pd

START = {'03': '2018-09-01', '04': '2018-01-01', '07': '2017-05-01', '08': '2016-07-01'}
if len(sys.argv) > 3:
    START = {k: START[k] for k in sys.argv[3].split(',')}
# usage: prep.py [max_seq] [out_dir]  (default 48 -> data/; the long-window study uses 288 -> data_w288/)
MAX_SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 48
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data'
HORIZON = 12
ROOT = os.path.dirname(os.path.abspath(__file__))

for k, start in START.items():
    raw = np.load(f'{ROOT}/raw/ASTGNN/data/PEMS{k}/PEMS{k}.npz')['data'].astype(np.float32)
    T, N, F = raw.shape
    ts = pd.date_range(start, periods=T, freq='5min')
    tod = ((ts - ts.normalize()) / pd.Timedelta('1D')).values.astype(np.float32)
    dow = (ts.dayofweek.values / 7).astype(np.float32)
    idx = np.arange(MAX_SEQ - 1, T - HORIZON)
    n_tr, n_va = round(len(idx) * 0.6), round(len(idx) * 0.2)
    idx_tr, idx_va, idx_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    train_span = raw[:idx_va[0] - MAX_SEQ + 1]  # no val/test timestep leaks into the stats
    mean, std = train_span.mean((0, 1)), train_span.std((0, 1))
    vals = (raw - mean) / std
    data = np.concatenate([vals, np.broadcast_to(tod[:, None, None], (T, N, 1)),
                           np.broadcast_to(dow[:, None, None], (T, N, 1))], -1).astype(np.float32)
    out = f'{ROOT}/{OUT}/PEMS{k}'
    os.makedirs(out, exist_ok=True)
    np.savez(f'{out}/his.npz', data=data, mean=mean[0], std=std[0])
    for name, ix in [('train', idx_tr), ('val', idx_va), ('test', idx_te)]:
        np.save(f'{out}/idx_{name}.npy', ix)
    print(f'PEMS{k}: T={T} N={N} F={F}+2 | train/val/test {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}'
          f' | flow mean {mean[0]:.1f} std {std[0]:.1f}')

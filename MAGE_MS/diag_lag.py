"""Does the way traffic propagates between road neighbours depend on time scale and on traffic state?

Data only (no model), training span. Flow is de-seasonalised per node by subtracting its mean for
the same 5-min slot of the week, then split into a slow part (centred 1-hour moving average) and a
fast part (the rest). For every road edge (from -> to, ASTGNN list) we compute the correlation of
the `from` series at time t with the `to` series at t + k, k = -6..6 (+-30 min), and average over
edges. Reported per band, and for the fast band separately for times when the `from` sensor is in
free flow (speed >= 40) vs congested (< 40).
Writes results/diag_lag.json.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
K = np.arange(-6, 7)
out = {}
for ds in ['PEMS08', 'PEMS04']:
    raw = np.load(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.npz')['data'].astype(np.float64)
    tr = np.load(f'{ROOT}/data/{ds}/idx_train.npy')
    flow, speed = raw[:tr[-1] + 1, :, 0], raw[:tr[-1] + 1, :, 2]
    T, N = flow.shape
    slot = np.arange(T) % (288 * 7)
    seas = np.zeros((288 * 7, N))
    for s in range(288 * 7):
        seas[s] = flow[slot == s].mean(0)
    r = flow - seas[slot]
    ker = np.ones(12) / 12
    slow = np.stack([np.convolve(r[:, i], ker, mode='same') for i in range(N)], 1)
    fast = r - slow
    ed = np.loadtxt(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.csv', delimiter=',', skiprows=1)
    ed = ed[ed[:, 0] != ed[:, 1]]
    a, b, dist = ed[:, 0].astype(int), ed[:, 1].astype(int), ed[:, 2]

    def xcorr(x, sel=None):
        res = []
        for k in K:
            lo, hi = max(0, -k), T - max(0, k)
            xa, xb = x[lo:hi, a], x[lo + k:hi + k, b]
            if sel is not None:
                m = sel[lo:hi, a]
                cs = []
                for j in range(len(a)):
                    mj = m[:, j]
                    if mj.sum() < 200:
                        continue
                    u, v = xa[mj, j], xb[mj, j]
                    cs.append(np.corrcoef(u, v)[0, 1])
                res.append(float(np.nanmean(cs)) if cs else float('nan'))
            else:
                xa = (xa - xa.mean(0)) / (xa.std(0) + 1e-9); xb = (xb - xb.mean(0)) / (xb.std(0) + 1e-9)
                res.append(float(np.nanmean((xa * xb).mean(0))))
        return res
    o = dict(lags_min=(K * 5).tolist(), n_edges=int(len(a)), median_edge_km=float(np.median(dist)) / 1000,
             slow=xcorr(slow), fast=xcorr(fast),
             fast_free=xcorr(fast, speed >= 40), fast_congested=xcorr(fast, speed < 40),
             share_congested=float((speed < 40).mean()))
    for key in ('slow', 'fast', 'fast_free', 'fast_congested'):
        c = np.array(o[key]); o[key + '_peak_lag_min'] = int(K[np.nanargmax(c)] * 5)
    out[ds] = o
    print(ds, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in o.items()}), flush=True)
json.dump(out, open(f'{ROOT}/results/diag_lag.json', 'w'), indent=1)

"""How much does space matter, and does MAGE leave spatial information unused? (T = 12, model 1)

1. Value of neighbours in the data itself (no deep model): per node, a ridge regression predicting
   flow at horizon h from (a) its own last 12 steps + its historical-average value for the target
   slot, vs (b) the same plus the last 12 steps of its road neighbours (ASTGNN edge list, both
   directions). Fit on train, MAE on test. The gap (a) - (b) is the information neighbours add to
   a node's own history, as seen by a linear model.
2. Leftover spatial signal in MAGE's errors (seed 0): correlation of residuals between road-adjacent
   node pairs vs random pairs, at the same time and with upstream leading downstream by 1-3 steps
   (residual of node i for target time t vs residual of node j for target time t + k).
Writes results/diag_space.json.
"""
import json
import os

import numpy as np

import post

ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
out = {}
for ds in ['PEMS08', 'PEMS04']:
    raw = np.load(f'{ROOT}/{post.RAW[ds]}')['data'][..., 0].astype(np.float64)
    N = raw.shape[1]
    ed = np.loadtxt(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.csv', delimiter=',', skiprows=1)[:, :2].astype(int)
    nb = [sorted(set(ed[ed[:, 0] == i, 1]) | set(ed[ed[:, 1] == i, 0])) for i in range(N)]
    tr = np.load(f'{ROOT}/data/{ds}/idx_train.npy'); te = np.load(f'{ROOT}/data/{ds}/idx_test.npy')
    mu, sd = raw[:tr[-1] + 1].mean(), raw[:tr[-1] + 1].std()
    z = (raw - mu) / sd
    slot = np.arange(len(raw)) % (288 * 7)
    ha = np.zeros((288 * 7, N))
    for k in range(288 * 7):
        ix = np.arange(tr[-1] + 1); ix = ix[slot[ix] == k]
        ha[k] = z[ix].mean(0)
    lags = np.arange(-11, 1)

    res = {}
    for h in (1, 6, 12):
        mae_own, mae_nb = [], []
        for i in range(N):
            def feats(ix, use_nb):
                f = [z[ix[:, None] + lags, i], ha[slot[ix + h], i][:, None], np.ones((len(ix), 1))]
                if use_nb and nb[i]:
                    f.append(z[ix[:, None] + lags][:, :, nb[i]].reshape(len(ix), -1))
                return np.concatenate(f, 1)
            ytr, yte = z[tr + h, i], raw[te + h, i]
            m = yte != 0
            for use_nb, acc in ((False, mae_own), (True, mae_nb)):
                A = feats(tr, use_nb)
                w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)
                p = feats(te, use_nb) @ w * sd + mu
                acc.append((np.abs(p - yte)[m].sum(), m.sum()))
        mo, mn = [sum(a for a, _ in acc) / sum(c for _, c in acc) for acc in (mae_own, mae_nb)]
        res[h] = dict(ridge_own=mo, ridge_own_plus_neighbours=mn, gain=mo - mn)
        print(ds, 'h', h, res[h], flush=True)

    P, _ = post.predict(*post.load(ds, 'linear', 0), ds)
    Y = raw[te[:, None] + np.arange(1, 13)]
    R = (P.numpy() - Y) * (Y != 0)
    pairs = np.array([(a, b) for a, b in ed if a != b])
    rnd = rng.integers(0, N, size=(2000, 2)); rnd = rnd[rnd[:, 0] != rnd[:, 1]]

    def corr(r, pr, k):
        x = r[:len(r) - k] if k else r
        y = r[k:]
        a, b = x[:, pr[:, 0]], y[:, pr[:, 1]]
        a = (a - a.mean(0)) / (a.std(0) + 1e-9); b = (b - b.mean(0)) / (b.std(0) + 1e-9)
        return float(np.nanmean((a * b).mean(0)))
    rc = {}
    for h in (1, 12):
        r = R[:, h - 1]
        rc[h] = {f'lag{k}': dict(road_edge_up_to_down=corr(r, pairs, k), road_edge_down_to_up=corr(r, pairs[:, ::-1], k),
                                 random_pairs=corr(r, rnd, k)) for k in (0, 1, 3)}
    print(ds, 'residual corr', json.dumps(rc), flush=True)
    out[ds] = dict(neighbour_value=res, residual_corr=rc, mean_degree=float(np.mean([len(x) for x in nb])),
                   isolated_nodes=int(sum(len(x) == 0 for x in nb)))
json.dump(out, open(f'{ROOT}/results/diag_space.json', 'w'), indent=1)

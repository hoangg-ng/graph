"""Where does MAGE's error come from? (T = 12 checkpoints, model 1, seeds 0-4)

Per dataset, on the test set (masked MAE on raw flow, y != 0):
  - MAE by horizon, vs persistence (last value) and a per-node time-of-week historical average
    (mean flow per node x 5-min slot of the week, training span only)
  - share of total absolute error by regime (speed-based, as in post.py) and by error size
  - concentration over nodes: share of error from the worst 10 % of nodes
  - structure of the residual: mean pairwise correlation of errors across nodes (common shocks)
    and lag-1 autocorrelation over time (predictable leftovers)
  - seed-ensemble: MAE of the 5-seed average and how much the seeds disagree
Writes results/diag.json and prints a summary.
"""
import json
import os
import sys

import numpy as np
import torch

sys.argv = sys.argv[:1]
import post  # noqa: E402  (reuses load/predict/regimes)

ROOT = os.path.dirname(os.path.abspath(__file__))
out = {}
for ds in ['PEMS08', 'PEMS04']:
    raw = np.load(f'{ROOT}/{post.RAW[ds]}')['data'][..., 0]
    te = np.load(f'{ROOT}/data/{ds}/idx_test.npy'); tr = np.load(f'{ROOT}/data/{ds}/idx_train.npy')
    yo = te[:, None] + np.arange(1, 13)
    Y = torch.from_numpy(raw[yo]).float()                          # (S, 12, n)
    mask = Y != 0

    def mae(P, sel=None):
        m = mask if sel is None else mask & sel
        return (P - Y).abs()[m].mean().item()

    Ps = [post.predict(*post.load(ds, 'linear', s), ds)[0] for s in range(5)]
    P = Ps[0]
    ens = torch.stack(Ps).mean(0)
    seed_sd = torch.stack(Ps).std(0)[mask].mean().item()

    # baselines
    persist = torch.from_numpy(np.repeat(raw[te][:, None], 12, 1)).float()
    slot = np.arange(raw.shape[0]) % (288 * 7)
    span = np.arange(tr[-1] + 1)
    ha_tab = np.zeros((288 * 7, raw.shape[1]))
    for k in range(288 * 7):
        ix = span[slot[span] == k]
        v = raw[ix]; ha_tab[k] = np.where((v != 0).sum(0) > 0, v.sum(0) / np.maximum((v != 0).sum(0), 1), 0)
    ha = torch.from_numpy(ha_tab[slot[yo]]).float()

    err = (P - Y).abs() * mask
    tot = err.sum().item()
    lab, _ = post.regimes(ds)
    reg = {k: dict(share_points=float(v.mean()), share_error=(err * torch.from_numpy(v)[:, None, :]).sum().item() / tot,
                   mae=mae(P, torch.from_numpy(v)[:, None, :].expand_as(Y)))
           for k, v in lab.items() if k in ('free flow', 'onset', 'recovery', 'congested')}
    e = err[mask]
    q = torch.quantile(e[torch.randperm(len(e))[:2_000_000]], torch.tensor([0.5, 0.9, 0.99]))
    big = {f'top{p}%': e[e >= torch.quantile(e[torch.randperm(len(e))[:2_000_000]], 1 - p / 100)].sum().item() / tot
           for p in (1, 5, 10)}
    node_err = err.sum((0, 1))
    worst = node_err.sort(descending=True).values
    nshare = worst[: max(1, len(worst) // 10)].sum().item() / tot

    r = ((P - Y) * mask)[:, 0]                                      # horizon-1 residual (S, n)
    r = r - r.mean(0)
    c = torch.corrcoef(r.T); n = c.shape[0]
    cross = ((c.sum() - n) / (n * n - n)).item()
    # lag-1 autocorrelation of horizon-1 residual along consecutive test samples
    ac = np.nanmean([np.corrcoef(r[:-1, j], r[1:, j])[0, 1] for j in range(n)])

    out[ds] = dict(
        mae_by_h={h: dict(mage=(P[:, h - 1] - Y[:, h - 1]).abs()[mask[:, h - 1]].mean().item(),
                          persistence=(persist[:, h - 1] - Y[:, h - 1]).abs()[mask[:, h - 1]].mean().item(),
                          hist_avg=(ha[:, h - 1] - Y[:, h - 1]).abs()[mask[:, h - 1]].mean().item())
                  for h in (1, 3, 6, 12)},
        mae=dict(mage=mae(P), mage_5seed_ensemble=mae(ens), persistence=mae(persist), hist_avg=mae(ha)),
        seed_disagreement_mean_abs_sd=seed_sd,
        error_quantiles_p50_p90_p99=q.tolist(), error_share_of_largest=big,
        worst10pct_nodes_error_share=nshare, regimes=reg,
        residual_cross_node_corr=cross, residual_lag1_autocorr=float(ac))
    print(ds, json.dumps(out[ds], indent=1), flush=True)
json.dump(out, open(f'{ROOT}/results/diag.json', 'w'), indent=1)

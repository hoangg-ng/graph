"""RQ test (2026-11-08): does a learned adaptive graph beat the network-wide average only for sensors that share
local dependence beyond the common network signal?  No training: uses the saved learned and average-trained models.

Per sensor i, from the TRAINING span only:
  a_i     = flow minus the sensor's usual slot-of-week flow (anomaly), standardised
  common  = R^2 of a_i on the network-average anomaly (CCE-style common-signal share)
  r_i     = residual of a_i after regressing on the network-average anomaly (current and lag 1)
  local   = mean |corr(r_i, r_j)| over the sensor's 5 most correlated other sensors   (leftover local dependence)
Per sensor i, on the TEST set: gain_i = MAE_i(average-trained model) - MAE_i(learned model)  (> 0: graph helps)
Reports Spearman(gain, local) and Spearman(gain, common) per model and dataset, and the mean gain for the sensors in
the top vs bottom quartile of `local`.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import reimpl
import reimpl2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sensor_stats(d, mean, std, T_end):
    flow = d[:T_end, :, 0] * std + mean
    N = flow.shape[1]
    sw = (np.rint(d[:T_end, 0, -1] * 7).astype(int) % 7) * 288 + np.rint(d[:T_end, 0, -2] * 288).astype(int) % 288
    usual = np.zeros((2016, N))
    for i in range(2016):
        m = sw == i
        if m.any():
            usual[i] = flow[m].mean(0)
    a = flow - usual[sw]; a = (a - a.mean(0)) / (a.std(0) + 1e-6)
    g = a.mean(1)
    X = np.c_[np.ones(len(g) - 1), g[1:], g[:-1]]                       # network average, current and lag 1
    B = np.linalg.lstsq(X, a[1:], rcond=None)[0]
    r = a[1:] - X @ B
    common = 1 - r.var(0) / a[1:].var(0)
    R = np.corrcoef(r.T); np.fill_diagonal(R, 0); R = np.abs(np.nan_to_num(R))
    local = np.sort(R, 1)[:, -5:].mean(1)
    return common, local


def per_sensor_mae(model_name, ds, tag, data, flow, te, mean, std, N, C, A):
    st = torch.load(f'{HERE}/results/{model_name}_{ds}_s0{tag}.pt', map_location=dev)
    if model_name == 'gwnet':
        m = reimpl.GWNet(N).to(dev)
        run = lambda x: m(torch.nn.functional.pad(torch.stack([x[..., 0], x[..., C - 2]], 1).transpose(2, 3), (1, 0, 0, 0)))[..., 0]
    elif model_name == 'agcrn':
        m = reimpl.AGCRN(N, embed_dim={'PEMS04': 10, 'PEMS08': 2}[ds]).to(dev); run = lambda x: m(x[..., :1])
    elif model_name == 'mtgnn':
        m = reimpl2.MTGNN(N).to(dev)
        run = lambda x: m(torch.stack([x[..., 0], x[..., C - 2]], 1).transpose(2, 3))[..., 0]
    elif model_name == 'staeformer':
        m = reimpl2.STAEformer(N).to(dev)
        run = lambda x: m(torch.stack([x[..., 0], x[..., C - 2], (x[..., C - 1] * 7).round()], -1))
    else:
        m = reimpl2.ASTGCN(N, A).to(dev); run = lambda x: m(x[..., :1].permute(0, 2, 3, 1))
    m.load_state_dict(st); m.eval()
    m.fixed = 'mean' if tag == '_mean' else None
    xo, yo = torch.arange(-11, 1, device=dev), torch.arange(1, 13, device=dev)
    err = torch.zeros(N, device=dev); cnt = torch.zeros(N, device=dev)
    with torch.no_grad():
        for ix in te.split(64):
            p = run(data[ix[:, None] + xo]) * std + mean; y = flow[ix[:, None] + yo]
            msk = (y != 0).float()
            err += ((p - y).abs() * msk).sum((0, 1)); cnt += msk.sum((0, 1))
    return (err / cnt.clamp_min(1)).cpu().numpy()


out = {}
print('model         data   rho(gain, local)  rho(gain, common)   mean gain: top-25% local | bottom-25% local | all')
for ds in ('PEMS08', 'PEMS04'):
    z = np.load(f'{ROOT}/data_w288/{ds}/his.npz'); d = z['data']; mean, std = float(z['mean']), float(z['std'])
    T_end = int(np.load(f'{ROOT}/data_w288/{ds}/idx_train.npy')[-1]) + 1
    common, local = sensor_stats(d, mean, std, T_end)
    data = torch.from_numpy(d).to(dev); flow = data[..., 0] * std + mean
    flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
    te = torch.from_numpy(np.load(f'{ROOT}/data_w288/{ds}/idx_test.npy')).to(dev)
    N, C = d.shape[1], d.shape[2]
    e = pd.read_csv(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.csv'); A = np.zeros((N, N)); A[e.iloc[:, 0].values, e.iloc[:, 1].values] = 1
    out[ds] = {'local_mean': float(local.mean()), 'common_mean': float(common.mean())}
    for mdl in ('gwnet', 'agcrn', 'mtgnn', 'staeformer', 'astgcn'):
        mae_l = per_sensor_mae(mdl, ds, '', data, flow, te, mean, std, N, C, A)
        mae_m = per_sensor_mae(mdl, ds, '_mean', data, flow, te, mean, std, N, C, A)
        gain = mae_m - mae_l
        q1, q3 = np.quantile(local, [.25, .75])
        r_l, r_c = spearmanr(gain, local).statistic, spearmanr(gain, common).statistic
        out[ds][mdl] = dict(rho_local=float(r_l), rho_common=float(r_c), gain_top=float(gain[local >= q3].mean()),
                            gain_bottom=float(gain[local <= q1].mean()), gain_all=float(gain.mean()))
        print(f'{mdl:12s}  {ds}   {r_l:+.2f}              {r_c:+.2f}               '
              f'{gain[local >= q3].mean():+.2f} | {gain[local <= q1].mean():+.2f} | {gain.mean():+.2f}')
    print(f'  {ds}: mean leftover local dependence {local.mean():.3f}, mean common-signal share {common.mean():.3f}')
json.dump(out, open(f'{HERE}/results/rq_test.json', 'w'), indent=1)

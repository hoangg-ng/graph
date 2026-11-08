"""What do learned graphs / spatial attention encode, if not roads? (2026-11-07)

For each trained model (others/results/*_s0.pt, plus MAGE with the graph-collapse fix from results/hier/ckpt), take
the learned spatial mixing matrix M (N x N; attention models: mean over 20 test batches, heads, layers, steps) and
compare it with pairwise sensor properties computed on the TRAINING span only:
  profile   : correlation of the two sensors' average weekday daily flow profiles   ("behave alike")
  commute   : -|log(AM peak / PM peak)_i - (...)_j|                                    (same commute direction)
  volume    : -|log mean flow_i - log mean flow_j|                                     (similar traffic volume)
  cofluct   : correlation of the anomalies (flow minus the sensor's usual slot-of-week flow)  (move together)
  road      : -(road hop distance), unreachable pairs = max + 1                        (physically close)
Two scores per (model, property):
  rho : Spearman rank correlation between M_ij and the property over all off-diagonal pairs
  top5: mean percentile of the property over each sensor's 5 strongest learned links (0.50 = random, 1.0 = best)
Usage: python what_graphs_learn.py   (writes results/what_graphs_learn.json and prints a table)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import reimpl  # noqa: E402
import reimpl2  # noqa: E402

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def properties(ds):
    z = np.load(f'{ROOT}/data_w288/{ds}/his.npz')
    d = z['data']; mean, std = float(z['mean']), float(z['std'])
    tr = np.load(f'{ROOT}/data_w288/{ds}/idx_train.npy')
    T_end = int(tr[-1]) + 1
    flow = d[:T_end, :, 0] * std + mean
    slot = np.rint(d[:T_end, 0, -2] * 288).astype(int) % 288
    dow = np.rint(d[:T_end, 0, -1] * 7).astype(int) % 7
    N = flow.shape[1]
    wd = dow < 5
    prof = np.stack([flow[(slot == i) & wd].mean(0) for i in range(288)])               # 288 x N
    P = {'profile': np.corrcoef(prof.T)}
    am, pm = prof[84:108].max(0), prof[192:216].max(0)                                    # 07-09 h, 16-18 h
    r = np.log((am + 1) / (pm + 1)); P['commute'] = -np.abs(r[:, None] - r[None, :])
    v = np.log(flow.mean(0) + 1); P['volume'] = -np.abs(v[:, None] - v[None, :])
    sw = dow * 288 + slot
    usual = np.zeros((2016, N))
    for i in range(2016):
        m = sw == i
        if m.any():
            usual[i] = flow[m].mean(0)
    an = flow - usual[sw]
    P['cofluct'] = np.corrcoef(an.T)
    e = pd.read_csv(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.csv')
    A = np.zeros((N, N)); A[e.iloc[:, 0], e.iloc[:, 1]] = 1; A = np.maximum(A, A.T)
    hop = shortest_path(A, unweighted=True, directed=False)
    hop[~np.isfinite(hop)] = np.nanmax(hop[np.isfinite(hop)]) + 1
    P['road'] = -hop
    return {k: np.nan_to_num(v) for k, v in P.items()}


def learned_matrix(model, ds, N):
    st = torch.load(f'{HERE}/results/{model}_{ds}_s0.pt', map_location=dev)
    if model == 'gwnet':
        m = reimpl.GWNet(N); m.load_state_dict(st); return m.adjacency().detach().cpu().numpy()
    if model == 'agcrn':
        m = reimpl.AGCRN(N, embed_dim={'PEMS04': 10, 'PEMS08': 2}[ds]); m.load_state_dict(st)
        return m.adjacency().detach().cpu().numpy()
    if model == 'mtgnn':
        m = reimpl2.MTGNN(N); m.load_state_dict(st); m.eval()
        with torch.no_grad():
            return m.gc.full(m.idx).cpu().numpy()                 # dense scores before the top-20 mask
    # attention models: run 20 test batches and average the recorded spatial maps
    z = np.load(f'{ROOT}/data_w288/{ds}/his.npz'); data = torch.from_numpy(z['data']).to(dev); C = data.shape[2]
    te = torch.from_numpy(np.load(f'{ROOT}/data_w288/{ds}/idx_test.npy')).to(dev)
    xo = torch.arange(-11, 1, device=dev)
    if model == 'staeformer':
        m = reimpl2.STAEformer(N).to(dev)
        prep = lambda x: torch.stack([x[..., 0], x[..., C - 2], (x[..., C - 1] * 7).round()], -1)
    else:
        e = pd.read_csv(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.csv')
        A = np.zeros((N, N)); A[e.iloc[:, 0].values, e.iloc[:, 1].values] = 1
        m = reimpl2.ASTGCN(N, A).to(dev)
        prep = lambda x: x[..., :1].permute(0, 2, 3, 1)
    m.load_state_dict(st); m.eval(); m.record = []
    with torch.no_grad():
        for i, ix in enumerate(te.split(64)):
            if i == 20:
                break
            m(prep(data[ix[:, None] + xo]))
    return torch.stack([mm for _, mm in m.record]).mean(0).numpy()


def mage_fixed(ds):
    st = torch.load(f'{ROOT}/results/hier/ckpt/{ds}_T96_mage_mage_f1_h0_s0.pt', map_location='cpu')['state']
    Ms = []
    for b in (1, 2, 3):
        E1 = torch.softmax(st[f'backbone{b}.attn.E1'][0], -1); E2 = torch.softmax(st[f'backbone{b}.attn.E2'][0], -1)
        Ms.append(torch.einsum('eid,edj->ij', E1, E2).numpy() / E1.shape[0])
    return np.mean(Ms, 0)


def scores(M, P):
    N = M.shape[0]; off = ~np.eye(N, dtype=bool)
    out = {}
    top = np.argsort(-np.where(off, M, -np.inf), 1)[:, :5]
    for k, S in P.items():
        rho = spearmanr(M[off], S[off]).statistic
        pct = np.empty_like(S)
        for i in range(N):                                         # percentile of each pair within the sensor's row
            row = S[i].copy(); row[i] = np.nan
            pct[i] = (rankdata(np.nan_to_num(row, nan=-np.inf)) - 1) / (N - 1)
        out[k] = dict(rho=float(rho), top5=float(np.mean([pct[i, top[i]].mean() for i in range(N)])))
    return out


res = {}
names = {'gwnet': 'Graph WaveNet', 'agcrn': 'AGCRN', 'mtgnn': 'MTGNN', 'staeformer': 'STAEformer', 'astgcn': 'ASTGCN',
         'mage_fixed': 'MAGE, graph fix'}
for ds in ('PEMS08', 'PEMS04'):
    P = properties(ds); N = P['road'].shape[0]
    off = ~np.eye(N, dtype=bool)
    res[ds] = {'property_vs_road_rho': {k: float(spearmanr(P[k][off], P['road'][off]).statistic) for k in P if k != 'road'}}
    for mdl in ('gwnet', 'agcrn', 'mtgnn', 'staeformer', 'astgcn', 'mage_fixed'):
        M = mage_fixed(ds) if mdl == 'mage_fixed' else learned_matrix(mdl, ds, N)
        res[ds][mdl] = scores(M, P)
json.dump(res, open(f'{HERE}/results/what_graphs_learn.json', 'w'), indent=1)

props = ['road', 'profile', 'commute', 'volume', 'cofluct']
for ds in res:
    print(f'\n{ds}  (rho = Spearman with learned link strength | top5 = mean percentile of the 5 strongest links, 0.50 = random)')
    print('  how the properties relate to road distance (rho): ' +
          ', '.join(f'{k} {v:+.2f}' for k, v in res[ds]['property_vs_road_rho'].items()))
    print(f'  {"model":16s}' + ''.join(f'{p:>16s}' for p in props))
    for mdl, sc in res[ds].items():
        if mdl == 'property_vs_road_rho':
            continue
        print(f'  {names[mdl]:16s}' + ''.join(f'{sc[p]["rho"]:+7.2f} / {sc[p]["top5"]:.2f}' for p in props))

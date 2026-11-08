"""What do per-sensor vectors encode? (2026-11-08, no training)

Decodes sensor properties (training span) from each model's per-sensor parameters with 5-fold cross-validated ridge
regression (R^2 on held-out sensors; 0 = no information, 1 = perfect):
  MAGE            : sensor embedding (prompt.spatial_prompt, N x 128)
  AGCRN           : node embeddings (N x embed_dim; they generate the graph AND per-sensor weights)
  Graph WaveNet   : graph node vectors [nodevec1 | nodevec2^T] (N x 20; used only inside the adaptive graph)
  MTGNN           : graph embeddings [emb1 | emb2] (N x 80)
  STAEformer      : adaptive embedding averaged over steps (N x 80)
  random          : N x 20 Gaussian (reference)
Properties: log volume, commute direction log(AM peak / PM peak), daily-profile shape (PC1, PC2 of the normalised
weekday profile), loading on the network-average anomaly (slope) and common-signal share (R^2).
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def props(ds):
    z = np.load(f'{ROOT}/data_w288/{ds}/his.npz'); d = z['data']; mean, std = float(z['mean']), float(z['std'])
    T = int(np.load(f'{ROOT}/data_w288/{ds}/idx_train.npy')[-1]) + 1
    flow = d[:T, :, 0] * std + mean; N = flow.shape[1]
    slot = np.rint(d[:T, 0, -2] * 288).astype(int) % 288; dow = np.rint(d[:T, 0, -1] * 7).astype(int) % 7
    prof = np.stack([flow[(slot == i) & (dow < 5)].mean(0) for i in range(288)])           # 288 x N
    pn = (prof - prof.mean(0)) / (prof.std(0) + 1e-6)
    U, S, Vt = np.linalg.svd(pn - pn.mean(1, keepdims=True), full_matrices=False)
    sw = dow * 288 + slot; usual = np.zeros((2016, N))
    for i in range(2016):
        m = sw == i
        if m.any():
            usual[i] = flow[m].mean(0)
    a = flow - usual[sw]; a = (a - a.mean(0)) / (a.std(0) + 1e-6); g = a.mean(1)
    slope = (a * g[:, None]).sum(0) / (g ** 2).sum()
    share = 1 - ((a - g[:, None] * slope) ** 2).mean(0) / a.var(0)
    return {'log volume': np.log(flow.mean(0) + 1),
            'commute (AM/PM)': np.log((prof[84:108].max(0) + 1) / (prof[192:216].max(0) + 1)),
            'profile shape PC1': Vt[0], 'profile shape PC2': Vt[1],
            'loading on network signal': slope, 'network-signal share': share}


def vectors(ds, N):
    v = {}
    ck = {'PEMS04': f'{ROOT}/results/pems04/ckpt/PEMS04_T96_mage_mage_s0.pt',
          'PEMS08': f'{ROOT}/results/hier/ckpt/PEMS08_T96_mage_mage_f0_h0_s0.pt'}[ds]
    v['MAGE sensor embedding'] = torch.load(ck, map_location='cpu')['state']['prompt.spatial_prompt'].numpy()
    st = torch.load(f'{HERE}/results/agcrn_{ds}_s0.pt', map_location='cpu'); v['AGCRN node embedding'] = st['node_embeddings'].numpy()
    st = torch.load(f'{HERE}/results/gwnet_{ds}_s0.pt', map_location='cpu')
    v['Graph WaveNet graph vectors'] = np.c_[st['nodevec1'].numpy(), st['nodevec2'].numpy().T]
    st = torch.load(f'{HERE}/results/mtgnn_{ds}_s0.pt', map_location='cpu')
    v['MTGNN graph embeddings'] = np.c_[st['gc.emb1.weight'].numpy(), st['gc.emb2.weight'].numpy()]
    st = torch.load(f'{HERE}/results/staeformer_{ds}_s0.pt', map_location='cpu')
    v['STAEformer adaptive embedding'] = st['adaptive_embedding'].numpy().mean(0)
    v['random (reference)'] = np.random.default_rng(0).normal(size=(N, 20))
    return v


res = {}
for ds in ('PEMS08', 'PEMS04'):
    P = props(ds); N = len(P['log volume']); V = vectors(ds, N); res[ds] = {}
    print(f'\n{ds}: cross-validated R^2 of predicting each sensor property from the per-sensor vectors')
    print(f'  {"vectors":30s}' + ''.join(f'{k[:14]:>16s}' for k in P))
    for name, X in V.items():
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        row = {}
        for k, y in P.items():
            pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), X, y, cv=5)
            row[k] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        res[ds][name] = row
        print(f'  {name:30s}' + ''.join(f'{row[k]:16.2f}' for k in P))
json.dump(res, open(f'{HERE}/results/identity_check.json', 'w'), indent=1)

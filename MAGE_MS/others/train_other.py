"""Do other adaptive-graph models' learned graphs collapse to the sensor mean? (2026-11-07)

Trains Graph WaveNet (adaptive graph only) or AGCRN, reimplemented in reimpl.py from the official code,
on our PEMS0X data (data_w288 split, 12 -> 12, the same test origins as our MAGE T = 12 runs), with each paper's own
recipe, then measures on the learned graph:
  - row entropy of the adjacency / log N (1.0 = every row is the uniform mean)
  - road alignment: weight on road neighbours vs a random pair, and share of each sensor's top-5 links that are road edges
  - test MAE when the learned graph is replaced by the exact uniform mean at inference (no retraining)
Usage: python train_other.py --model gwnet|agcrn --dataset PEMS08 --seed 0 [--device cuda:0]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
p = argparse.ArgumentParser()
p.add_argument('--model', required=True, choices=['gwnet', 'agcrn'])
p.add_argument('--dataset', default='PEMS08')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='cuda:0')
p.add_argument('--sensor_emb', type=int, default=0, help='identity test: add a learned sensor embedding (gwnet, astgcn)')
p.add_argument('--mixing', default='learned', choices=['learned', 'mean', 'self', 'random'],
               help='train with the spatial mixing learned (default), fixed to the sensor mean, or off (self only)')
p.add_argument('--epochs', type=int, default=100)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # A100: faster matmuls (2026-11-08)
dev = torch.device(a.device)

z = np.load(f'{ROOT}/data_w288/{a.dataset}/his.npz')
data = torch.from_numpy(z['data']).to(dev)
mean, std = float(z['mean']), float(z['std'])
flow = data[..., 0] * std + mean
flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
idx = {k: torch.from_numpy(np.load(f'{ROOT}/data_w288/{a.dataset}/idx_{k}.npy')).to(dev) for k in ('train', 'val', 'test')}
N, C = data.shape[1], data.shape[2]
xo, yo = torch.arange(-11, 1, device=dev), torch.arange(1, 13, device=dev)


def masked_mae(pred, y):
    m = (y != 0).float(); m = m / m.mean()
    return ((pred - y).abs() * m).mean()


UNIFORM = {'on': False}   # inference switch: replace the learned graph by the exact uniform mean
import reimpl   # own reimplementations following the official code (running the cloned repos was not permitted)
if a.model == 'gwnet':
    model = reimpl.GWNet(N, in_dim=2, out_dim=12, sensor_emb=bool(a.sensor_emb)).to(dev)                  # official defaults, --aptonly
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    clip, patience = 5, None

    def run(x):                                   # x: (b, 12, N, C) -> (b, 12, N) raw flow
        model.uniform = UNIFORM['on']
        inp = torch.stack([x[..., 0], x[..., C - 2]], 1).transpose(2, 3)   # (b, 2, N, 12): flow + time of day
        inp = F.pad(inp, (1, 0, 0, 0))                                       # as in the official engine
        return model(inp)[..., 0] * std + mean                              # (b, 12, N)
else:
    emb = {'PEMS04': 10, 'PEMS08': 2}.get(a.dataset, 10)                   # official configs
    model = reimpl.AGCRN(N, input_dim=1, hidden=64, horizon=12, num_layers=2, embed_dim=emb, cheb_k=2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0)
    clip, patience = None, 15

    def run(x):
        model.uniform = UNIFORM['on']
        return model(x[..., :1]) * std + mean


def adjacency():
    model.uniform = False
    return model.adjacency()


model.fixed = None if a.mixing == 'learned' else a.mixing
if a.mixing == 'random':
    from reimpl import make_random_mix
    model.register_buffer('random_mix', make_random_mix(N, seed=1000 + a.seed).to(dev))
TAG = ('' if a.mixing == 'learned' else '_' + a.mixing) + ('_emb' if a.sensor_emb else '')   # result-file suffix


def batch(ix):
    return data[ix[:, None] + xo], flow[ix[:, None] + yo]


def evaluate(split):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for ix in idx[split].split(64):
            x, y = batch(ix); P.append(run(x)); Y.append(y)
    return masked_mae(torch.cat(P), torch.cat(Y)).item()


best, best_state, wait, log, t0 = 1e9, None, 0, [], time.time()
for ep in range(a.epochs):
    model.train(); L = []
    for ix in idx['train'][torch.randperm(len(idx['train']), device=dev)].split(64):
        x, y = batch(ix)
        loss = masked_mae(run(x), y)
        opt.zero_grad(); loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); L.append(loss.item())
    v = evaluate('val'); log.append((ep + 1, float(np.mean(L)), v))
    if v < best:
        best, wait, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
    else:
        wait += 1
    print(f'ep {ep + 1} train {np.mean(L):.3f} val {v:.3f} best {best:.3f} {(time.time() - t0) / (ep + 1):.1f}s/ep', flush=True)
    if patience and wait >= patience:
        break
model.load_state_dict(best_state)

# ---------------------------------------------------------------- graph analysis
test = evaluate('test')
UNIFORM['on'] = True; test_uniform = evaluate('test'); UNIFORM['on'] = False
with torch.no_grad():
    A = adjacency().cpu().numpy()
H = (-(A * np.log(np.clip(A, 1e-12, None))).sum(1) / math.log(N)).mean()
e = pd.read_csv(f'{ROOT}/raw/ASTGNN/data/{a.dataset}/{a.dataset}.csv')
s_, d_ = e.iloc[:, 0].values, e.iloc[:, 1].values
adj = np.zeros((N, N), bool); adj[s_, d_] = adj[d_, s_] = True
off = A[~np.eye(N, dtype=bool)].mean()
nb = np.r_[A[s_, d_], A[d_, s_]].mean() / off
top = np.argsort(-A, 1)[:, :6]
hit = np.mean([adj[i, [j for j in top[i] if j != i][:5]].mean() for i in range(N)])
chance = adj.sum() / N / (N - 1)
res = dict(mixing=a.mixing, model=a.model, dataset=a.dataset, seed=a.seed, test_mae=test, test_mae_graph_to_mean=test_uniform,
           best_val=best, epochs=len(log), graph_entropy=float(H), road_neighbour_weight_vs_random=float(nb),
           top5_road_share=float(hit), top5_road_chance=float(chance),
           params=sum(t.numel() for t in model.parameters()), log=log)
os.makedirs(f'{HERE}/results', exist_ok=True)
json.dump(res, open(f'{HERE}/results/{a.model}_{a.dataset}_s{a.seed}{TAG}.json', 'w'))
torch.save(best_state, f'{HERE}/results/{a.model}_{a.dataset}_s{a.seed}{TAG}.pt')
print(f'RESULT {a.model} {a.dataset} test {test:.3f} | graph->mean {test_uniform:.3f} ({test_uniform - test:+.3f}) | '
      f'entropy {H:.3f} | road-neighbour weight {nb:.2f}x random | top-5 road {100 * hit:.1f}% (chance {100 * chance:.1f}%)')

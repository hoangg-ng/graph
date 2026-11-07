"""Collapse check, part 2: MTGNN, STAEformer, ASTGCN (reimpl2.py), each with its paper's own recipe, on our PEMS0X
data (data_w288 split, 12 -> 12, the same test origins as our MAGE and train_other.py runs). After training:
  - uniformity of the learned spatial mixing: mean row entropy / log(support size); 1.0 = uniform over the support
    (MTGNN: over the 20 kept links plus self; attention models: over all N sensors, recorded on 20 test batches)
  - road alignment of the (mean) mixing matrix: weight on road neighbours vs a random pair, top-5 links on roads
  - test MAE when the learned mixing is replaced by the exact uniform mean 1/N at inference
Usage: python train_other2.py --model mtgnn|staeformer|astgcn --dataset PEMS08 --seed 0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import reimpl2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
p = argparse.ArgumentParser()
p.add_argument('--model', required=True, choices=['mtgnn', 'staeformer', 'astgcn'])
p.add_argument('--dataset', default='PEMS08')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='cuda:0')
p.add_argument('--epochs', type=int, default=None, help='override the recipe (smoke tests)')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device(a.device)

z = np.load(f'{ROOT}/data_w288/{a.dataset}/his.npz')
data = torch.from_numpy(z['data']).to(dev)
mean, std = float(z['mean']), float(z['std'])
flow = data[..., 0] * std + mean
flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
idx = {k: torch.from_numpy(np.load(f'{ROOT}/data_w288/{a.dataset}/idx_{k}.npy')).to(dev) for k in ('train', 'val', 'test')}
N, C = data.shape[1], data.shape[2]
xo, yo = torch.arange(-11, 1, device=dev), torch.arange(1, 13, device=dev)
e = pd.read_csv(f'{ROOT}/raw/ASTGNN/data/{a.dataset}/{a.dataset}.csv')
src, dst = e.iloc[:, 0].values, e.iloc[:, 1].values


def masked_mae(pred, y):
    m = (y != 0).float(); m = m / m.mean()
    return ((pred - y).abs() * m).mean()


sched, cl_step = None, None
if a.model == 'mtgnn':
    model = reimpl2.MTGNN(N).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs, epochs, patience, clip, lossf, cl_step = 64, 100, None, 5, 'mae', 100      # curriculum: +1 step / 100 iters

    def run(x):   # (b, 12, N, C) -> (b, 12, N)
        inp = torch.stack([x[..., 0], x[..., C - 2]], 1).transpose(2, 3)          # (b, 2, N, 12): flow + tod
        return model(inp)[..., 0] * std + mean
elif a.model == 'staeformer':
    model = reimpl2.STAEformer(N).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0015)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [25, 45, 65], 0.1)
    bs, epochs, patience, clip, lossf = 16, 300, 30, None, 'mae'

    def run(x):
        inp = torch.stack([x[..., 0], x[..., C - 2], (x[..., C - 1] * 7).round()], -1)   # flow, tod, dow index
        return model(inp) * std + mean
else:
    A = np.zeros((N, N)); A[src, dst] = 1                                          # official: A[i, j] = 1 per edge
    model = reimpl2.ASTGCN(N, A).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs, epochs, patience, clip, lossf = 32, 80, None, None, 'mse'

    def run(x):
        return model(x[..., :1].permute(0, 2, 3, 1)) * std + mean                  # (b, N, 1, 12)
epochs = a.epochs or epochs


def batch(ix):
    return data[ix[:, None] + xo], flow[ix[:, None] + yo]


def evaluate(split, nb=None):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for i, ix in enumerate(idx[split].split(64)):
            if nb is not None and i >= nb:
                break
            x, y = batch(ix); P.append(run(x)); Y.append(y)
    return masked_mae(torch.cat(P), torch.cat(Y)).item()


best, best_state, wait, log, t0, it, level = 1e9, None, 0, [], time.time(), 0, 1
for ep in range(epochs):
    model.train(); L = []
    for ix in idx['train'][torch.randperm(len(idx['train']), device=dev)].split(bs):
        x, y = batch(ix)
        out = run(x)
        if cl_step:
            if it % cl_step == 0 and level < 12:
                level += 1 if it else 0
            out, y = out[:, :level], y[:, :level]
        loss = masked_mae(out, y) if lossf == 'mae' else F.mse_loss(out, y)
        opt.zero_grad(); loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); L.append(loss.item()); it += 1
    if sched:
        sched.step()
    v = evaluate('val'); log.append((ep + 1, float(np.mean(L)), v))
    if v < best:
        best, wait, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
    else:
        wait += 1
    print(f'ep {ep + 1} train {np.mean(L):.3f} val {v:.3f} best {best:.3f} {(time.time() - t0) / (ep + 1):.1f}s/ep', flush=True)
    if patience and wait >= patience:
        break
model.load_state_dict(best_state)

# ---------------------------------------------------------------- analysis
test = evaluate('test')
model.uniform = True; test_uniform = evaluate('test'); model.uniform = False
with torch.no_grad():
    if a.model == 'mtgnn':
        Aj = model.adjacency(); Aj = Aj + torch.eye(N, device=dev); M = (Aj / Aj.sum(1, keepdim=True)).cpu().numpy()
        nnz = (M > 0).sum(1)
        H = float(np.mean([-(r[r > 0] * np.log(r[r > 0])).sum() / math.log(k) for r, k in zip(M, nnz) if k > 1]))
        support = f'its {int(nnz.mean())} kept links'
    else:
        model.record = []; evaluate('test', nb=20); maps = model.record; model.record = None
        # STAEformer maps: (heads*B, T, N, N) rows over keys; ASTGCN maps: (B, N, N) softmax over dim 1 (columns)
        if a.model == 'astgcn':
            maps = [m.transpose(1, 2) for m in maps]
        ent = [(-(m * m.clamp_min(1e-12).log()).sum(-1) / math.log(N)).mean().item() for m in maps]
        H = float(np.mean(ent))
        M = torch.stack([m.reshape(-1, N, N).mean(0) for m in maps]).mean(0).cpu().numpy()
        support = 'all sensors'
adj = np.zeros((N, N), bool); adj[src, dst] = adj[dst, src] = True
off = M[~np.eye(N, dtype=bool)].mean()
nb = np.r_[M[src, dst], M[dst, src]].mean() / off
top = np.argsort(-M, 1)[:, :6]
hit = np.mean([adj[i, [j for j in top[i] if j != i][:5]].mean() for i in range(N)])
chance = adj.sum() / N / (N - 1)
res = dict(model=a.model, dataset=a.dataset, seed=a.seed, test_mae=test, test_mae_mixing_to_mean=test_uniform,
           best_val=best, epochs=len(log), mixing_entropy=H, entropy_support=support,
           road_neighbour_weight_vs_random=float(nb), top5_road_share=float(hit), top5_road_chance=float(chance),
           params=sum(t.numel() for t in model.parameters()), log=log)
os.makedirs(f'{HERE}/results', exist_ok=True)
json.dump(res, open(f'{HERE}/results/{a.model}_{a.dataset}_s{a.seed}.json', 'w'))
torch.save(best_state, f'{HERE}/results/{a.model}_{a.dataset}_s{a.seed}.pt')
print(f'RESULT {a.model} {a.dataset} test {test:.3f} | mixing->mean {test_uniform:.3f} ({test_uniform - test:+.3f}) | '
      f'entropy {H:.3f} over {support} | road-neighbour weight {nb:.2f}x random | top-5 road {100 * hit:.1f}% (chance {100 * chance:.1f}%)')

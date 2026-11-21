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
import sys
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
p.add_argument('--sensor_emb', type=int, default=0, help='identity test: add a learned sensor embedding (gwnet, astgcn)')
p.add_argument('--mixing', default='learned', choices=['learned', 'mean', 'self', 'random', 'mean_sim'],
               help='train with the spatial mixing learned (default), fixed to the sensor mean, or off (self only)')
p.add_argument('--identity', default='learned', choices=['learned', 'profile', 'none', 'pca', 'scpt', 'rel', 'enc'],
               help='staeformer only: sensor identity = learned adaptive embedding, computed profile, or none')
p.add_argument('--holdout', type=float, default=0.0, help='staeformer only: fraction of sensors kept out of training')
# identity-transfer test (2026-11-13), staeformer with a computed identity only; see reimpl2.STAEformer
p.add_argument('--pca_k', type=int, default=8, help='--identity pca: number of principal components')
p.add_argument('--id_dim', type=int, default=0, help='bottleneck width of the identity projection (0 = none)')
p.add_argument('--id_noise', type=float, default=0.0, help='Gaussian noise std on identity features while training')
p.add_argument('--id_mix', type=float, default=0.0, help='blend with a random training sensor, weight ~ U(0, id_mix)')
p.add_argument('--id_drop', type=float, default=0.0, help='zero the identity of this fraction of (sample, sensor) pairs')
p.add_argument('--id_mode', default='concat', choices=['concat', 'affine'], help='affine: identity only scales/shifts the output')
p.add_argument('--enc_episodes', type=int, default=1, help='--identity enc: 0 = always the full history while training (ablation)')
p.add_argument('--id_resample', type=int, default=0, help='identity of each sensor recomputed from k random days per step (pca / rel)')
p.add_argument('--epochs', type=int, default=None, help='override the recipe (smoke tests)')
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
    hold = None
    if a.holdout > 0:                                   # same sensor split as the slim-MAGE runs (train.py)
        rng = np.random.default_rng(123)
        hold = np.zeros(N, bool); hold[rng.choice(N, int(round(a.holdout * N)), replace=False)] = True
    prof = None
    if a.identity == 'profile':
        sys.path.insert(0, ROOT); import streams
        prof = streams.sensor_profile(z['data'], int(idx['train'][-1].item()) + 1, None if hold is None else ~hold)
    dayf = None
    if a.identity in ('pca', 'scpt'):                   # baselines from published inductive methods (id_features.py)
        sys.path.insert(0, ROOT); import id_features
        prof = id_features.build(a.identity, z['data'], int(idx['train'][-1].item()) + 1, None if hold is None else ~hold, dev=dev, seed=a.seed, k=a.pca_k)
    if a.identity in ('rel', 'enc') or a.id_resample:            # per-day features (2026-11-16): network-relative identity, resampling
        sys.path.insert(0, ROOT); import id_features
        assert a.identity in ('pca', 'rel', 'enc'), '--id_resample needs --identity pca or rel'
        dayf = id_features.day_features('rel' if a.identity == 'enc' else a.identity, z['data'], int(idx['train'][-1].item()) + 1,
                                        np.ones(N, bool) if hold is None else ~hold, mean, std, k=a.pca_k,
                                        legacy_stats=(a.identity == 'pca'))
        if a.identity == 'rel':
            prof = id_features.identity_from_days(dayf, np.arange(dayf['D']))
    day_dow = None
    if a.identity == 'enc':
        day_dow = torch.from_numpy(np.rint(z['data'][dayf['s0']:dayf['s0'] + dayf['D'] * 288:288, 0, -1] * 7) % 7)
    model = reimpl2.STAEformer(N, identity='profile' if a.identity in ('pca', 'scpt', 'rel') else a.identity, profile=prof,
                              id_dim=a.id_dim, id_noise=a.id_noise, id_mix=a.id_mix, id_drop=a.id_drop, id_mode=a.id_mode,
                              day_feats=dayf, id_resample=a.id_resample, day_dow=day_dow).to(dev)
    model.enc_episodes = bool(a.enc_episodes)
    if hold is not None:
        model.mean_mask = torch.from_numpy(~hold).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0015)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [25, 45, 65], 0.1)
    bs, epochs, patience, clip, lossf = 16, 300, 30, None, 'mae'

    def run(x):
        inp = torch.stack([x[..., 0], x[..., C - 2], (x[..., C - 1] * 7).round()], -1)   # flow, tod, dow index
        return model(inp) * std + mean
else:
    A = np.zeros((N, N)); A[src, dst] = 1                                          # official: A[i, j] = 1 per edge
    model = reimpl2.ASTGCN(N, A, sensor_emb=8 if a.sensor_emb else 0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs, epochs, patience, clip, lossf = 32, 80, None, None, 'mse'

    def run(x):
        return model(x[..., :1].permute(0, 2, 3, 1)) * std + mean                  # (b, N, 1, 12)
epochs = a.epochs or epochs
if a.model != 'staeformer':
    hold = None
cols = slice(None) if hold is None else torch.from_numpy(np.where(~hold)[0]).to(dev)   # sensors used for training


model.fixed = None if a.mixing == 'learned' else a.mixing
if a.mixing == 'mean_sim':
    # similar sensors (2026-11-16): for every sensor, the 10 TRAINING sensors (never itself, never a held-out one) whose
    # 5-min flow is most Pearson-correlated with it over the last 7 days of the training span; a held-out (new) sensor
    # gets its neighbours from the same 7 days, which is all the history a new sensor is assumed to have
    w7 = flow[int(idx['train'][-1].item()) + 1 - 7 * 288:int(idx['train'][-1].item()) + 1].T     # (N, 2016)
    w7 = (w7 - w7.mean(1, keepdim=True)) / (w7.std(1, keepdim=True) + 1e-6)
    Cc = (w7 @ w7.T) / w7.shape[1]
    Cc.fill_diagonal_(-2.0)
    if hold is not None:
        Cc[:, torch.from_numpy(hold).to(dev)] = -2.0
    nb = Cc.topk(10, dim=1).indices
    sim = torch.zeros(N, N, device=dev); sim.scatter_(1, nb, 0.1)
    model.register_buffer('sim_mix', sim)
if a.mixing == 'random':
    from reimpl import make_random_mix
    model.register_buffer('random_mix', make_random_mix(N, seed=1000 + a.seed).to(dev))
TAG = ('' if a.mixing == 'learned' else '_' + a.mixing) + ('_emb' if a.sensor_emb else '') \
      + ('' if a.identity == 'learned' else '_id' + a.identity) + (f'_hold' if hold is not None else '')   # result-file suffix
IDX = ''.join(f'_{n}{v:g}' for n, v, d in [('k', a.pca_k, 8), ('dim', a.id_dim, 0), ('noise', a.id_noise, 0),
                                            ('mix', a.id_mix, 0), ('drop', a.id_drop, 0), ('rs', a.id_resample, 0), ('ep', a.enc_episodes, 1)] if v != d) \
      + ('_affine' if a.id_mode == 'affine' else '')
TAG = TAG + IDX                                          # identity-transfer variants (2026-11-13)


def batch(ix):
    return data[ix[:, None] + xo], flow[ix[:, None] + yo]


def evaluate(split, nb=None, sel=None):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for i, ix in enumerate(idx[split].split(64)):
            if nb is not None and i >= nb:
                break
            x, y = batch(ix); P.append(run(x)); Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y)
    if sel is not None:
        P, Y = P[..., sel], Y[..., sel]
    return masked_mae(P, Y).item()


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
        out, y = out[..., cols], y[..., cols]
        loss = masked_mae(out, y) if lossf == 'mae' else F.mse_loss(out, y)
        opt.zero_grad(); loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); L.append(loss.item()); it += 1
    if sched:
        sched.step()
    v = evaluate('val', sel=cols); log.append((ep + 1, float(np.mean(L)), v))
    if v < best:
        best, wait, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
    else:
        wait += 1
    print(f'ep {ep + 1} train {np.mean(L):.3f} val {v:.3f} best {best:.3f} {(time.time() - t0) / (ep + 1):.1f}s/ep', flush=True)
    if patience and wait >= patience:
        break
model.load_state_dict(best_state)
os.makedirs(f'{HERE}/results', exist_ok=True)
torch.save(best_state, f'{HERE}/results/{a.model}_{a.dataset}_s{a.seed}{TAG}.pt')   # save before the analysis

# ---------------------------------------------------------------- analysis
test = evaluate('test')
unseen = None
if hold is not None:                                    # accuracy on sensors never used in training
    unseen = dict(unseen_mae=evaluate('test', sel=torch.from_numpy(np.where(hold)[0]).to(dev)), seen_mae=evaluate('test', sel=cols))
    print(f"unseen sensors MAE {unseen['unseen_mae']:.3f} | seen sensors MAE {unseen['seen_mae']:.3f}", flush=True)
model.uniform = True; test_uniform = evaluate('test'); model.uniform = False
with torch.no_grad():
    if a.model == 'mtgnn':
        Aj = model.adjacency(); Aj = Aj + torch.eye(N, device=dev); M = (Aj / Aj.sum(1, keepdim=True)).cpu().numpy()
        nnz = (M > 0).sum(1)
        H = float(np.mean([-(r[r > 0] * np.log(r[r > 0])).sum() / math.log(k) for r, k in zip(M, nnz) if k > 1]))
        support = f'its {int(nnz.mean())} kept links'
    else:
        model.record = []; evaluate('test', nb=20); rec = model.record; model.record = None
        H = float(np.mean([h for h, _ in rec]))                     # rows are over keys for both models
        M = torch.stack([m for _, m in rec]).mean(0).numpy()
        support = 'all sensors'
adj = np.zeros((N, N), bool); adj[src, dst] = adj[dst, src] = True
off = M[~np.eye(N, dtype=bool)].mean()
nb = np.r_[M[src, dst], M[dst, src]].mean() / off
top = np.argsort(-M, 1)[:, :6]
hit = np.mean([adj[i, [j for j in top[i] if j != i][:5]].mean() for i in range(N)])
chance = adj.sum() / N / (N - 1)
res = dict(id_resample=a.id_resample, pca_k=a.pca_k, id_dim=a.id_dim, id_noise=a.id_noise, id_mix=a.id_mix, id_drop=a.id_drop, id_mode=a.id_mode, identity=a.identity, holdout=a.holdout, unseen=unseen, mixing=a.mixing, model=a.model, dataset=a.dataset, seed=a.seed, test_mae=test, test_mae_mixing_to_mean=test_uniform,
           best_val=best, epochs=len(log), mixing_entropy=H, entropy_support=support,
           road_neighbour_weight_vs_random=float(nb), top5_road_share=float(hit), top5_road_chance=float(chance),
           params=sum(t.numel() for t in model.parameters()), log=log)
json.dump(res, open(f'{HERE}/results/{a.model}_{a.dataset}_s{a.seed}{TAG}.json', 'w'))
print(f'RESULT {a.model} {a.dataset} test {test:.3f} | mixing->mean {test_uniform:.3f} ({test_uniform - test:+.3f}) | '
      f'entropy {H:.3f} over {support} | road-neighbour weight {nb:.2f}x random | top-5 road {100 * hit:.1f}% (chance {100 * chance:.1f}%)')

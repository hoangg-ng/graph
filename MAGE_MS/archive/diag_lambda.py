"""Step 8 diagnostic: does the power schedule actually move the spatial operator?

For trained step-8 checkpoints (PEMS08, T = 96), on 20 test batches, per block and scale:
  pos  = ||P^k X||, neg = ||lambda S- X||           (share of the positive / power term)
  dpow = ||P^k X - P X|| / ||S_m X||                (how much the power changes the operator output)
Writes results/diag_lambda.json and prints a summary."""
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

import models
import streams

ROOT = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
z = np.load(f'{ROOT}/data_w288/PEMS08/his.npz')
data = torch.from_numpy(z['data']).to(dev)
te = torch.from_numpy(np.load(f'{ROOT}/data_w288/PEMS08/idx_test.npy')).to(dev)
N, C = data.shape[1], data.shape[2]
xo = torch.arange(-95, 1, device=dev)
models.use_linear_contraction(True)
out = {}
for arm in ('shared', 'proposed', 'reversed'):
    for seed in range(5):
        ck = torch.load(f'{ROOT}/results/step8/ckpt/PEMS08_T96_{arm}_s{seed}.pt', map_location=dev)
        margs = SimpleNamespace(model_dim=128, recur_num=16, topk=4, node_num=N, feature_dim=C - 2, second=0,
                                minute=288, hour=0, day=7, week=0, weekday=0, month=0, quarter=0, year=0, bs=64)
        m = streams.MAGEStream(margs, arm=arm, node_num=N, input_dim=C, output_dim=1, seq_len=96, horizon=12).to(dev)
        m.load_state_dict(ck['state']); m.eval(); bias = ck['bias']
        stats = {}  # (block, scale) -> list of (pos, neg, dpow)

        def make_hook(bname):
            def blk(bk, Hs, b, mode, gamma=None):
                attn = bk.attn
                route, _ = attn.selector(bk.attn_norm(Hs[0]), b)
                ops = streams.GraphOps(attn, route)
                for i, H in enumerate(Hs):
                    X = bk.attn_norm(H)
                    Pk = ops.power(X, mode[i]); P1 = ops.lazy(X); Nn = ops.lam * ops.apply(1, X)
                    S = Pk + Nn
                    stats.setdefault((bname, i), []).append(
                        (Pk.norm().item(), Nn.norm().item(), (Pk - P1).norm().item() / S.norm().item()))
                return orig(bk, Hs, b, mode, gamma)
            return blk

        orig = streams.stream_block
        with torch.no_grad():
            for j, ix in enumerate(te.split(64)):
                if j == 20:
                    break
                x = data[ix[:, None] + xo]
                # run block by block with the hook to label the block
                prompt = m.prompt(x)
                hs = [enc(models._pool(x, s)) + prompt for s, enc in zip(m.scales, m.encoders)]
                zs, _ = make_hook(1)(m.backbone1, hs, bias[0], m.mode)
                xs, _ = make_hook(2)(m.backbone2, zs, bias[1], m.mode)
                make_hook(3)(m.backbone3, [h - a for h, a in zip(hs, xs)], bias[2], m.mode)
        lam = [(-blk.attn.cal_lambda()).item() for blk in (m.backbone1, m.backbone2, m.backbone3)]
        out[f'{arm}_s{seed}'] = {'lambda': lam, 'stats': {f'b{b}_m{i}': np.mean(v, 0).tolist() for (b, i), v in stats.items()}}

json.dump(out, open(f'{ROOT}/results/diag_lambda.json', 'w'), indent=1)
print('arm      | block | power-term share pos/(pos+neg), scales x1..x8 | power effect dpow, scales x1..x8 | lambda')
for k, v in out.items():
    for b in (1, 2, 3):
        sh = [v['stats'][f'b{b}_m{i}'][0] / (v['stats'][f'b{b}_m{i}'][0] + v['stats'][f'b{b}_m{i}'][1]) for i in range(4)]
        dp = [v['stats'][f'b{b}_m{i}'][2] for i in range(4)]
        print(f'{k:12s} b{b} | ' + ' '.join(f'{s:.2f}' for s in sh) + ' | ' + ' '.join(f'{d:.2f}' for d in dp) + f' | {v["lambda"][b-1]:.2f}')

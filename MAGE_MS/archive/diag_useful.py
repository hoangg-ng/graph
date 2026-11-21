"""Which of MAGE's parameters do useful work? (2026-11-07)

For a trained MAGE (streams arm 'mage', T = 96 -> 12): (1) parameter count per component, (2) test MAE when
one component is switched off at inference, no retraining. A component whose removal barely changes MAE is not
doing useful work. Usage: python diag_useful.py PEMS04 results/pems04/ckpt/PEMS04_T96_mage_mage_s0.pt
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

import models
import streams

ROOT = os.path.dirname(os.path.abspath(__file__))
ds, ck = sys.argv[1], sys.argv[2]
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
z = np.load(f'{ROOT}/data_w288/{ds}/his.npz')
data = torch.from_numpy(z['data']).to(dev); mean, std = float(z['mean']), float(z['std'])
flow = data[..., 0] * std + mean
flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
te = torch.from_numpy(np.load(f'{ROOT}/data_w288/{ds}/idx_test.npy')).to(dev)
N, C = data.shape[1], data.shape[2]
xo, yo = torch.arange(-95, 1, device=dev), torch.arange(1, 13, device=dev)
margs = SimpleNamespace(model_dim=128, recur_num=16, topk=4, node_num=N, feature_dim=C - 2, second=0, minute=288,
                        hour=0, day=7, week=0, weekday=0, month=0, quarter=0, year=0, bs=64)
models.use_linear_contraction(True)
m = streams.MAGEStream(margs, arm='mage', node_num=N, input_dim=C, output_dim=1, seq_len=96, horizon=12).to(dev)
st = torch.load(ck, map_location=dev); m.load_state_dict(st['state']); bias = st['bias']; m.eval()

# ---------------------------------------------------------------- 1. parameter accounting
groups = {'graph experts (E1, E2)': ('.E1', '.E2'), 'router': ('.router',), 'lambda + outer': ('lambda_', 'outer', 'coefficient'),
          'feed-forward (SwiGLU)': ('.ffn.',), 'norms': ('_norm',), 'sensor embedding': ('spatial_prompt',),
          'time-of-day / day-of-week embeddings': ('prompt.pool',), 'encoder': ('encoders',), 'decoders': ('decoders',)}
count = {g: 0 for g in groups}
for n, p in m.named_parameters():
    g = next(g for g, keys in groups.items() if any(k in n for k in keys))
    count[g] += p.numel()
total = sum(count.values())


# ---------------------------------------------------------------- 2. switch-off ablations
def block(bk, H, b, opt, i):
    X = bk.attn_norm(H)
    if opt == 'no_space':
        S = torch.zeros_like(X)
    elif opt == 'graph_to_mean':   # replace both routed graphs by the exact sensor mean, keep lambda
        Mx = X.mean(1, keepdim=True).expand_as(X)
        S = Mx + bk.attn.cal_lambda() * Mx
    else:
        route, _ = bk.attn.selector(X, b)
        ops = streams.GraphOps(bk.attn, route)
        S = ops.apply(0, X) + ops.lam * ops.apply(1, X)
    U = H + S
    return U if opt == f'no_ffn{i}' else U + bk.ffn(bk.ffn_norm(U))


def forward(x, opt):
    prompt = m.prompt.spatial_prompt * (0 if opt == 'no_sensor_emb' else 1)
    for i, emb in enumerate(m.prompt.pool):
        if opt == ['no_tod', 'no_dow'][i]:
            continue
        prompt = prompt + emb((x[:, -1, :, i + m.prompt.feature_dim] * m.prompt.denorm[i]).long())
    h = m.encoders[0](models._pool(x, 1)) + prompt
    zz = block(m.backbone1, h, bias[0], opt, 1)
    xx = zz if opt == 'skip_block2' else block(m.backbone2, zz, bias[1], opt, 2)
    xx = h - xx if opt == 'skip_block3' else block(m.backbone3, h - xx, bias[2], opt, 3)
    y = (0 if opt == 'no_decoder_final' else m.decoders[0](xx)) + (0 if opt == 'no_decoder_block1' else m.decoders2[0](zz))
    return y.reshape(x.shape[0], N, 12, 1).transpose(1, 2)[..., 0] * std + mean


def mae(opt):
    P, Y = [], []
    with torch.no_grad():
        for ix in te.split(64):
            P.append(forward(data[ix[:, None] + xo], opt)); Y.append(flow[ix[:, None] + yo])
    P, Y = torch.cat(P), torch.cat(Y)
    msk = (Y != 0).float(); msk = msk / msk.mean()
    return ((P - Y).abs() * msk).mean().item()


opts = {'full model': 'none', 'graph experts -> exact sensor mean': 'graph_to_mean', 'no spatial mixing at all': 'no_space',
        'sensor embedding off': 'no_sensor_emb', 'time-of-day embedding off': 'no_tod', 'day-of-week embedding off': 'no_dow',
        'feed-forward off, block 1': 'no_ffn1', 'feed-forward off, block 2': 'no_ffn2', 'feed-forward off, block 3': 'no_ffn3',
        'skip block 2': 'skip_block2', 'skip block 3': 'skip_block3', 'final decoder off': 'no_decoder_final',
        'block-1 decoder off': 'no_decoder_block1'}
base = mae('none')
res = {'dataset': ds, 'ckpt': ck, 'params': count, 'total_params': total, 'ablations': {}}
print(f'{ds}: {total:,} parameters')
for g, c in sorted(count.items(), key=lambda kv: -kv[1]):
    print(f'  {g:40s} {c:>10,}  {100 * c / total:5.1f} %')
print(f'{"switched off":40s} {"test MAE":>9s} {"change":>8s}')
for name, o in opts.items():
    v = base if o == 'none' else mae(o)
    res['ablations'][name] = v
    print(f'  {name:38s} {v:9.3f} {v - base:+8.3f}')
os.makedirs(f'{ROOT}/results/useful', exist_ok=True)
json.dump(res, open(f'{ROOT}/results/useful/{ds}_{os.path.basename(ck)[:-3]}.json', 'w'), indent=1)

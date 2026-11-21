"""Post-hoc analyses from saved checkpoints (research_design.md steps 4 and 5).

Routing: graph-expert usage over the test set per MAGE block (top-4 of 16 experts), captured
with forward hooks; normalised entropy of global usage, mean per-node usage entropy, and
Jensen-Shannon divergence between models 1 and 5 (same seed). Also the learned scale gates of
models 4 and 5.

Regimes (per sample x node, horizon-averaged masked MAE):
  speed-based, threshold 40 mph on the raw speed channel (PEMS04/08 carry speed):
    free flow   all input and all target speeds >= 40
    onset       last input speed >= 40, some target speed < 40
    recovery    last input speed < 40, some target speed >= 40
    congested   all input and all target speeds < 40
  volatility: std of first differences of input flow; low / high = bottom / top quartile,
    thresholds from training windows.
The target window is used only to stratify errors, not as model input.
"""
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

import models

ROOT = os.path.dirname(os.path.abspath(__file__))
dev = torch.device('cuda:0')
ENC = ['linear', 'mslin', 'm3', 'm4', 'm5']
RAW = {'PEMS04': 'raw/ASTGNN/data/PEMS04/PEMS04.npz', 'PEMS08': 'raw/ASTGNN/data/PEMS08/PEMS08.npz'}
models.use_linear_contraction(True)


def load(ds, enc, seed):
    ck = torch.load(f'{ROOT}/results/ckpt/{ds}_T12_{enc}_s{seed}.pt', map_location=dev)
    z = np.load(f'{ROOT}/data/{ds}/his.npz')
    N, C = z['data'].shape[1:]
    ma = SimpleNamespace(model_dim=128, recur_num=16, topk=4, node_num=N, feature_dim=C - 2, second=0,
                         minute=288, hour=0, day=7, week=0, weekday=0, month=0, quarter=0, year=0, bs=64)
    m = models.MAGEX(ma, encoder=enc, node_num=N, input_dim=C, output_dim=1, seq_len=12, horizon=12).to(dev)
    m.load_state_dict(ck['state']); m.eval()
    return m, ck['bias']


def predict(m, bias, ds):
    z = np.load(f'{ROOT}/data/{ds}/his.npz')
    data = torch.from_numpy(z['data']).to(dev); mean, std = float(z['mean']), float(z['std'])
    idx = torch.from_numpy(np.load(f'{ROOT}/data/{ds}/idx_test.npy')).to(dev)
    xo, yo = torch.arange(-11, 1, device=dev), torch.arange(1, 13, device=dev)
    picks = []
    hooks = [blk.attn.register_forward_hook(lambda mod, i, o: picks.append(o[1].cpu()))
             for blk in (m.backbone1, m.backbone2, m.backbone3)]
    P = []
    with torch.no_grad():
        for ix in idx.split(64):
            P.append((m(data[ix[:, None] + xo], bias=bias)[..., 0] * std + mean).cpu())
    for h in hooks:
        h.remove()
    # picks: per batch, 3 tensors (b, n, 4) in block order
    route = [torch.cat(picks[k::3]) for k in range(3)]       # 3 x (S, n, 4)
    return torch.cat(P), route


def usage_stats(route, K=16):
    out = []
    for r in route:
        g = torch.bincount(r.flatten(), minlength=K).float(); g /= g.sum()
        ent = -(g * g.clamp_min(1e-12).log()).sum() / np.log(K)
        S, n, k = r.shape
        per_node = torch.zeros(n, K).scatter_add_(1, r.permute(1, 0, 2).reshape(n, -1),
                                                  torch.ones(n, S * k))
        per_node /= per_node.sum(1, keepdim=True)
        ne = (-(per_node * per_node.clamp_min(1e-12).log()).sum(1) / np.log(K)).mean()
        out.append(dict(usage=g.tolist(), entropy=ent.item(), node_entropy=ne.item(), max_share=g.max().item()))
    return out


def jsd(p, q):
    p, q = np.asarray(p) + 1e-12, np.asarray(q) + 1e-12
    mm = (p + q) / 2
    return 0.5 * (p * np.log(p / mm)).sum() + 0.5 * (q * np.log(q / mm)).sum()


def regimes(ds):
    raw = np.load(f'{ROOT}/{RAW[ds]}')['data']
    flow, speed = raw[..., 0], raw[..., 2]
    te = np.load(f'{ROOT}/data/{ds}/idx_test.npy'); tr = np.load(f'{ROOT}/data/{ds}/idx_train.npy')
    xin, yout = te[:, None] + np.arange(-11, 1), te[:, None] + np.arange(1, 13)
    sin, sout = speed[xin], speed[yout]                   # (S, 12, n)
    lab = {}
    lab['free flow'] = (sin >= 40).all(1) & (sout >= 40).all(1)
    lab['onset'] = (sin[:, -1] >= 40) & (sout < 40).any(1)
    lab['recovery'] = (sin[:, -1] < 40) & (sout >= 40).any(1)
    lab['congested'] = (sin < 40).all(1) & (sout < 40).all(1)
    vol = lambda ix: np.diff(flow[ix[:, None] + np.arange(-11, 1)], axis=1).std(1)
    vtr = vol(tr[::5]); lo, hi = np.quantile(vtr, [0.25, 0.75])
    v = vol(te)
    lab['low volatility'] = v <= lo
    lab['high volatility'] = v >= hi
    y = torch.from_numpy(flow[yout]).float()               # (S, 12, n)
    return lab, y


def masked_mae(p, y, sel):
    m = (y != 0) & torch.from_numpy(sel)[:, None, :]
    return ((p - y).abs()[m]).mean().item()


def main():
    res = {'routing': {}, 'gates': {}, 'regimes': {}, 'regime_counts': {}}
    for ds in ['PEMS08', 'PEMS04']:
        lab, y = regimes(ds)
        res['regime_counts'][ds] = {k: int(v.sum()) for k, v in lab.items()}
        for enc in ENC:
            for seed in range(5):
                m, bias = load(ds, enc, seed)
                P, route = predict(m, bias, ds)
                res['regimes'][f'{ds}|{enc}|{seed}'] = {k: masked_mae(P, y, v) for k, v in lab.items()}
                if enc in ('linear', 'm5'):
                    res['routing'][f'{ds}|{enc}|{seed}'] = usage_stats(route)
                if enc in ('m4', 'm5'):
                    g = m.encoder.gates(m.prompt.spatial_prompt).detach().cpu()
                    ent = (-(g * g.clamp_min(1e-12).log()).sum(1) / np.log(g.shape[1]))
                    res['gates'][f'{ds}|{enc}|{seed}'] = dict(mean=g.mean(0).tolist(), sd_across_nodes=g.std(0).tolist(),
                                                             mean_entropy=ent.mean().item(), min_entropy=ent.min().item())
                print(ds, enc, seed, flush=True)

    # summaries
    lines = []
    for ds in ['PEMS08', 'PEMS04']:
        lines.append(f'\n## {ds}\nregime counts (sample x node): {res["regime_counts"][ds]}\n')
        regs = list(res['regime_counts'][ds])
        lines.append('| model | ' + ' | '.join(regs) + ' |'); lines.append('|---|' + '---|' * len(regs))
        for enc in ENC:
            cells = []
            for rg in regs:
                v = [res['regimes'][f'{ds}|{enc}|{s}'][rg] for s in range(5)]
                b = [res['regimes'][f'{ds}|linear|{s}'][rg] for s in range(5)]
                d = np.array(v) - np.array(b)
                cells.append(f'{np.mean(v):.2f}' + ('' if enc == 'linear' else f' ({d.mean():+.2f}, {(d < 0).sum()}/5)'))
            lines.append(f'| {enc} | ' + ' | '.join(cells) + ' |')
        lines.append('\nrouting (mean over seeds), per block: usage entropy / mean per-node entropy / max expert share')
        for enc in ('linear', 'm5'):
            st = [res['routing'][f'{ds}|{enc}|{s}'] for s in range(5)]
            lines.append(f'{enc}: ' + ' ; '.join(
                f'b{b+1} {np.mean([x[b]["entropy"] for x in st]):.3f} / {np.mean([x[b]["node_entropy"] for x in st]):.3f} / '
                f'{np.mean([x[b]["max_share"] for x in st]):.3f}' for b in range(3)))
        j = [[jsd(res['routing'][f'{ds}|linear|{s}'][b]['usage'], res['routing'][f'{ds}|m5|{s}'][b]['usage']) for b in range(3)] for s in range(5)]
        jl = [[jsd(res['routing'][f'{ds}|linear|{s}'][b]['usage'], res['routing'][f'{ds}|linear|{(s+1)%5}'][b]['usage']) for b in range(3)] for s in range(5)]
        lines.append(f'JSD(global usage) linear vs m5 same seed, per block: {np.mean(j, 0).round(4).tolist()}; '
                     f'linear vs linear other seed: {np.mean(jl, 0).round(4).tolist()}')
        for enc in ('m4', 'm5'):
            gs = [res['gates'][f'{ds}|{enc}|{s}'] for s in range(5)]
            lines.append(f'{enc} gates (scale x1, x2, x4) mean {np.mean([g["mean"] for g in gs], 0).round(3).tolist()}, '
                         f'sd across nodes {np.mean([g["sd_across_nodes"] for g in gs], 0).round(3).tolist()}, '
                         f'normalised entropy mean {np.mean([g["mean_entropy"] for g in gs]):.3f} min {np.mean([g["min_entropy"] for g in gs]):.3f}')
    open(f'{ROOT}/results/post_summary.md', 'w').write('\n'.join(lines))
    json.dump(res, open(f'{ROOT}/results/post.json', 'w'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

"""Train/evaluate MAGE with a chosen temporal encoder on PEMS0X; write one JSON per run.

Protocol follows the MAGE paper (Sec. 5 / LargeST platform) and official main.py:
Adam lr 0.02 wd 1e-4, MultiStepLR(0.5) at the official milestones, grad-clip 5, masked L1 on
raw flow, bs 64, <=300 epochs, patience 30 on val MAE, d=128, dG=32, 16 experts, top-4,
load-balancing bias updated by sign rule with lr 1e-3 (paper Eq. on beta_k; the line is
commented out in the release, we enable it as the paper describes).
The official engine is not usable as released (MAGE_enine.evaluate saves to /data/mjm1/... and
calls sys.exit(0)), so this is a re-implementation of the same loop with data kept on GPU.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

import models
import streams

ROOT = os.path.dirname(os.path.abspath(__file__))


def masked_metrics(pred, y):
    m = (y != 0).float()
    m = m / m.mean()
    err = (pred - y).abs() * m
    mae = err.mean()
    rmse = (((pred - y) ** 2) * m).mean().sqrt()
    mape = torch.where(y != 0, err / y.abs().clamp_min(1e-6), torch.zeros_like(err)).mean() * 100
    return mae, rmse, mape


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='PEMS08')
    p.add_argument('--encoder', default='linear')
    p.add_argument('--arm', default=None, help='step 8: build streams.MAGEStream with this arm')
    p.add_argument('--lam', default='mage', choices=['mage', 'sigmoid'],
                   help="step 8 only: MAGE's unconstrained lambda, or a sigmoid-bounded lambda per expert")
    p.add_argument('--seq_len', type=int, default=12)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lb', type=int, default=1)
    p.add_argument('--data_dir', default='data')
    p.add_argument('--out', required=True)
    p.add_argument('--graph_fix', type=int, default=0, help='step 8 only: spread-out E1/E2 init and no weight decay on them')
    p.add_argument('--hier_rank', type=int, default=0, help='step 8 only: rank of the time and sensor levels (0 = off)')
    p.add_argument('--switch_at', type=int, default=0,
                   help='method B (arm mage): after this epoch, if every block\'s graph entropy > 0.99, replace the '
                        'graph experts by the slim mean term (keep all other weights) and continue')
    p.add_argument('--identity', default='learned', choices=['learned', 'profile', 'none'],
                   help='step 8 only: sensor identity = learned embedding, computed profile, or none')
    p.add_argument('--holdout', type=float, default=0.0,
                   help='unseen-sensor test: fraction of sensors excluded from training (loss and network average)')
    p.add_argument('--log_graph', type=int, default=0, help='step 8 only: print graph entropy after every epoch')
    p.add_argument('--tf32', type=int, default=0, help='1: TF32 matmuls (Ampere+; faster, ~1e-3 relative precision)')
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32); torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    dev = torch.device(a.device)
    z = np.load(f'{ROOT}/{a.data_dir}/{a.dataset}/his.npz')
    data = torch.from_numpy(z['data']).to(dev)              # (T, N, C) normalised values + tod + dow
    mean, std = float(z['mean']), float(z['std'])
    flow = data[..., 0] * std + mean                         # raw flow for labels
    flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
    idx = {k: torch.from_numpy(np.load(f'{ROOT}/{a.data_dir}/{a.dataset}/idx_{k}.npy')).to(dev) for k in ['train', 'val', 'test']}
    N, C = data.shape[1], data.shape[2]
    xo = torch.arange(-(a.seq_len - 1), 1, device=dev)
    yo = torch.arange(1, a.horizon + 1, device=dev)

    def batch(ix):
        return data[ix[:, None] + xo], flow[ix[:, None] + yo]  # (b,T,N,C), (b,H,N)

    margs = SimpleNamespace(model_dim=128, recur_num=16, topk=4, node_num=N, feature_dim=C - 2,
                            second=0, minute=288, hour=0, day=7, week=0, weekday=0, month=0,
                            quarter=0, year=0, bs=a.bs)
    models.use_linear_contraction(True)
    G = None
    if a.arm in ('lite_cof', 'lite_road'):
        import pandas as pd
        e = pd.read_csv(f'{ROOT}/raw/ASTGNN/data/{a.dataset}/{a.dataset}.csv')
        G = streams.fixed_graph('cof' if a.arm == 'lite_cof' else 'road', z['data'], int(idx['train'][-1]) + 1,
                                (e.iloc[:, 0].values, e.iloc[:, 1].values))
    hold = None
    if a.holdout > 0:
        rng = np.random.default_rng(123)
        hold = np.zeros(N, bool); hold[rng.choice(N, int(round(a.holdout * N)), replace=False)] = True
    prof = streams.sensor_profile(z['data'], int(idx['train'][-1]) + 1, None if hold is None else ~hold) if a.identity == 'profile' else None
    if a.arm:
        model = streams.MAGEStream(margs, arm=a.arm, lam=a.lam, graph_fix=a.graph_fix, hier_rank=a.hier_rank, fixed_graph=G,
                                   identity=a.identity, profile=prof, node_num=N, input_dim=C, output_dim=1,
                                   seq_len=a.seq_len, horizon=a.horizon).to(dev)
        n_enc = model.encoder_params()
    else:
        model = models.MAGEX(margs, encoder=a.encoder, node_num=N, input_dim=C, output_dim=1,
                             seq_len=a.seq_len, horizon=a.horizon).to(dev)
        n_enc = models.n_params(model.encoder)
    train_cols = slice(None)
    if hold is not None:
        model.mean_mask = torch.from_numpy(~hold).to(dev)
        train_cols = torch.from_numpy(np.where(~hold)[0]).to(dev)
    n_total = models.n_params(model)
    print(f'{a.dataset} {a.arm or a.encoder} T={a.seq_len} seed={a.seed} params={n_total} enc={n_enc}', flush=True)

    if a.arm and a.graph_fix:
        graph = [p_ for n_, p_ in model.named_parameters() if n_.endswith('.E1') or n_.endswith('.E2')]
        ids = {id(p_) for p_ in graph}
        opt = torch.optim.Adam([{'params': [p_ for p_ in model.parameters() if id(p_) not in ids]},
                                {'params': graph, 'weight_decay': 0.0}], lr=a.lr, weight_decay=a.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 30, 50, 70, 90, 130, 170, 210, 250, 290, 300], gamma=0.5)
    bias = torch.zeros(3, margs.recur_num, device=dev)

    def predict(split):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for ix in idx[split].split(a.bs):
                x, y = batch(ix)
                P.append(model(x, bias=bias)[..., 0] * std + mean); Y.append(y)
        return torch.cat(P), torch.cat(Y)

    best, best_state, wait, ep_times, log = float('inf'), None, 0, [], []
    torch.cuda.reset_peak_memory_stats(dev)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); losses = []
        perm = idx['train'][torch.randperm(len(idx['train']), device=dev)]
        for ix in perm.split(a.bs):
            x, y = batch(ix)
            out, counts = model(x, bias=bias)
            loss = masked_metrics((out[..., 0] * std + mean)[..., train_cols], y[..., train_cols])[0]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            if a.lb:
                bias -= 1e-3 * torch.sign(counts.float() - counts.float().mean(-1, keepdim=True))
            losses.append(loss.item())
        torch.cuda.synchronize(dev); ep_times.append(time.time() - t0)
        sched.step()
        Pv, Yv = predict('val')
        vmae = masked_metrics(Pv[..., train_cols], Yv[..., train_cols])[0].item()   # held-out sensors never used for selection
        log.append((ep + 1, float(np.mean(losses)), vmae))
        if a.arm == 'mage' and a.switch_at and ep + 1 == a.switch_at and not hasattr(model, 'switched'):
            ent = streams.graph_entropy(model)
            print(f'switch check ep {ep + 1}: graph entropy ' + ' '.join(f'{v:.4f}' for v in ent), flush=True)
            if min(ent) > 0.99:                       # collapsed: swap experts for c_b * mean, keep everything else
                lite = streams.MAGEStream(margs, arm='lite', node_num=N, input_dim=C, output_dim=1,
                                          seq_len=a.seq_len, horizon=a.horizon).to(dev)
                own = model.state_dict(); new = lite.state_dict()
                lite.load_state_dict({k: (own[k] if k in own and own[k].shape == v.shape else v) for k, v in new.items()})
                with torch.no_grad():                 # collapsed MAGE computes (1 - lambda_b) * mean per block
                    coef = torch.tensor([1 + bk.attn.cal_lambda().item() for bk in (model.backbone1, model.backbone2, model.backbone3)])
                    lite.c_logit.copy_(torch.atanh(coef.clamp(-0.95, 0.95)))
                lr_now = opt.param_groups[0]['lr']
                model = lite; model.switched = ep + 1
                opt = torch.optim.Adam(model.parameters(), lr=lr_now, weight_decay=a.wd)
                sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m - ep - 1 for m in [1, 30, 50, 70, 90, 130, 170, 210, 250, 290, 300] if m > ep + 1], gamma=0.5)
                best, best_state, wait = float('inf'), None, 0
                n_total = models.n_params(model)
                print(f'switched to slim model at epoch {ep + 1}: {n_total} params', flush=True)
                continue
        if a.arm and a.log_graph:   # per-epoch graph uniformity (1.0 = every expert is the sensor mean)
            print(f'graph_entropy ep {ep + 1} ' + ' '.join(f'{v:.4f}' for v in streams.graph_entropy(model)), flush=True)
        if vmae < best:
            best, wait = vmae, 0
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}, bias.clone())
        else:
            wait += 1
        if ep % 10 == 0 or wait == 0:
            print(f'ep {ep+1} train {np.mean(losses):.3f} val {vmae:.3f} best {best:.3f} {ep_times[-1]:.1f}s', flush=True)
        if wait >= a.patience:
            break
    model.load_state_dict(best_state[0]); bias = best_state[1]
    P, Y = predict('test')
    per_h = [[v.item() for v in masked_metrics(P[:, h], Y[:, h])] for h in range(a.horizon)]
    avg = [v.item() for v in masked_metrics(P, Y)]
    unseen = None
    if hold is not None:                       # accuracy on the sensors never used in training, and on the others
        hc = torch.from_numpy(np.where(hold)[0]).to(dev)
        unseen = dict(unseen_mae=masked_metrics(P[..., hc], Y[..., hc])[0].item(),
                      seen_mae=masked_metrics(P[..., train_cols], Y[..., train_cols])[0].item())
        print(f'unseen sensors MAE {unseen["unseen_mae"]:.3f} | seen sensors MAE {unseen["seen_mae"]:.3f}', flush=True)
    res = dict(vars(a), params=n_total, enc_params=n_enc, epochs_run=len(log), best_val_mae=best,
               test_mae=avg[0], test_rmse=avg[1], test_mape=avg[2], per_horizon=per_h,
               train_s_per_epoch=float(np.mean(ep_times)),
               peak_mem_mb=torch.cuda.max_memory_allocated(dev) / 2**20, log=log,
               graph_entropy=streams.graph_entropy(model) if a.arm else None,
               switched_at=getattr(model, 'switched', None), unseen=unseen)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # best weights + balancing bias, for the routing / regime analyses (re-run predictions later)
    ck = os.path.join(os.path.dirname(os.path.abspath(a.out)), 'ckpt', os.path.basename(a.out).replace('.json', '.pt'))
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({'state': best_state[0], 'bias': best_state[1], 'args': vars(a)}, ck)
    json.dump(res, open(a.out, 'w'))
    print(f'TEST MAE {avg[0]:.3f} RMSE {avg[1]:.3f} MAPE {avg[2]:.2f} | epochs {len(log)}', flush=True)


if __name__ == '__main__':
    main()

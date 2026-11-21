"""New-sensor evaluation (2026-11-16) of a trained STAEformer (network mean or learned attention) with a computed
identity (--identity pca | rel) or the standard learned embedding (identity learned: `computed` = the trained, decayed
rows; fitting adjusts those rows), 20 % unseen sensors, rng 123. Nothing in the trained model changes; only the unseen sensors'
identity is rebuilt as if each new sensor arrived with the LAST k days of the training span as its only history:
  computed      identity from those k days (pca: per-day PCA projections; rel: log(sensor / network mean) per hour)
  sft_computed  start from `computed`, then fit ONLY the unseen sensors' identity features by gradient descent on
                forecasting error over those k days (model frozen; chronological 80/20 split of the k days' forecast
                origins, early stopping on the last 20 %)
  sft_zero      the same fit started from the average sensor (identity 0 in standardised units), no computed prior
k = 1, 3, 7 days, and 0 = the full training span (computed only; = the training-time setting).
Usage: python newsensor_eval.py --tag staeformer_PEMS08_s0_mean_idpca_hold_noise0.5
Writes results/newsensor/<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import reimpl2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import id_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--device', default='cuda:0')
p.add_argument('--days', default='1,3,7,0')
p.add_argument('--steps', type=int, default=400)
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device(a.device)
R = json.load(open(f'{HERE}/results/{a.tag}.json'))
assert R['identity'] in ('pca', 'rel', 'learned', 'enc') and R['holdout'] > 0, 'needs a holdout run'
LEARNED = R['identity'] == 'learned'      # 2026-11-16: fit the learned embedding rows of the unseen sensors instead
ds = R['dataset']; torch.manual_seed(R['seed'])

z = np.load(f'{ROOT}/data_w288/{ds}/his.npz')
data = torch.from_numpy(z['data']).to(dev); mean, std = float(z['mean']), float(z['std'])
flow = data[..., 0] * std + mean
flow = torch.where(flow.abs() < 1e-3, torch.zeros_like(flow), flow)
idx = {k: torch.from_numpy(np.load(f'{ROOT}/data_w288/{ds}/idx_{k}.npy')).to(dev) for k in ('train', 'test')}
N, C = data.shape[1], data.shape[2]
train_end = int(idx['train'][-1].item()) + 1
rng = np.random.default_rng(123)
hold = np.zeros(N, bool); hold[rng.choice(N, int(round(R['holdout'] * N)), replace=False)] = True
H = torch.from_numpy(np.where(hold)[0]).to(dev); S = torch.from_numpy(np.where(~hold)[0]).to(dev)
xo, yo = torch.arange(-11, 1, device=dev), torch.arange(1, 13, device=dev)

st = torch.load(f'{HERE}/results/{a.tag}.pt', map_location=dev)
if R['identity'] == 'enc':
    # trained history encoder (2026-11-16): a new sensor's identity = the encoder on its last k days; nothing is fitted
    dayf = id_features.day_features('rel', z['data'], train_end, ~hold, mean, std)
    day_dow = torch.from_numpy(np.rint(z['data'][dayf['s0']:dayf['s0'] + dayf['D'] * 288:288, 0, -1] * 7) % 7)
    model = reimpl2.STAEformer(N, identity='enc', day_feats=dayf, day_dow=day_dow).to(dev)
    model.load_state_dict(st); model.fixed = None if R['mixing'] == 'learned' else R['mixing']; model.eval()

    def test_enc():
        P, Y = [], []
        with torch.no_grad():
            for ix in idx['test'].split(64):
                x = data[ix[:, None] + xo]
                inp = torch.stack([x[..., 0], x[..., C - 2], (x[..., C - 1] * 7).round()], -1)
                P.append((model(inp) * std + mean)[..., H]); Y.append(flow[ix[:, None] + yo][..., H])
        P, Y = torch.cat(P), torch.cat(Y)
        m = (Y != 0).float(); m = m / m.mean()
        return ((P - Y).abs() * m).mean().item()

    out = dict(tag=a.tag, identity='enc', dataset=ds, trained_unseen_mae=R['unseen']['unseen_mae'],
               seen_mae=R['unseen']['seen_mae'], days={})
    D = dayf['D']
    for k in [0, 1, 2, 3, 5, 7, D]:
        m = torch.ones(D, N, dtype=torch.bool, device=dev)
        m[:, H] = False
        if k:
            m[D - k:, H] = True
        model.ev_mask = m
        name = 'full' if k == D else str(k)
        out['days'][name] = dict(computed=test_enc())
        print(f'k={name} days: computed {out["days"][name]["computed"]:.3f}', flush=True)
    os.makedirs(f'{HERE}/results/newsensor', exist_ok=True)
    json.dump(out, open(f'{HERE}/results/newsensor/{a.tag}.json', 'w'), indent=1)
    print('DONE', flush=True)
    sys.exit(0)
if LEARNED:
    model = reimpl2.STAEformer(N).to(dev)
else:
    dayf = id_features.day_features(R['identity'], z['data'], train_end, ~hold, mean, std, k=R.get('pca_k', 8),
                                    legacy_stats=(R['identity'] == 'pca'))
    model = reimpl2.STAEformer(N, identity='profile', profile=st['profile'], id_dim=R.get('id_dim', 0),
                               id_mode=R.get('id_mode', 'concat')).to(dev)
model.load_state_dict(st); model.fixed = None if R['mixing'] == 'learned' else R['mixing']; model.eval()
for q in model.parameters():
    q.requires_grad_(False)
if LEARNED:                                     # identity = rows of the (T, N, 80) adaptive embedding
    stored = st['adaptive_embedding'].clone()
    del model.adaptive_embedding
    model.adaptive_embedding = stored.clone()
    dayf = dict(s0=int(np.argmax(np.rint(z['data'][:train_end, 0, -2] * 288).astype(int) % 288 == 0)))
    dayf['D'] = (train_end - dayf['s0']) // 288
else:
    stored = st['profile'].clone()
    full = id_features.identity_from_days(dayf, np.arange(dayf['D'])).to(dev)
    print(f'feature check: stored vs recomputed full-span identity, max |diff| {float((stored - full).abs().max()):.2e}', flush=True)


def run(x):
    inp = torch.stack([x[..., 0], x[..., C - 2], (x[..., C - 1] * 7).round()], -1)
    return model(inp) * std + mean


def mae(P, Y):
    m = (Y != 0).float(); m = m / m.mean()
    return ((P - Y).abs() * m).mean()


def set_unseen(f):                      # f: unseen sensors' identity, (len(H), F) or learned (T, len(H), 80)
    prof = stored.clone()
    if LEARNED:
        prof[:, H] = f; model.adaptive_embedding = prof
    else:
        prof[H] = f; model.profile = prof


def unseen_rows():                      # the trained model's own identity for the unseen sensors
    return stored[:, H].clone() if LEARNED else stored[H].clone()


def test_mae():
    P, Y = [], []
    with torch.no_grad():
        for ix in idx['test'].split(64):
            P.append(run(data[ix[:, None] + xo])[..., H]); Y.append(flow[ix[:, None] + yo][..., H])
    return mae(torch.cat(P), torch.cat(Y)).item()


def fit(init, k):
    """Fit only the unseen sensors' identity on forecasts whose input and target lie in the last k days."""
    start = dayf['s0'] + (dayf['D'] - k) * 288; end = dayf['s0'] + dayf['D'] * 288
    org = torch.arange(start + 11, end - 12, device=dev); n_fit = int(0.8 * len(org))
    fit_o, stop_o = org[:n_fit], org[n_fit:]
    delta = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=0.05)

    def loss_on(o):
        set_unseen(delta)
        return mae(run(data[o[:, None] + xo])[..., H], flow[o[:, None] + yo][..., H])

    def stop_loss():                    # early-stopping loss in chunks of 64 origins (memory; 2026-11-16 OOM fix)
        set_unseen(delta)
        P = torch.cat([run(data[o[:, None] + xo])[..., H] for o in stop_o.split(64)])
        return float(mae(P, flow[stop_o[:, None] + yo][..., H]))

    with torch.no_grad():
        best, best_d = stop_loss(), init.clone()
    wait, used = 0, 0
    for it in range(a.steps):
        b = fit_o[torch.randint(len(fit_o), (32,), device=dev)]
        loss = loss_on(b); opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 10 == 0:
            with torch.no_grad():
                v = stop_loss()
            if v < best - 1e-4:
                best, best_d, wait, used = v, delta.detach().clone(), 0, it + 1
            else:
                wait += 1
                if wait >= 5:
                    break
    return best_d, used


out = dict(tag=a.tag, identity=R['identity'], dataset=ds, trained_unseen_mae=R['unseen']['unseen_mae'],
           seen_mae=R['unseen']['seen_mae'], no_history=None, days={})
set_unseen(torch.zeros_like(unseen_rows())); out['no_history'] = test_mae()   # identity 0, 0 days
if LEARNED:
    set_unseen(unseen_rows()); out['trained_rows'] = test_mae()                # what training left (decayed)
    print(f'trained (decayed) embedding rows: unseen MAE {out["trained_rows"]:.3f}', flush=True)
print(f'no history (average sensor): unseen MAE {out["no_history"]:.3f}', flush=True)
for k in [int(s) for s in a.days.split(',')]:
    days = np.arange(dayf['D']) if k == 0 else np.arange(dayf['D'] - k, dayf['D'])
    if LEARNED and k == 0:
        continue
    comp = unseen_rows() if LEARNED else id_features.identity_from_days(dayf, days).to(dev)[H]
    set_unseen(comp); r = dict(computed=test_mae())
    if k > 0:
        d, used = fit(comp, k); set_unseen(d); r['sft_computed'] = test_mae(); r['sft_computed_steps'] = used
        d, used = fit(torch.zeros_like(comp), k); set_unseen(d); r['sft_zero'] = test_mae(); r['sft_zero_steps'] = used
    out['days'][str(k)] = r
    print(f'k={k} days: ' + ', '.join(f'{n} {v:.3f}' if isinstance(v, float) else f'{n} {v}' for n, v in r.items()), flush=True)
os.makedirs(f'{HERE}/results/newsensor', exist_ok=True)
json.dump(out, open(f'{HERE}/results/newsensor/{a.tag}.json', 'w'), indent=1)
print('DONE', flush=True)

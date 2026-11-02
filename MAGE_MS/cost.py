"""Cost vs number of nodes N: one training step (fwd+bwd+Adam) of full MAGE with each encoder.

Synthetic inputs (cost does not depend on values). Reports ms/step, peak memory, and params,
plus the official einsum contraction (torch's default left-to-right path, no opt_einsum) for
contrast. Fits log-log slopes so "linear in N" is a measured claim, not an assumption.
"""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

import models

dev = torch.device('cuda:0')
T, C = 12, 5
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NS = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else [250, 500, 1000, 2000, 4000, 8000, 16000]
res = []


def step_cost(N, enc, linear_contraction=True, reps=10):
    models.use_linear_contraction(linear_contraction)
    ma = SimpleNamespace(model_dim=128, recur_num=16, topk=4, node_num=N, feature_dim=C - 2,
                         second=0, minute=288, hour=0, day=7, week=0, weekday=0, month=0,
                         quarter=0, year=0, bs=BS)
    m = models.MAGEX(ma, encoder=enc, node_num=N, input_dim=C, output_dim=1, seq_len=T, horizon=12).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x = torch.randn(BS, T, N, C, device=dev); x[..., -2:] = torch.rand(BS, T, N, 2, device=dev)
    y = torch.randn(BS, 12, N, 1, device=dev)
    bias = torch.zeros(3, 16, device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    def one():
        out, _ = m(x, bias=bias); loss = (out - y).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(3): one()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): one()
    torch.cuda.synchronize()
    r = dict(N=N, enc=enc, contraction='linear' if linear_contraction else 'official_einsum', T=T,
             ms=(time.time() - t0) / reps * 1000, mem_mb=torch.cuda.max_memory_allocated(dev) / 2**20,
             params=models.n_params(m), enc_params=models.n_params(m.encoder))
    m.eval()
    with torch.no_grad():
        for _ in range(3): m(x, bias=bias)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps): m(x, bias=bias)
        torch.cuda.synchronize()
    r['infer_ms'] = (time.time() - t0) / reps * 1000
    del m, opt
    return r


for enc in ['linear', 'mslin', 'm3', 'm4', 'm5']:
    for N in NS:
        try:
            r = step_cost(N, enc)
        except torch.OutOfMemoryError:
            r = dict(N=N, enc=enc, contraction='linear', T=T, oom=True)
        res.append(r); print(r, flush=True)
for N in NS:
    try:
        r = step_cost(N, 'linear', linear_contraction=False, reps=3)
    except torch.OutOfMemoryError:
        r = dict(N=N, enc='linear', contraction='official_einsum', T=T, oom=True)
        res.append(r); print(r, flush=True)
        break
    res.append(r); print(r, flush=True)
    torch.cuda.empty_cache()

for key in sorted({(r['enc'], r['contraction']) for r in res}):
    pts = [(r['N'], r['ms'], r['mem_mb']) for r in res if (r['enc'], r['contraction']) == key and 'ms' in r and r['N'] >= 1000]
    if len(pts) >= 3:
        n, t, mm = map(np.log, zip(*pts))
        print(key, 'slope time %.2f  mem %.2f' % (np.polyfit(n, t, 1)[0], np.polyfit(n, mm, 1)[0]))
json.dump(res, open(f'results/cost_bs{BS}_{"-".join(map(str, NS))}.json', 'w'), indent=1)

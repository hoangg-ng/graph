"""Step-8 pre-check: does correlation reach further along the road graph at coarser time scales?

Data only, training span. Flow is de-seasonalised (per-node 5-min slot-of-week mean), then
average-pooled by s in {1, 2, 4, 8}. For every sensor pair, the hop distance on the undirected
ASTGNN road graph (BFS) is computed. Reported: mean pairwise correlation by hop 1..8, per scale,
and the decay ratio corr(hop h) / corr(hop 1). Writes results/precheck_hops.json.
"""
import json
import os
from collections import deque

import numpy as np

np.seterr(invalid="ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
out = {}
for ds in ['PEMS08', 'PEMS04']:
    raw = np.load(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.npz')['data'][..., 0].astype(np.float64)
    tr = np.load(f'{ROOT}/data/{ds}/idx_train.npy')
    flow = raw[:tr[-1] + 1]
    T, N = flow.shape
    slot = np.arange(T) % (288 * 7)
    seas = np.stack([flow[slot == s].mean(0) for s in range(288 * 7)])
    r = flow - seas[slot]
    if os.environ.get("NOCOMMON"):  # remove the network-wide common shock at each time step
        r = r - r.mean(1, keepdims=True)
    ed = np.loadtxt(f'{ROOT}/raw/ASTGNN/data/{ds}/{ds}.csv', delimiter=',', skiprows=1)[:, :2].astype(int)
    nb = [set() for _ in range(N)]
    for a, b in ed:
        if a != b:
            nb[a].add(b); nb[b].add(a)
    hop = np.full((N, N), -1)
    for i in range(N):
        hop[i, i] = 0; q = deque([i])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if hop[i, v] < 0:
                    hop[i, v] = hop[i, u] + 1; q.append(v)
    iu = np.triu_indices(N, 1)
    res = {}
    for s in (1, 2, 4, 8):
        x = r[T % s:].reshape(-1, s, N).mean(1)
        c = np.corrcoef(x.T)[iu]; h = hop[iu]
        curve = [float(np.nanmean(c[h == k])) if (h == k).any() else float('nan') for k in range(1, 9)]
        res[s] = dict(corr_by_hop=curve, decay_vs_hop1=[v / curve[0] for v in curve],
                      unreachable_pairs_corr=float(np.nanmean(c[h < 0])) if (h < 0).any() else None)
        print(ds, 'x%d' % s, np.round(curve, 3), 'ratio', np.round(res[s]['decay_vs_hop1'], 2), flush=True)
    out[ds] = dict(scales=res, pairs_per_hop=[int(((hop[iu]) == k).sum()) for k in range(1, 9)],
                   unreachable_pairs=int((hop[iu] < 0).sum()))
    print(ds, 'pairs per hop', out[ds]['pairs_per_hop'], 'unreachable', out[ds]['unreachable_pairs'], flush=True)
json.dump(out, open(f'{ROOT}/results/precheck_hops' + ('_nocommon' if os.environ.get('NOCOMMON') else '') + '.json', 'w'), indent=1)

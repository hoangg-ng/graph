"""2026-11-18: fair unseen-sensor comparison. Every trained checkpoint (slim MAGE, 20 % unseen sensors, rng 123) is
re-tested on the SAME unseen sensors with 0 days (identity = mean of known sensors), last 1 / 3 / 7 days, or the full
training span. Training does not depend on the unseen history, so each checkpoint covers every column; methods with
several identically trained checkpoints (e.g. window training at k 1/3/7) give a mean and a spread.
Results -> results/history/<ckpt name>.json, table printed at the end. Runs in one process on one GPU.
learned / none / SCPT have no checkpoints: their stored unseen MAE is used (learned / none do not use unseen history)."""
import glob
import json
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import train

METHODS = {   # name: checkpoint globs (per dataset)
    'learned embedding':         'results/profile/{ds}_T96_lite_hold_learned_s*.json results/idty/{ds}_T96_lite_hold_learned_s0.json',
    'no identity':               'results/profile/{ds}_T96_lite_hold_noid_s*.json results/idty/{ds}_T96_lite_hold_noid_s0.json',
    'SCPT (fixed)':              'results/profile/{ds}_T96_lite_hold_scpt_s0.json',
    'PCA (fixed, as published)': 'results/coldstart/ckpt/{ds}_pca_fixed_k*.pt',
    'raw profile (fixed)':       'results/relative/ckpt/{ds}_hraw_k*.pt',
    'relative profile (fixed)':  'results/relative/ckpt/{ds}_hrel_k*.pt',
    'PCA + window':              'results/coldstart/ckpt/{ds}_pca_win_k*.pt',
    'PCA of relative + window':  'results/coldstart/ckpt/{ds}_pcarel_win_k*.pt',
    'relative + window (ours)':  'results/coldstart/ckpt/{ds}_win_k*.pt',
    'relative + occ/speed + window (ours)': 'results/coldstart/ckpt/{ds}_win_os_k*.pt',
}
os.makedirs('results/history', exist_ok=True)
DAYS = ['0', '1', '3', '7', 'full']
table = {}
for ds in ('PEMS04', 'PEMS08'):
    for m, pat in METHODS.items():
        cks = sorted({c for g in pat.format(ds=ds).split() for c in glob.glob(g)})
        res = []
        for ck in cks:
            if ck.endswith('.json'):           # no checkpoint saved: learned / none ignore unseen history; SCPT whole span only
                u = json.load(open(ck))['unseen']['unseen_mae']
                res.append({d: u for d in DAYS} if 'scpt' not in ck else {'full': u}); continue
            out = f'results/history/{os.path.basename(ck).replace(".pt", ".json")}'
            if not os.path.exists(out):
                args = __import__('torch').load(ck, map_location='cpu')['args']
                argv = []
                for k in ('dataset', 'seq_len', 'arm', 'seed', 'tf32', 'data_dir', 'holdout', 'identity', 'id_window', 'id_os', 'profile_days'):
                    if k in args and args[k] is not None:
                        argv += [f'--{k}', str(args[k])]
                argv += ['--out', out, '--eval_ckpt', ck]
                print(ds, m, os.path.basename(ck), flush=True)
                train.main(argv)
            res.append(json.load(open(out))['unseen_mae'])
        table[(ds, m)] = (len(res), {d: [r[d] for r in res if d in r] for d in DAYS})
for ds in ('PEMS04', 'PEMS08'):
    print(f'\n{ds} unseen-sensor MAE (mean [min-max] over checkpoints)')
    print(f'{"method":40s} n ' + ''.join(f'{d:>18s}' for d in DAYS))
    for m in METHODS:
        n, r = table[(ds, m)]
        cells = ''.join((f'{np.mean(v):7.2f} [{min(v):.2f}-{max(v):.2f}]' if len(v) > 1 else f'{v[0]:7.2f}           ') if v else f'{"-":>18s}'
                        for v in (r[d] for d in DAYS))
        print(f'{m:40s} {n} {cells}')
json.dump({f'{k[0]}|{k[1]}': v for k, v in table.items()}, open('results/history/table.json', 'w'))

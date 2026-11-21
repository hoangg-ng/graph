"""Aggregate results/runs/*.json into RESULTS tables (markdown on stdout)."""
import glob
import json
from collections import defaultdict

import numpy as np

runs = [json.load(open(f)) for f in sorted(glob.glob('results/runs/*_T12_*.json'))]
runs = [r for r in runs if r['encoder'] in ('linear', 'mslin', 'm3', 'm4', 'm5')]
G = defaultdict(list)
for r in runs:
    G[(r['dataset'], r['seq_len'], r['encoder'])].append(r)
ENC = ['linear', 'mslin', 'm3', 'm4', 'm5']
LABEL = {'linear': '1 MAGE (linear)', 'mslin': '2 pooled→linear', 'm3': '3 1-scale MLP', 'm4': '4 MS global gate', 'm5': '5 MS node gate'}
DS = ['PEMS04', 'PEMS08']
REPORTED = {'PEMS03': 14.72, 'PEMS04': 18.16, 'PEMS07': 19.49, 'PEMS08': 13.66}  # MAGE Table 5, MAE


def ms(v):
    return f'{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f}'


for T in sorted({k[1] for k in G}):
    print(f'\n### Input window T = {T}  (test MAE / RMSE / MAPE%, mean ± sd over seeds)\n')
    print('| dataset | encoder | n | MAE | RMSE | MAPE | ΔMAE vs linear | epochs | s/epoch* | peak MB | enc params |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for ds in DS:
        base = {r['seed']: r['test_mae'] for r in G.get((ds, T, 'linear'), [])}
        for e in ENC:
            rs = G.get((ds, T, e))
            if not rs:
                continue
            d = [r['test_mae'] - base[r['seed']] for r in rs if r['seed'] in base]
            delta = f'{np.mean(d):+.2f} ({sum(x < 0 for x in d)}/{len(d)} better)' if d and e != 'linear' else ''
            print(f"| {ds} | {LABEL[e]} | {len(rs)} | {ms([r['test_mae'] for r in rs])} | {ms([r['test_rmse'] for r in rs])} | "
                  f"{ms([r['test_mape'] for r in rs])} | {delta} | {np.mean([r['epochs_run'] for r in rs]):.0f} | "
                  f"{np.mean([r['train_s_per_epoch'] for r in rs]):.1f} | {np.mean([r['peak_mem_mb'] for r in rs]):.0f} | {rs[0]['enc_params']} |")
        if T == 12 and ds in REPORTED:
            print(f'| {ds} | *MAGE paper* | 5 | {REPORTED[ds]:.2f} | | | | | | | |')

    print(f'\nHorizon MAE (T={T}), mean over seeds: @3 / @6 / @12\n')
    print('| dataset | ' + ' | '.join(ENC) + ' |')
    print('|---|' + '---|' * len(ENC))
    for ds in DS:
        cells = []
        for e in ENC:
            rs = G.get((ds, T, e))
            cells.append(' / '.join(f"{np.mean([r['per_horizon'][h][0] for r in rs]):.2f}" for h in (2, 5, 11)) if rs else '')
        if any(cells):
            print(f'| {ds} | ' + ' | '.join(cells) + ' |')
print('\n*s/epoch measured with 4 jobs sharing one GPU; use cost.py for clean timings.')

import re
print('\n### Reproduction: model 1 through the official code (lr 0.01, no load balancing)\n')
print('| dataset | seed | epochs | MAE | RMSE | MAPE |')
print('|---|---|---|---|---|---|')
for ds, k in [('PEMS04', '4'), ('PEMS08', '8')]:
    for f in sorted(glob.glob(f'official_repro/experiments/mage/{k}/record_s*.log')):
        t = open(f).read()
        m = re.search(r'Average Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)', t)
        ep = len(re.findall(r'Epoch: \d+', t))
        seed = re.search(r'record_s(\d+)_', f).group(1)
        print(f'| {ds} | {seed} | {ep} | ' + (' | '.join([m.group(1), m.group(2), f'{float(m.group(3))*100:.2f}']) if m else 'running | | ') + ' |')

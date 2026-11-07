"""Summarise the PEMS04 step-8 runs (run_pems04.py) against linear MAGE and apply the decision rules.

Rule (research_design.md, fixed before results): "A beats B" = lower mean MAE and better in >= 4/5 paired seeds;
differences under ~0.05 MAE are not interpretable. With 1 seed (screen) the go rule is:
proposed beats shared AND reversed by > 0.1, and is not worse than MAGE by more than 0.1."""
import json
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def mae(path):
    return json.load(open(path))['test_mae'] if os.path.exists(path) else None


def runs(fmt):
    return {s: v for s in range(5) if (v := mae(fmt.format(s=s))) is not None}


ref = {'linear MAGE': runs('results/long/PEMS04_T96_linear_s{s}.json'),
       'MAGE, same setting as arms': runs('results/pems04/PEMS04_T96_mage_mage_s{s}.json'),
       'MAGE, bounded lambda': runs('results/pems04/PEMS04_T96_mage_sigmoid_s{s}.json'),
       'MAGE, no space (nomix1)': runs('results/space/PEMS04_T96_H12_nomix1_s{s}.json')}
arms = {f'{arm} ({lam} lambda)' if arm != 'nomix' else '4 streams, no space (nomix)':
        runs(f'results/pems04/PEMS04_T96_{arm}_{lam}_s{{s}}.json')
        for lam in ('mage', 'sigmoid') for arm in ('shared', 'proposed', 'reversed', 'nomix')
        if not (arm == 'nomix' and lam == 'sigmoid')}
# C4 baseline: MAGE run in the same setting (GPU type, TF32) if available, else the step-7 runs (A6000, fp32)
base = ref['MAGE, same setting as arms'] or ref['linear MAGE']


def paired(a, b):
    s = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in s])
    return (f'{d.mean():+.3f}' + (f' ± {d.std(ddof=1):.3f}' if len(d) > 1 else '') + f', better {int((d < 0).sum())}/{len(d)}') if len(d) else '–'


print(f'{"arm":32s} {"seeds":>5s} {"MAE":>16s}   vs MAGE baseline (paired)')
for name, r in {**ref, **arms}.items():
    if not r:
        print(f'{name:32s} {"0":>5s}   (not run yet)'); continue
    v = np.array(list(r.values()))
    m = f'{v.mean():.3f}' + (f' ± {v.std(ddof=1):.3f}' if len(v) > 1 else '')
    print(f'{name:32s} {len(v):5d} {m:>16s}   {paired(r, base) if r is not base else "(baseline)"}')

print('\nComparisons (negative = first arm better):')
for lam in ('mage', 'sigmoid'):
    P, S, R = (arms[f'{a} ({lam} lambda)'] for a in ('proposed', 'shared', 'reversed'))
    print(f'  [{lam} lambda] C1 proposed - shared: {paired(P, S)} | C2 proposed - reversed: {paired(P, R)}'
          f' | C4 proposed - MAGE: {paired(P, base)}')
    if len(P) == 1 and len(S) and len(R) and base:
        s = next(iter(P)); p = P[s]
        go = p < S.get(s, 1e9) - 0.1 and p < R.get(s, 1e9) - 0.1 and p < base[s] + 0.1
        print(f'    screen (seed {s}): {"GO -> run seeds 1-4" if go else "STOP: no clear effect"}')
NS, N4, SH = ref['MAGE, no space (nomix1)'], arms['4 streams, no space (nomix)'], arms['shared (mage lambda)']
print(f'\n  Value of space, plain MAGE (nomix1 - MAGE): {paired(NS, base)}')
print(f'  Value of space, 4 streams (nomix - shared): {paired(N4, SH)}')
print('  If the second is much smaller, the scale streams substitute for spatial mixing (as on PEMS08: 0.03 vs 0.22).')
MB = ref['MAGE, bounded lambda']
print(f'\nPart A, is bounding lambda a fix? MAGE bounded - MAGE: {paired(MB, base)}'
      '  (rule: lower mean and better in >= 4/5 seeds; under ~0.05 is noise)')

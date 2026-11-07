"""Summarise run_hier.py: test / val MAE vs MAGE (same GPU and precision), and whether the graph stayed structured."""
import glob
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
rows = {}
for f in glob.glob('results/hier/*.json') + glob.glob('results/pems04/PEMS04_T96_mage_*_s*.json'):
    r = json.load(open(f))
    key = (r['dataset'], r['seed'], r.get('lam', 'mage'), r.get('graph_fix', 0), r.get('hier_rank', 0))
    rows[key] = r
names = {(0, 0): 'MAGE (released)', (1, 0): '+ graph fix', (1, 32): '+ graph fix + hierarchy'}
print(f'{"dataset":7s} {"variant":40s} {"test MAE":>9s} {"val MAE":>8s} {"vs MAGE":>8s} {"epochs":>6s}  graph entropy b1-b3 (1 = collapsed to mean)')
for ds in ('PEMS08', 'PEMS04'):
    for seed in sorted({k[1] for k in rows if k[0] == ds}):
        base = rows.get((ds, seed, 'mage', 0, 0))
        for (d, s, lam, f, h), r in sorted(rows.items(), key=lambda kv: (kv[0][3], kv[0][4], kv[0][2])):
            if d != ds or s != seed:
                continue
            name = names.get((f, h), f'fix={f} rank={h}') + (' + bounded lambda' if lam == 'sigmoid' else '')
            dv = f'{r["test_mae"] - base["test_mae"]:+.3f}' if base and r is not base else ''
            ge = r.get('graph_entropy')
            ge = ' '.join(f'{v:.3f}' for v in ge) if ge else '(not logged)'
            print(f'{ds:7s} {name:40s} {r["test_mae"]:9.3f} {r["best_val_mae"]:8.3f} {dv:>8s} {r["epochs_run"]:6d}  {ge}')

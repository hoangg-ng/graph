"""Where does space matter? MAGE (arm 'mage') vs the same model with the spatial operator set to 0
(arm 'nomix1'), T = 12, 2 seeds (follow-up: does space matter more with a short window, as in the paper?). Value of space = MAE(nomix1) - MAE(mage).
PEMS04/08 at horizon 12 reuse the step-7 linear MAGE runs (same code path, checked in test_streams.py),
so only nomix1 is run there. PEMS03/07 are flow-only; PEMS08 h48 uses data_w288_h48.
Skips runs that already have a log. GPU 1 is left free."""
import os, subprocess, time
SLOTS = {0: 2, 2: 2, 3: 2}
PY = os.path.abspath('.venv/bin/python')
jobs = []
for seed in (0, 1):
    jobs += [('PEMS03', 12, arm, seed, 'data_w288') for arm in ('mage', 'nomix1')]
    jobs += [('PEMS08', 12, 'nomix1', seed, 'data_w288')]   # mage side = step-7 linear T=12
name = lambda j: f'{j[0]}_T12_H{j[1]}_{j[2]}_s{j[3]}'
os.makedirs('logs/space', exist_ok=True); os.makedirs('results/space', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/space/{name(j)}.log')]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [p for p in running[g] if p.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            j = jobs.pop(0); n = name(j)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen(
                [PY, 'train.py', '--dataset', j[0], '--seq_len', '12', '--horizon', str(j[1]), '--arm', j[2],
                 '--seed', str(j[3]), '--data_dir', j[4], '--out', f'results/space/{n}.json'],
                env=env, stdout=open(f'logs/space/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

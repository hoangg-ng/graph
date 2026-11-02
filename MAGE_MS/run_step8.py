"""Step 8 cheap check: PEMS08, T = 96, arms x seeds 0-1 (+ arm mage seed 0 as code check).
Free GPUs on 2026-10-31: 3 (A6000, 5 jobs) and 0 (A4500, 4 jobs). Skips runs that already have a log."""
import os, subprocess, time
SLOTS = {3: 3, 0: 3}
PY = os.path.abspath('.venv/bin/python')
jobs = [('PEMS08', 96, 'mage', 0)] + [('PEMS08', 96, arm, seed) for seed in (0, 1)
                                      for arm in ['proposed', 'shared', 'uniform', 'nomix']]
# stage 2 (2026-10-31 05:15): proposed was not clearly dead after seeds 0-1, so seeds 2-4 for proposed/shared
jobs += [('PEMS08', 96, arm, seed) for seed in (2, 3, 4) for arm in ['proposed', 'shared']]
name = lambda j: f'{j[0]}_T{j[1]}_{j[2]}_s{j[3]}'
os.makedirs('logs/step8', exist_ok=True); os.makedirs('results/step8', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/step8/{name(j)}.log')]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [p for p in running[g] if p.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            j = jobs.pop(0); n = name(j)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen(
                [PY, 'train.py', '--dataset', j[0], '--seq_len', str(j[1]), '--arm', j[2], '--seed', str(j[3]),
                 '--data_dir', 'data_w288', '--out', f'results/step8/{n}.json'],
                env=env, stdout=open(f'logs/step8/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

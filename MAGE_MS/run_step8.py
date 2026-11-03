"""Step 8: PEMS08, T = 96, arms x seeds 0-1 (+ arm mage seed 0 as code check).
Prepare the disjoint-target split first with `python prep.py 288 data_w288_fixed 08`.
Uses separate fixed-data, log, and result directories to avoid mixing older runs."""
import os, subprocess, time
SLOTS = {3: 3, 0: 3}
PY = os.path.abspath('.venv/bin/python')
DATA_DIR = 'data_w288_fixed'
LOG_DIR = 'logs/step8_fixed'
RESULT_DIR = 'results/step8_fixed'
jobs = [('PEMS08', 96, 'mage', 0)] + [('PEMS08', 96, arm, seed) for seed in (0, 1)
                                      for arm in ['proposed', 'reversed', 'shared', 'uniform', 'nomix']]
# stage 2 (2026-10-31 05:15): proposed was not clearly dead after seeds 0-1, so seeds 2-4 for proposed/shared
jobs += [('PEMS08', 96, arm, seed) for seed in (2, 3, 4) for arm in ['proposed', 'shared']]
name = lambda j: f'{j[0]}_T{j[1]}_{j[2]}_s{j[3]}'
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(RESULT_DIR, exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'{LOG_DIR}/{name(j)}.log')]
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
                 '--data_dir', DATA_DIR, '--out', f'{RESULT_DIR}/{n}.json'],
                env=env, stdout=open(f'{LOG_DIR}/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

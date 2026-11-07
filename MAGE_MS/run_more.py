"""2026-11-02: GPU 1 opened up. Takes over the unfinished part of run_space.py (whose launcher was
stopped; its running jobs continue) and adds the pending step-8 arms on PEMS08, T = 96:
  - PEMS07 T96 seed 0 (mage, nomix1): restarted here from GPU 0, which was nearly out of memory;
  - the 4 run_space.py jobs that had not started yet;
  - step 8 'reversed' (8, 4, 2, 1) and 'uniform0' (pure node mean, no learned S-), seeds 0-4.
Skips runs that already have a log. GPU 0 gets no new jobs."""
import os, subprocess, time
SLOTS = {1: 8, 2: 2, 3: 1}
PY = os.path.abspath('.venv/bin/python')

# (log/result name, args)
space = [('PEMS07', 12, a, 0, 'data_w288') for a in ('mage', 'nomix1')]
space += [('PEMS08', 48, a, 1, 'data_w288_h48') for a in ('mage', 'nomix1')]
space += [(ds, 12, 'nomix1', 1, 'data_w288') for ds in ('PEMS04', 'PEMS08')]
jobs = [(f'space/{d}_T96_H{h}_{a}_s{s}',
         ['--dataset', d, '--seq_len', '96', '--horizon', str(h), '--arm', a, '--seed', str(s), '--data_dir', dd])
        for d, h, a, s, dd in space]
step8 = [(arm, s) for arm in ('reversed', 'uniform0') for s in range(5)]
step8.sort(key=lambda j: j[0] != 'reversed')          # the slow arm first
jobs += [(f'step8/PEMS08_T96_{a}_s{s}',
          ['--dataset', 'PEMS08', '--seq_len', '96', '--arm', a, '--seed', str(s), '--data_dir', 'data_w288'])
         for a, s in step8]

for d in ('space', 'step8'):
    os.makedirs(f'logs/{d}', exist_ok=True); os.makedirs(f'results/{d}', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/{j[0]}.log')]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [p for p in running[g] if p.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            n, args = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen([PY, 'train.py', *args, '--out', f'results/{n}.json'],
                                               env=env, stdout=open(f'logs/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

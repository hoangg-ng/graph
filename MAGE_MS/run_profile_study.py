"""2026-11-08: does the computed sensor profile hold up? Slim MAGE (arm 'lite'), TF32, GPUs 0 and 3, 8 jobs each.
Identity: learned embedding / computed profile / none; each on all sensors and with 20 % unseen sensors (holdout).
  A: seeds 1, 2 on PEMS04 / PEMS08, 96 -> 12          (seed 0 already in results/lite and results/idty)
  B: PEMS03, PEMS07, seed 0, 96 -> 12
  C: MAGE's own setting 12 -> 12, PEMS04 / PEMS08, seed 0
  D: unseen-sensor test with the profile from only the last 3 / 7 days, PEMS04 / PEMS08, seed 0
Results -> results/profile/<name>.json; skips runs that already have a log."""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('.venv/bin/python')
SLOTS = {0: 8, 3: 8}
VARIANTS = {'learned': [], 'profile': ['--identity', 'profile'], 'noid': ['--identity', 'none'],
            'hold_learned': ['--holdout', '0.2'], 'hold_profile': ['--holdout', '0.2', '--identity', 'profile'],
            'hold_noid': ['--holdout', '0.2', '--identity', 'none']}
jobs = []
def add(ds, T, seed, v, extra=(), tag=''):
    n = f'{ds}_T{T}_lite_{v}{tag}_s{seed}'
    jobs.append((n, ['train.py', '--dataset', ds, '--seq_len', str(T), '--arm', 'lite', '--seed', str(seed), '--tf32', '1',
                     '--data_dir', 'data_w288', '--out', f'results/profile/{n}.json'] + VARIANTS[v] + list(extra)))
for ds in ('PEMS07', 'PEMS03'):                       # largest first
    for v in VARIANTS: add(ds, 96, 0, v)
for seed in (1, 2):
    for ds in ('PEMS04', 'PEMS08'):
        for v in VARIANTS: add(ds, 96, seed, v)
for ds in ('PEMS04', 'PEMS08'):
    for v in VARIANTS: add(ds, 12, 0, v)
    for d in (3, 7): add(ds, 96, 0, 'hold_profile', ['--profile_days', str(d)], f'_{d}d')
os.makedirs('logs/profile', exist_ok=True); os.makedirs('results/profile', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/profile/{j[0]}.log')]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [q for q in running[g] if q.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            n, argv = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen([PY] + argv, env=env, stdout=open(f'logs/profile/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(20)
print('ALL DONE', flush=True)

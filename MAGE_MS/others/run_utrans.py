"""2026-11-09: U-Trans-style comparison (STAEformer + network-mean spatial mixing), seed 0, GPU 2 only.
Identity: learned adaptive embedding / computed profile / none, on all sensors and with 20 % unseen sensors.
Results -> results/staeformer_<ds>_s0_mean[_idprofile|_idnone][_hold].json; skips runs that already have a log."""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('../.venv/bin/python')
SLOTS = {int(k): int(v) for k, v in (s.split(':') for s in os.environ.get('UTRANS_SLOTS', '2:8').split(','))}   # e.g. UTRANS_SLOTS=0:4
jobs = []
for ds in ('PEMS04', 'PEMS08'):
    for ident in ('learned', 'profile', 'none'):
        for hold in (False, True):
            n = f'utrans_{ds}_{ident}{"_hold" if hold else ""}'
            jobs.append((n, ['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0', '--mixing', 'mean',
                             '--epochs', '80', '--identity', ident] + (['--holdout', '0.2'] if hold else [])))
jobs = [j for j in jobs if not os.path.exists(f'logs/{j[0]}.log')]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [q for q in running[g] if q.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            n, argv = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen([PY] + argv, env=env, stdout=open(f'logs/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(20)
print('ALL DONE', flush=True)

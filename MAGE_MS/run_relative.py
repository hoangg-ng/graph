"""2026-11-18: does dividing by the network mean give unseen sensors a better identity, and better forecasts?
Slim MAGE (arm lite, 96 -> 12), seed 0, 20 % unseen sensors (rng 123), TF32. Identity = 24-value hourly profile:
  hraw : log hourly flow                                   hrel : log(sensor / network mean of known sensors)
Known sensors use their whole training span; unseen sensors only their last k days before the test period
(k = 1, 3, 7; 0 = whole span). Last days: PEMS04 ends Mon..Sun, PEMS08 ends Sun..Sat (weekend-heavy for k <= 3).
Reference already run: no identity (results/idty/<ds>_T96_lite_hold_noid_s0.json).
GPUs 6 and 7 on achilleas, 8 jobs each, 2 batch processes per GPU (batch_run.py). Results -> results/relative/<ds>_<kind>_k<k>.json; skips runs that have a log."""
import os
import json
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('.venv/bin/python')
SLOTS = {6: 8, 7: 8}   # jobs per GPU
PROCS = 2              # batch processes per GPU (2026-11-18: one process is GIL-bound at ~2 cores, GPU ~60 % busy)
jobs = []
for ds in ('PEMS04', 'PEMS08'):
    for k in (1, 3, 7, 0):
        for kind in ('hraw', 'hrel'):
            n = f'{ds}_{kind}_k{k}'
            jobs.append((n, ['train.py', '--dataset', ds, '--seq_len', '96', '--arm', 'lite', '--seed', '0', '--tf32', '1',
                             '--data_dir', 'data_w288', '--holdout', '0.2', '--identity', kind, '--unseen_days', str(k),
                             '--out', f'results/relative/{n}.json']))
for d in ('logs/relative', 'results/relative'):
    os.makedirs(d, exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'results/relative/{j[0]}.json')]
print(len(jobs), 'jobs', flush=True)
# 2026-11-18: one process per GPU (batch_run.py runs its jobs as threads), so each GPU shows a single task
procs = []
workers = [(g, w) for w in range(PROCS) for g in SLOTS]
for i, (g, w) in enumerate(workers):
    mine = [(f'logs/relative/{n}.log', argv[1:]) for n, argv in jobs[i::len(workers)]]
    if not mine:
        continue
    jf = f'logs/relative/batch_gpu{g}_{w}.json'; json.dump(mine, open(jf, 'w'))
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
    procs.append(subprocess.Popen([PY, 'batch_run.py', jf], env=env, stdout=open(f'logs/relative/batch_gpu{g}_{w}.out', 'w'),
                                  stderr=subprocess.STDOUT))
    print(time.strftime('%m-%d %H:%M'), 'gpu', g, 'process', w, ':', len(mine), 'jobs', flush=True)
for p in procs:
    p.wait()
print('ALL DONE', flush=True)

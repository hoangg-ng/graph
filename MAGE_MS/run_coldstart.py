"""2026-11-18: cold start. The 1-day network-relative identity is already 79-96 % right, yet forecasts were worse than no
identity: the model, trained only with precise whole-span identities, is too sensitive to identity error.
Fix tested: train with identities resampled from random 1-7 day windows (whole span with prob 0.3), optionally adding
occupancy and speed relative to the network mean. Slim MAGE, seed 0, 20 % unseen sensors (rng 123), TF32.
  win     : --identity hrel --id_window 1
  win_os  : --identity hrel --id_window 1 --id_os 1
Unseen sensors at test: identity from their last k days, k = 1, 3, 7 (same windows as results/relative).
Baselines: results/relative/<ds>_hrel_k<k>.json (same identity, fixed training) and no identity (results/idty).
GPUs 6 and 7, 2 batch processes per GPU (batch_run.py). Results -> results/coldstart/<ds>_<variant>_k<k>.json."""
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('.venv/bin/python')
GPUS, PROCS = (6, 7), 2
VARIANTS = {'win': ['--id_window', '1'], 'win_os': ['--id_window', '1', '--id_os', '1']}
jobs = []
for ds in ('PEMS04', 'PEMS08'):
    for k in (1, 3, 7):
        for v, extra in VARIANTS.items():
            n = f'{ds}_{v}_k{k}'
            jobs.append((n, ['--dataset', ds, '--seq_len', '96', '--arm', 'lite', '--seed', '0', '--tf32', '1',
                             '--data_dir', 'data_w288', '--holdout', '0.2', '--identity', 'hrel', '--unseen_days', str(k),
                             '--out', f'results/coldstart/{n}.json'] + extra))
for d in ('logs/coldstart', 'results/coldstart'):
    os.makedirs(d, exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'results/coldstart/{j[0]}.json')]
print(len(jobs), 'jobs', flush=True)
procs, workers = [], [(g, w) for w in range(PROCS) for g in GPUS]
for i, (g, w) in enumerate(workers):
    mine = [(f'logs/coldstart/{n}.log', argv) for n, argv in jobs[i::len(workers)]]
    if not mine:
        continue
    jf = f'logs/coldstart/batch_gpu{g}_{w}.json'; json.dump(mine, open(jf, 'w'))
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
    procs.append(subprocess.Popen([PY, 'batch_run.py', jf], env=env, stdout=open(f'logs/coldstart/batch_gpu{g}_{w}.out', 'w'),
                                  stderr=subprocess.STDOUT))
    print(time.strftime('%m-%d %H:%M'), 'gpu', g, 'process', w, ':', len(mine), 'jobs', flush=True)
for p in procs:
    p.wait()
print('ALL DONE', flush=True)

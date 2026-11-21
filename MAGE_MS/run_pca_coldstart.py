"""2026-11-18: cold start, PCA comparison (same protocol as run_coldstart.py: slim MAGE, seed 0, 20 % unseen sensors,
unseen identity from their last k = 1 / 3 / 7 days, GPUs 6 and 7, 2 batch processes per GPU).
  pca_fixed  : PCA embedding (arXiv 2411.11448, k = 8 components) trained with whole-span identities, as published
  pca_win    : PCA embedding + training with identities from random 1-7 day windows
  pcarel_win : PCA of network-relative day curves log(sensor / network mean) + window training
Compare with results/coldstart/<ds>_win_k<k>.json (our relative hourly identity + window training).
Results -> results/coldstart/<ds>_<variant>_k<k>.json."""
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('.venv/bin/python')
GPUS, PROCS = (6, 7), 2
VARIANTS = {'pca_fixed': ['--identity', 'wpca'], 'pca_win': ['--identity', 'wpca', '--id_window', '1'],
            'pcarel_win': ['--identity', 'wpcarel', '--id_window', '1']}   # PCA as published / + window training / of relative curves
jobs = []
for ds in ('PEMS04', 'PEMS08'):
    for k in (1, 3, 7):
        for v, extra in VARIANTS.items():
            n = f'{ds}_{v}_k{k}'
            jobs.append((n, ['--dataset', ds, '--seq_len', '96', '--arm', 'lite', '--seed', '0', '--tf32', '1',
                             '--data_dir', 'data_w288', '--holdout', '0.2', '--unseen_days', str(k),
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
    jf = f'logs/coldstart/pca_batch_gpu{g}_{w}.json'; json.dump(mine, open(jf, 'w'))
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
    procs.append(subprocess.Popen([PY, 'batch_run.py', jf], env=env, stdout=open(f'logs/coldstart/pca_batch_gpu{g}_{w}.out', 'w'),
                                  stderr=subprocess.STDOUT))
    print(time.strftime('%m-%d %H:%M'), 'gpu', g, 'process', w, ':', len(mine), 'jobs', flush=True)
for p in procs:
    p.wait()
print('ALL DONE', flush=True)

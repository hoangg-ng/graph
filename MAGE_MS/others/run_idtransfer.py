"""2026-11-13: why does computed identity fail on unseen sensors in the U-Trans-style backbone (STAEformer + network
mean)? Hypothesis: the 80-dim identity concatenated to every token lets the model memorise the training sensors.
All runs: PCA identity, 20 % unseen sensors (rng 123), seed 0, 80 epochs. Each variant changes one thing:
  k2, k4        fewer principal components              (less information to key on)
  dim8          8-wide bottleneck before the 80-dim slot (less capacity)
  noise0.5      Gaussian noise on the features           (no exact memorisation)
  mix0.5        blend with a random training sensor      (no exact memorisation)
  drop0.3       identity zeroed for 30 % of sensors      (cannot rely on identity alone)
  affine        identity only scales / shifts the output (slim-MAGE-like low-capacity pathway)
Baselines already run (2026-11-09): identity none / pca (k 8) / profile / scpt, same split.
Memorisation confirmed if 1-3 of these lower unseen MAE with seen MAE close to the k 8 run.
GPUs: UTRANS_SLOTS="gpu:jobs,..." (e.g. 0:4,3:4 on lerna). PEMS08 first (faster, and where identity hurts most).
Results -> results/staeformer_<ds>_s0_mean_idpca_hold_<variant>.json; skips runs that already have a log."""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('../.venv/bin/python')
SLOTS = {int(k): int(v) for k, v in (s.split(':') for s in os.environ['UTRANS_SLOTS'].split(','))}
VARIANTS = {'k2': ['--pca_k', '2'], 'k4': ['--pca_k', '4'], 'dim8': ['--id_dim', '8'], 'noise0.5': ['--id_noise', '0.5'],
            'mix0.5': ['--id_mix', '0.5'], 'drop0.3': ['--id_drop', '0.3'], 'affine': ['--id_mode', 'affine']}
jobs = []
for ds in ('PEMS08', 'PEMS04'):
    for v, extra in VARIANTS.items():
        jobs.append((f'logs/idtransfer/{ds}_{v}.log',
                     ['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0', '--mixing', 'mean',
                      '--epochs', '80', '--identity', 'pca', '--holdout', '0.2'] + extra))
os.makedirs('logs/idtransfer', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(j[0])]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [q for q in running[g] if q.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            log, argv = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen([PY] + argv, env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', os.path.basename(log), 'gpu', g, flush=True)
    time.sleep(20)
print('ALL DONE', flush=True)

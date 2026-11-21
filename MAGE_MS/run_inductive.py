"""2026-11-09: published inductive identity baselines vs our computed profile, seed 0, 20 % unseen sensors (rng 123).
Identity = PCA embedding (arXiv 2411.11448) or SCPT-style contrastive encoder (arXiv 2305.05237), see id_features.py.
  U-Trans-style (STAEformer + network mean), PEMS04 / PEMS08, holdout       -> others/results/staeformer_<ds>_s0_mean_id<k>_hold.json
  slim MAGE (arm lite, 96 -> 12), PEMS04 / PEMS08, holdout and all sensors  -> results/profile/<ds>_T96_lite[_hold]_<k>_s0.json
Learned / profile / none for the same settings already exist (results/idty, results/profile, others/results).
GPUs: 0 (A4500) and 3 (A6000) on lerna, as requested. Skips runs that already have a log."""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('.venv/bin/python')
SLOTS = {3: 6, 0: 4}
jobs = []   # (log, cwd, argv), longest first
for ds in ('PEMS04', 'PEMS08'):
    for k in ('pca', 'scpt'):
        jobs.append((f'logs/inductive/utrans_{ds}_{k}_hold.log', 'others',
                     ['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0', '--mixing', 'mean',
                      '--epochs', '80', '--identity', k, '--holdout', '0.2']))
for hold in (True, False):
    for ds in ('PEMS04', 'PEMS08'):
        for k in ('pca', 'scpt'):
            n = f'{ds}_T96_lite_{"hold_" if hold else ""}{k}_s0'
            jobs.append((f'logs/inductive/{n}.log', '.',
                         ['train.py', '--dataset', ds, '--seq_len', '96', '--arm', 'lite', '--seed', '0', '--tf32', '1',
                          '--data_dir', 'data_w288', '--identity', k, '--out', f'results/profile/{n}.json']
                         + (['--holdout', '0.2'] if hold else [])))
os.makedirs('logs/inductive', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(j[0])]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [q for q in running[g] if q.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            log, cwd, argv = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen([PY] + argv, cwd=os.path.join(HERE, cwd), env=env,
                                               stdout=open(log, 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', os.path.basename(log), 'gpu', g, flush=True)
    time.sleep(20)
print('ALL DONE', flush=True)

"""2026-11-07: one launcher for the "slim MAGE + structure", method A, method B and gain-decomposition runs.
Seed 0, PEMS04 and PEMS08, GPUs 0 and 4 (as requested). Skips runs that already have a log.

MAGE path (T = 96 -> 12, data_w288, TF32; same setting as results/lite and the MAGE baselines), -> results/ab/:
  lite_cof, lite_road            slim MAGE + fixed co-fluctuation / road graph (no extra parameters)
  lite_r2, lite_r8, lite_r32     method A: slim MAGE + learned rank-r graph (rank 0 = results/lite)
  mage --switch_at 1             method B: switch to slim MAGE after epoch 1 if the graph has collapsed
Other models (12 -> 12, each paper's recipe), -> others/results/<model>_<ds>_s0_<mixing>.json:
  gwnet, agcrn, mtgnn, staeformer, astgcn with --mixing mean (sensor average) and --mixing self (no spatial mixing)
"""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.abspath('.venv/bin/python')
SLOTS = {0: 8, 4: 8}
jobs = []   # (log path, cwd, argv); longest first
for ds in ('PEMS04', 'PEMS08'):
    for m in ('staeformer', 'agcrn', 'mtgnn', 'astgcn', 'gwnet'):
        for mx in ('mean', 'self'):
            script = 'train_other.py' if m in ('gwnet', 'agcrn') else 'train_other2.py'
            extra = ['--epochs', '80'] if m == 'staeformer' else []
            jobs.append((f'others/logs/{m}_{ds}_s0_{mx}.log', 'others',
                         [script, '--model', m, '--dataset', ds, '--seed', '0', '--mixing', mx] + extra))
for ds in ('PEMS04', 'PEMS08'):
    for arm, extra in [('lite_cof', []), ('lite_road', []), ('lite_r2', []), ('lite_r8', []), ('lite_r32', []),
                       ('mage', ['--switch_at', '1'])]:
        name = f'{ds}_T96_{arm}{"_switch1" if extra else ""}_s0'
        jobs.append((f'logs/ab/{name}.log', '.',
                     ['train.py', '--dataset', ds, '--seq_len', '96', '--arm', arm, '--seed', '0', '--tf32', '1',
                      '--data_dir', 'data_w288', '--out', f'results/ab/{name}.json'] + extra))
jobs.sort(key=lambda j: 0 if 'staeformer' in j[0] else (1 if 'PEMS04' in j[0] and 'others' in j[0] else 2))
for d in ('logs/ab', 'results/ab', 'others/logs', 'others/results'):
    os.makedirs(d, exist_ok=True)
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

"""Step 8 on PEMS04 (the proposal's primary dataset; space is worth ~0.43 MAE there), T = 96 -> 12.

Arms per seed (8 runs):
  lam = mage    : shared, proposed, reversed     (as on PEMS08)
  lam = sigmoid : shared, proposed, reversed     (lambda_e in (0, 1) per expert, so the power term is not swamped)
  nomix         : 4 scale streams, no spatial mixing (does multi-scale replace space on PEMS04 too?)
  mage          : MAGE itself in the same setting (GPU type, TF32), so C4 is not confounded by precision
Reference runs already exist: linear MAGE (results/long/PEMS04_T96_linear_s0-4), MAGE without
space (results/space/PEMS04_T96_H12_nomix1_s0-1).

Usage (from MAGE_MS/):
  .venv/bin/python run_pems04.py --gpus 3 --per_gpu 4    # screen on one A100 (achilleas), TF32 on by default
  .venv/bin/python run_pems04.py --set A --seeds 0,1,2,3,4 --gpus 0,3 --per_gpu 3   # part A: bounded lambda in MAGE
  .venv/bin/python run_pems04.py                         # screen: seed 0 only, GPUs 1,2,3, 3 jobs per GPU
  .venv/bin/python run_pems04.py --seeds 0,1,2,3,4       # full test
  .venv/bin/python run_pems04.py --gpus 0,1,2,3 --per_gpu 2
  nohup .venv/bin/python run_pems04.py > logs/run_pems04.out 2>&1 &     # detached
Then: .venv/bin/python summarize_pems04.py
Skips runs that already have a log, so it is safe to re-run with more seeds."""
import argparse
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--seeds', default='0')
p.add_argument('--gpus', default='1,2,3')
p.add_argument('--per_gpu', type=int, default=3)
p.add_argument('--tf32', type=int, default=1, help='TF32 matmuls (A100: ~20%% faster per epoch); 0 = plain fp32')
p.add_argument('--set', default='screen', choices=['screen', 'A', 'B'],
               help="screen: all 8 arms; A: MAGE with MAGE's vs bounded lambda (is bounding lambda a fix?); "
                    "B: shared/proposed/reversed with bounded lambda (does the schedule matter once lambda is bounded?)")
p.add_argument('--dry', action='store_true', help='print the job list and exit')
a = p.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.abspath('.venv/bin/python')
SLOTS = {int(g): a.per_gpu for g in a.gpus.split(',')}
jobs = []
for s in (int(x) for x in a.seeds.split(',')):
    if a.set == 'A':
        jobs += [(f'PEMS04_T96_mage_{lam}_s{s}', 'mage', lam, s) for lam in ('sigmoid', 'mage')]
        continue
    if a.set == 'B':
        jobs += [(f'PEMS04_T96_{arm}_sigmoid_s{s}', arm, 'sigmoid', s) for arm in ('reversed', 'proposed', 'shared')]
        continue
    for lam in ('mage', 'sigmoid'):
        for arm in ('reversed', 'proposed', 'shared'):          # slowest arms first
            jobs.append((f'PEMS04_T96_{arm}_{lam}_s{s}', arm, lam, s))
    jobs.append((f'PEMS04_T96_nomix_mage_s{s}', 'nomix', 'mage', s))
    jobs.append((f'PEMS04_T96_mage_mage_s{s}', 'mage', 'mage', s))   # MAGE itself, same GPU / precision
os.makedirs('logs/pems04', exist_ok=True); os.makedirs('results/pems04', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/pems04/{j[0]}.log')]
print(len(jobs), 'jobs:', ', '.join(j[0] for j in jobs), flush=True)
if a.dry:
    raise SystemExit

running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [q for q in running[g] if q.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            n, arm, lam, s = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen(
                [PY, 'train.py', '--dataset', 'PEMS04', '--seq_len', '96', '--arm', arm, '--lam', lam,
                 '--seed', str(s), '--tf32', str(a.tf32), '--data_dir', 'data_w288', '--out', f'results/pems04/{n}.json'],
                env=env, stdout=open(f'logs/pems04/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

"""2026-11-07 quick screen (1 seed, T = 96 -> 12, plain MAGE path = MAGEStream arm 'mage', A100 + TF32):
does fixing the graph collapse, and the rank hierarchy (general full rank + time rank r + sensor rank r), help?

  base     : MAGE as released (PEMS08 only; PEMS04 base is results/pems04/PEMS04_T96_mage_mage_s0)
  fix      : graph-collapse fix (spread-out E1/E2 init, no weight decay on them)
  fix+hier : + rank hierarchy, r = 32
  fix+hier+sig : + bounded lambda (our earlier fix)
Usage: .venv/bin/python run_hier.py [--gpus 0] [--per_gpu 7] [--seeds 0]; then summarize_hier.py"""
import argparse
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--gpus', default='0')
p.add_argument('--per_gpu', type=int, default=7)
p.add_argument('--seeds', default='0')
p.add_argument('--rank', type=int, default=32)
p.add_argument('--set', default='screen', choices=['screen', 'fair'],
               help="fair: hierarchy alone on released MAGE (no graph fix, MAGE's lambda) + PEMS08 MAGE baselines")
a = p.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.abspath('.venv/bin/python')
SLOTS = {int(g): a.per_gpu for g in a.gpus.split(',')}

# (dataset, lam, graph_fix, hier_rank)
variants = [('PEMS08', 'mage', 0, 0), ('PEMS08', 'mage', 1, 0), ('PEMS08', 'mage', 1, a.rank), ('PEMS08', 'sigmoid', 1, a.rank),
            ('PEMS04', 'mage', 1, 0), ('PEMS04', 'mage', 1, a.rank), ('PEMS04', 'sigmoid', 1, a.rank)]
if a.set == 'fair':   # one change only; PEMS04 MAGE baselines s0-4 already in results/pems04
    variants = [('PEMS08', 'mage', 0, a.rank), ('PEMS04', 'mage', 0, a.rank), ('PEMS08', 'mage', 0, 0)]
jobs = [(f'{d}_T96_mage_{lam}_f{f}_h{h}_s{s}', d, lam, f, h, s)
        for s in (int(x) for x in a.seeds.split(',')) for d, lam, f, h in variants]
os.makedirs('logs/hier', exist_ok=True); os.makedirs('results/hier', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/hier/{j[0]}.log')]
print(len(jobs), 'jobs:', ', '.join(j[0] for j in jobs), flush=True)
running = {g: [] for g in SLOTS}
while jobs or any(running.values()):
    for g in SLOTS:
        running[g] = [q for q in running[g] if q.poll() is None]
        while len(running[g]) < SLOTS[g] and jobs:
            n, d, lam, f, h, s = jobs.pop(0)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen(
                [PY, 'train.py', '--dataset', d, '--seq_len', '96', '--arm', 'mage', '--lam', lam,
                 '--graph_fix', str(f), '--hier_rank', str(h), '--seed', str(s), '--tf32', '1',
                 '--data_dir', 'data_w288', '--out', f'results/hier/{n}.json'],
                env=env, stdout=open(f'logs/hier/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

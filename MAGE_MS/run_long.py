"""Step 7 grid: long input windows. K jobs per GPU; skips runs that already have a log."""
import os, subprocess, time
GPUS, PER_GPU = [0, 2, 3], 4
PY = os.path.abspath('.venv/bin/python')
jobs = [(ds, T, enc, seed) for ds in ['PEMS08', 'PEMS04'] for seed in range(5)
        for T in [288, 96, 12] for enc in ['linear', 'm3', 'm4', 'm5']]
name = lambda j: f'{j[0]}_T{j[1]}_{j[2]}_s{j[3]}'
os.makedirs('logs/long', exist_ok=True); os.makedirs('results/long', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/long/{name(j)}.log')]
print(len(jobs), 'jobs', flush=True)
running = {g: [] for g in GPUS}
while jobs or any(running.values()):
    for g in GPUS:
        running[g] = [p for p in running[g] if p.poll() is None]
        while len(running[g]) < PER_GPU and jobs:
            j = jobs.pop(0); n = name(j)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            running[g].append(subprocess.Popen(
                [PY, 'train.py', '--dataset', j[0], '--seq_len', str(j[1]), '--encoder', j[2], '--seed', str(j[3]),
                 '--data_dir', 'data_w288', '--out', f'results/long/{n}.json'],
                env=env, stdout=open(f'logs/long/{n}.log', 'w'), stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

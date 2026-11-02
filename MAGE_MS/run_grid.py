"""Run the research_design.md grid, K jobs per GPU. Skips runs that already have a log
(finished or still running). PIDs passed on the command line as pid:gpu hold a slot until they exit."""
import os, subprocess, sys, time
GPUS, PER_GPU = [0, 2, 3], 4
PY = os.path.abspath('.venv/bin/python')
ROOT = os.path.dirname(os.path.abspath(__file__))
jobs = []
for ds, off in [('PEMS08', '8'), ('PEMS04', '4')]:
    for seed in range(5):
        for enc in ['linear', 'mslin', 'm3', 'm4', 'm5']:
            jobs.append(('ours', ds, enc, seed))
        if seed < 3:  # model 1 through the (minimally patched) official code, as the reproduction check
            jobs.append(('official', ds, off, seed))
def name(j):
    return f'{j[1]}_T12_{j[2]}_s{j[3]}' if j[0] == 'ours' else f'{j[1]}_official_s{j[3]}'
os.makedirs('logs', exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f'logs/{name(j)}.log')]
print(len(jobs), 'jobs', flush=True)
external = [tuple(map(int, a.split(':'))) for a in sys.argv[1:]]
def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False
running = {g: [] for g in GPUS}
for pid, g in external:
    running[g].append(pid)
while jobs or any(running.values()):
    for g in GPUS:
        running[g] = [p for p in running[g] if (p.poll() is None if hasattr(p, 'poll') else alive(p))]
        while len(running[g]) < PER_GPU and jobs:
            j = jobs.pop(0); n = name(j)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            if j[0] == 'ours':
                cmd, cwd = [PY, 'train.py', '--dataset', j[1], '--encoder', j[2], '--seed', str(j[3]),
                            '--out', f'results/runs/{n}.json'], ROOT
            else:
                cmd, cwd = [PY, 'experiments/mage/main.py', '--device', 'cuda:0', '--dataset', j[2],
                            '--years', '2019', '--model_name', 'mage', '--seed', str(j[3])], f'{ROOT}/official_repro'
            running[g].append(subprocess.Popen(cmd, cwd=cwd, env=env, stdout=open(f'logs/{n}.log', 'w'),
                                               stderr=subprocess.STDOUT))
            print(time.strftime('%m-%d %H:%M'), 'start', n, 'gpu', g, flush=True)
    time.sleep(15)
print('ALL DONE', flush=True)

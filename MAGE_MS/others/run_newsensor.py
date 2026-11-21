"""2026-11-16: how should a new sensor get its identity? U-Trans-style backbone (STAEformer + network mean), PCA or
network-relative identity, 20 % unseen sensors (rng 123), seed 0, PEMS04 / PEMS08.
Training (80 epochs), -> results/staeformer_<ds>_s0_mean_id<kind>_hold[_variant].json:
  rel            network-relative identity: log(sensor / network mean) per hour (id_features.day_features)
  rel_rs3, rel_rs7, pca_rs7   identity recomputed from 3 / 7 random days every step (no memorising; like a new sensor)
  rel_noise0.5   Gaussian noise, for comparison with the 2026-11-15 PCA noise run
New-sensor evaluation of every model (newsensor_eval.py, -> results/newsensor/<tag>.json): new sensors given 1 / 3 / 7
days or the full span; computed identity vs identity fine-tuned on those days (from computed / from average sensor).
Also evaluates the existing PCA, PCA noise 0.5 and PCA mix 0.5 checkpoints.

GPUs: any GPU on this machine with no other user's process above 1 GB (checked every 30 s, so GPUs that free up
later are picked up automatically); a job starts when measured free memory covers its estimate. Skips jobs that already have a log.
"""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.path.abspath('../.venv/bin/python')
MEM = {'PEMS04': 6500, 'PEMS08': 3800}                    # MB per training job (measured on 2026-11-15, + margin)
EVAL_MEM = {'PEMS04': 12000, 'PEMS08': 5000}         # identity fine-tuning backprops through the model
RESERVE = 1500                                             # MB left free per GPU
MAX_PER_GPU = 4                                            # spread jobs over GPUs (speed), not just memory
TRAIN = {'rel': ['--identity', 'rel'], 'rel_rs3': ['--identity', 'rel', '--id_resample', '3'],
         'rel_rs7': ['--identity', 'rel', '--id_resample', '7'], 'pca_rs7': ['--identity', 'pca', '--id_resample', '7'],
         'rel_noise0.5': ['--identity', 'rel', '--id_noise', '0.5']}
TAGS = {'rel': 'idrel_hold', 'rel_rs3': 'idrel_hold_rs3', 'rel_rs7': 'idrel_hold_rs7', 'pca_rs7': 'idpca_hold_rs7',
        'rel_noise0.5': 'idrel_hold_noise0.5'}
os.makedirs('logs/newsensor', exist_ok=True)
STAGE = int(os.environ.get('NEWSENSOR_STAGE', '1'))
# stage 2 (2026-11-16): baselines for identity fitting. Standard learned embedding with the network mean, and learned
# spatial attention (standard STAEformer) with the learned embedding or with PCA identity. Held-out sensors are hidden
# from attention during training (reimpl2.Attention).
STAGE2 = {('mean', 'learned'): 'mean_hold', ('learned', 'learned'): 'hold', ('learned', 'pca'): 'idpca_hold'}

jobs = []        # dict(log, argv, mem, needs) ; needs = result stem that must exist first (evaluation after training)
for ds in ('PEMS04', 'PEMS08'):
    for v, extra in TRAIN.items():
        jobs.append(dict(log=f'logs/newsensor/train_{ds}_{v}.log', mem=MEM[ds], needs=None,
                         argv=['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0', '--mixing',
                               'mean', '--epochs', '80', '--holdout', '0.2'] + extra))
for ds in ('PEMS08', 'PEMS04'):
    for t in ['idpca_hold', 'idpca_hold_noise0.5', 'idpca_hold_mix0.5'] + list(TAGS.values()):
        tag = f'staeformer_{ds}_s0_mean_{t}'
        jobs.append(dict(log=f'logs/newsensor/eval_{tag}.log', mem=EVAL_MEM[ds], needs=f'results/{tag}',
                         argv=['newsensor_eval.py', '--tag', tag]))
if STAGE == 2:
    jobs = []
    for ds in ('PEMS04', 'PEMS08'):
        for (mix, ident), t in STAGE2.items():
            jobs.append(dict(log=f'logs/newsensor/train2_{ds}_{mix}_{ident}.log', mem=MEM[ds] + (3000 if mix == 'learned' else 0),
                             needs=None, argv=['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0',
                                               '--mixing', mix, '--identity', ident, '--epochs', '80', '--holdout', '0.2']))
    for ds in ('PEMS08', 'PEMS04'):
        for t in STAGE2.values():
            tag = f'staeformer_{ds}_s0_{t}'
            jobs.append(dict(log=f'logs/newsensor/eval_{tag}.log', mem=EVAL_MEM[ds], needs=f'results/{tag}',
                             argv=['newsensor_eval.py', '--tag', tag]))
if STAGE == 3:
    # stage 3 (2026-11-16): spatial module = network mean (half the heads) + mean of each sensor's 10 most correlated
    # training sensors over the last 7 training days (other half); compare with network mean only (existing runs)
    jobs = []
    for ds in ('PEMS04', 'PEMS08'):
        for name, extra in [('none_hold', ['--identity', 'none', '--holdout', '0.2']),
                            ('pca_noise_hold', ['--identity', 'pca', '--id_noise', '0.5', '--holdout', '0.2']),
                            ('learned_all', [])]:
            jobs.append(dict(log=f'logs/newsensor/train3_{ds}_meansim_{name}.log', mem=MEM[ds] + 1000, needs=None,
                             argv=['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0',
                                   '--mixing', 'mean_sim', '--epochs', '80'] + extra))
if STAGE == 4:
    # stage 4 (2026-11-16): TRAINED history encoder identity (reimpl2 'enc') with cold-start episodes; ablation without
    # episodes; and the normal setting (all sensors known) to compare with U-Trans-style learned identity
    jobs = []
    for ds in ('PEMS04', 'PEMS08'):
        for name, extra, tag in [('enc_hold', ['--holdout', '0.2'], 'idenc_hold'),
                                 ('enc_noep_hold', ['--holdout', '0.2', '--enc_episodes', '0'], 'idenc_hold_ep0'),
                                 ('enc_all', [], None)]:
            jobs.append(dict(log=f'logs/newsensor/train4_{ds}_{name}.log', mem=MEM[ds], needs=None,
                             argv=['train_other2.py', '--model', 'staeformer', '--dataset', ds, '--seed', '0',
                                   '--mixing', 'mean', '--identity', 'enc', '--epochs', '80'] + extra))
            if tag:
                t_ = f'staeformer_{ds}_s0_mean_{tag}'
                jobs.append(dict(log=f'logs/newsensor/eval_{t_}.log', mem=EVAL_MEM[ds] // 2, needs=f'results/{t_}',
                                 argv=['newsensor_eval.py', '--tag', t_]))
jobs = [j for j in jobs if not os.path.exists(j['log'])]
print(len(jobs), 'jobs', flush=True)


ME = os.getuid()


def gpu_state():
    """Per GPU: total and used memory, and memory held by OTHER users (processes of this user count as ours, so
    jobs moved by hand or left by an earlier launcher do not block a GPU)."""
    q = lambda f: subprocess.run(['nvidia-smi', f'--query-{f}', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True).stdout.strip().splitlines()
    gpus = {}
    for line in q('gpu=index,uuid,memory.total,memory.used'):
        i, u, tot, used = [s.strip() for s in line.split(',')]
        gpus[u] = dict(index=int(i), total=int(tot), used=int(used), foreign=0, mine=set())
    for line in q('compute-apps=gpu_uuid,pid,used_memory'):
        u, pid, mem = [s.strip() for s in line.split(',')]
        try:
            uid = os.stat(f'/proc/{pid}').st_uid
        except OSError:
            continue
        if u in gpus and uid != ME:
            gpus[u]['foreign'] += int(mem) if mem.isdigit() else 0
        elif u in gpus:
            gpus[u]['mine'].add(int(pid))
    return {g['index']: g for g in gpus.values()}


running = []     # (Popen, gpu, mem, start time)
while jobs or running:
    running = [r for r in running if r[0].poll() is None]
    state = gpu_state()
    for g, s in sorted(state.items()):
        if s['foreign'] > 1000:
            continue                                        # another user is on this GPU
        # measured use, plus the full estimate for our jobs started < 4 min ago (not allocated yet)
        used = s['used'] + sum(m for _, gg, m, t0 in running if gg == g and time.time() - t0 < 240)
        for j in list(jobs):
            if len(s['mine'] | {r[0].pid for r in running if r[1] == g}) >= MAX_PER_GPU:
                break
            if j['needs'] and not (os.path.exists(j['needs'] + '.json') and os.path.exists(j['needs'] + '.pt')):
                continue
            if used + j['mem'] + RESERVE > s['total']:
                continue
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(g))
            pr = subprocess.Popen([PY] + j['argv'], env=env, stdout=open(j['log'], 'w'), stderr=subprocess.STDOUT)
            running.append((pr, g, j['mem'], time.time())); used += j['mem']; jobs.remove(j); s['mine'].add(pr.pid)
            print(time.strftime('%m-%d %H:%M'), 'start', os.path.basename(j['log']), 'gpu', g, flush=True)
    if jobs and not running and all(j['needs'] for j in jobs) \
            and not subprocess.run(['pgrep', '-u', str(ME), '-f', '^[^ ]*python[^ ]* train_other2.py'], capture_output=True).stdout.strip():
        print('remaining evaluations wait for training runs that are not running; stopping', flush=True)
        break
    time.sleep(30)
print('ALL DONE', flush=True)

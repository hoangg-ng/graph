"""Capacity-aware 4-GPU experiment scheduler for the MAGE scale-dependent mixing study.

Runs each worker in a separate process, pinned to one physical CUDA GPU.
Uses VRAM headroom, a per-GPU slot limit, logs, restartable checkpoints,
and a lock to avoid launching two copies of this scheduler in the same directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Optional

import train_eval

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Job:
    dataset: str
    arm: str
    seed: int

    @property
    def name(self):
        return f"{self.dataset}_{self.arm}_s{self.seed}"


def planned_jobs(profile):
    if profile == "smoke":
        return [Job("PEMS04", arm, 0) for arm in
                ("shared", "proposed", "reversed", "independent", "uniform", "mage", "mage_features")]
    if profile == "pilot":
        return [Job("PEMS04", arm, 0) for arm in ("shared", "proposed", "reversed")]
    main = [Job("PEMS04", arm, seed) for seed in range(5)
            for arm in ("shared", "proposed", "reversed")]
    main += [Job("PEMS04", arm, seed) for seed in range(5)
             for arm in ("independent", "uniform")]
    confirmation = [Job("PEMS08", arm, seed) for seed in range(3)
                    for arm in ("shared", "proposed", "reversed", "uniform")]
    if profile == "core":
        return main + confirmation
    if profile == "full":
        anchors = [Job("PEMS04", arm, seed) for seed in range(3)
                   for arm in ("mage", "mage_features")]
        return main + anchors + confirmation
    raise ValueError(f"Unknown profile: {profile}")


def gpu_status():
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, check=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f"Cannot query nvidia-smi: {err}") from err
    parsed = {}
    for line in proc.stdout.splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) < 5:
            continue
        try:
            idx = int(float(cols[0]))
            uuid = cols[1]
            total, free, util = [int(float(x)) for x in cols[2:5]]
        except ValueError:
            continue
        parsed[idx] = {"uuid": uuid, "total_mib": total, "free_mib": free, "util_pct": util}
    if not parsed:
        raise RuntimeError(f"Unexpected nvidia-smi output: {proc.stdout[:500]}")
    return parsed


def worker_arguments(a, job, out):
    opts = [
        "--dataset", job.dataset, "--arm", job.arm, "--seed", str(job.seed),
        "--seq-len", "96", "--horizon", "12", "--device", "cuda:0",
        "--epochs", str(a.epochs), "--patience", str(a.patience),
        "--lr", str(a.lr), "--wd", str(a.wd), "--bs", str(a.bs),
        "--lb", str(a.lb), "--lb-lr", str(a.lb_lr),
        "--dropout", str(a.dropout), "--model-dim", str(a.model_dim),
        "--experts", str(a.experts), "--topk", str(a.topk),
        "--graph-dim", str(a.graph_dim), "--pdm-width", str(a.pdm_width),
        "--pdm-depth", str(a.pdm_depth), "--moving-average", str(a.moving_average),
        "--data-dir", str(a.data_dir), "--threads", str(a.threads_per_job),
        "--tf32", str(a.tf32), "--save-every", str(a.save_every),
        "--out", str(out),
    ]
    if a.overwrite:
        opts.append("--overwrite")
    if a.reset_resume:
        opts.append("--reset-resume")
    return opts


def result_file(a, job):
    return a.results_dir / f"{job.name}.json"


def log_file(a, job):
    return a.logs_dir / f"{job.name}.log"


def expected_signature(a, job, code_sha, data_sha):
    opts = worker_arguments(a, job, result_file(a, job))
    parsed = train_eval.make_parser().parse_args(opts)
    return train_eval.run_identity(parsed, source_hash=code_sha, dataset_hash=data_sha[job.dataset])["signature"]


def check_result(a, job, expected):
    path = result_file(a, job)
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unreadable result: {path}: {error}") from error
    if result.get("completed") and result.get("signature") == expected:
        return True
    if not a.overwrite:
        raise RuntimeError(
            f"Existing result uses a different protocol or source version: {path}. "
            "Keep old results and set a new --results-dir; or use --overwrite explicitly."
        )
    return False


def event_log(logpath: Path, **data):
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with logpath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _prepare_datasets(args, required):
    for dataset in required:
        folder = train_eval.data_dir_path(str(args.data_dir)) / dataset
        names = ("his.npz", "idx_train.npy", "idx_val.npy", "idx_test.npy")
        if all((folder / name).is_file() for name in names):
            continue
        if not args.prepare:
            raise FileNotFoundError(
                f"Missing prepared {dataset} data in {folder}. "
                "Pass --prepare --raw-root <raw-data-folder> or run prep_proposal.py manually."
            )
        command = [
            sys.executable, str(ROOT / "prep_proposal.py"), "--datasets", dataset[-2:],
            "--out", str(args.data_dir), "--raw_root", str(args.raw_root),
        ]
        print(f"Preparing {dataset}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return {d: train_eval.data_digest(d, str(args.data_dir)) for d in required}


def _tests():
    tests = [sys.executable, "-m", "unittest", "test_proposal", "test_proposal_prep", "test_eval_tools"]
    print("Unit tests:", " ".join(tests), flush=True)
    subprocess.run(tests, cwd=ROOT, check=True)


def status(args):
    jobs = planned_jobs(args.profile)
    counts = Counter()
    for job in jobs:
        path = result_file(args, job)
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as f:
                    outcome = json.load(f)
                state = "completed" if outcome.get("completed") else "unverified"
            except (OSError, ValueError):
                state = "invalid-json"
        elif Path(str(path) + ".resume.pt").exists():
            state = "partial-resume"
        else:
            state = "not-started"
        counts[state] += 1
        print(f"{job.name:<32} {state}")
    print("Status summary:", dict(counts))


def _available_gpu(args, gpu, info, active, launch_free):
    if info["total_mib"] < args.gpu_job_mib + args.gpu_reserve_mib:
        return False
    assigned = [entry for entry in active.values() if entry["gpu"] == gpu]
    if len(assigned) >= args.slots_per_gpu:
        return False
    if not assigned and info["util_pct"] > args.max_idle_gpu_util:
        return False
    # Prevent launching a second process before nvidia-smi has observed the
    # first process's memory allocation. launch_free tracks a conservative
    # per-process budget from the GPU's initially observed free capacity.
    physical_free = info["free_mib"]
    accounting_free = launch_free[gpu] - args.gpu_job_mib * len(assigned)
    safe_free = min(physical_free, accounting_free)
    return safe_free >= max(args.min_free_mib, args.gpu_job_mib + args.gpu_reserve_mib)


def run_all(args):
    jobs = planned_jobs(args.profile)
    print(f"Profile {args.profile}: {len(jobs)} jobs ({len(args.gpus)} GPUs; "
          f"up to {args.slots_per_gpu} process/GPU; {args.threads_per_job} CPU threads/process)", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job.name} -> {result_file(args, job)}")
        return 0
    if args.status_only:
        status(args)
        return 0
    if args.preflight_tests:
        _tests()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    lockfile = args.logs_dir / ".eval_scheduler.lock"
    lock = lockfile.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another scheduler is already using {args.logs_dir}")

    data_sha = _prepare_datasets(args, sorted({job.dataset for job in jobs}))
    code_sha = train_eval.code_digest()
    expected = {job: expected_signature(args, job, code_sha, data_sha) for job in jobs}
    pending = [job for job in jobs if not check_result(args, job, expected[job])]
    done_count = len(jobs) - len(pending)
    print(f"Ready: {len(pending)} pending; {done_count} matching completed runs; source={code_sha[:12]}", flush=True)
    if not pending:
        print("All results already completed for this source/config/data signature.", flush=True)
        return 0

    stats = gpu_status()
    for idx in args.gpus:
        if idx not in stats:
            raise RuntimeError(f"Requested GPU {idx} is not reported by nvidia-smi: {stats}")
    launch_free = {gpu: stats[gpu]["free_mib"] for gpu in args.gpus}
    for idx in args.gpus:
        print(f"GPU {idx}: {stats[idx]}", flush=True)

    active = {}
    failed = []
    event_path = args.logs_dir / "events.jsonl"
    idle_start = time.monotonic()
    interrupted = False

    def shutdown(signum, _frame):
        nonlocal interrupted
        interrupted = True
        print(f"Received signal {signum}; terminating child process groups.", flush=True)
        for state in active.values():
            if state["proc"].poll() is None:
                try:
                    os.killpg(state["proc"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    previous_int = signal.signal(signal.SIGINT, shutdown)
    previous_term = signal.signal(signal.SIGTERM, shutdown)
    try:
        while pending or active:
            for job, state in list(active.items()):
                ret = state["proc"].poll()
                if ret is None:
                    continue
                state["stream"].close()
                del active[job]
                elapsed = time.monotonic() - state["start"]
                try:
                    finished = ret == 0 and check_result(args, job, expected[job])
                except RuntimeError:
                    finished = False
                if finished:
                    done_count += 1
                    print(f"DONE {job.name:<30} GPU={state['gpu']} time={elapsed/60:.1f}min "
                          f"({done_count}/{len(jobs)})", flush=True)
                    event_log(event_path, state="completed", job=job.name, gpu=state["gpu"],
                              elapsed_seconds=round(elapsed, 1))
                else:
                    failed.append(job)
                    print(f"FAILED {job.name} exit={ret}; inspect {log_file(args, job)}", flush=True)
                    event_log(event_path, state="failed", job=job.name, gpu=state["gpu"], exit_code=ret)
                idle_start = time.monotonic()
            if interrupted:
                break

            stats = gpu_status()
            # Keep an accounting floor while jobs are running; refresh it when
            # no worker is assigned, so released VRAM becomes available again.
            for gpu in args.gpus:
                if not any(item["gpu"] == gpu for item in active.values()):
                    launch_free[gpu] = stats[gpu]["free_mib"]
            launched_any = False
            while pending and not interrupted:
                selected = None
                # Prefer less-loaded GPUs to leave room for other lab users.
                for gpu in sorted(args.gpus, key=lambda g: (
                    sum(item["gpu"] == g for item in active.values()),
                    -stats[g]["free_mib"], g
                )):
                    if _available_gpu(args, gpu, stats[gpu], active, launch_free):
                        selected = gpu
                        break
                if selected is None:
                    break
                job = pending.pop(0)
                logpath = log_file(args, job)
                stream = logpath.open("a", encoding="utf-8", buffering=1)
                env = dict(os.environ)
                env.update({
                    "CUDA_VISIBLE_DEVICES": stats[selected]["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "OMP_NUM_THREADS": str(args.threads_per_job),
                    "MKL_NUM_THREADS": str(args.threads_per_job),
                    "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
                    "NUMEXPR_NUM_THREADS": str(args.threads_per_job),
                    "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                })
                command = [sys.executable, "-u", str(ROOT / "train_eval.py"),
                           *worker_arguments(args, job, result_file(args, job))]
                try:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT, env=env,
                                            start_new_session=True)
                except Exception:
                    stream.close()
                    pending.insert(0, job)
                    raise
                active[job] = {"proc": proc, "gpu": selected,
                               "stream": stream, "start": time.monotonic()}
                print(f"START {job.name:<30} GPU={selected} PID={proc.pid} "
                      f"log={logpath}", flush=True)
                event_log(event_path, state="started", job=job.name, gpu=selected, pid=proc.pid)
                launched_any = True
                idle_start = time.monotonic()
            if not active and pending and args.max_wait_minutes > 0:
                if time.monotonic() - idle_start > 60 * args.max_wait_minutes:
                    print("No GPU meets headroom requirements before max wait. "
                          "Unstarted jobs remain pending; rerun the same command later.", flush=True)
                    break
            if active or pending:
                time.sleep(args.poll_seconds)

    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        if active:
            for state in active.values():
                if state["proc"].poll() is None:
                    try:
                        os.killpg(state["proc"].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for state in active.values():
                try:
                    state["proc"].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(state["proc"].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                state["stream"].close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

    print(f"Completed/skip={done_count}, failed={len(failed)}, "
          f"unstarted={len(pending)}, interrupted={interrupted}", flush=True)
    if failed:
        for job in failed:
            print(f"  failed: {job.name} ({log_file(args, job)})", flush=True)
    return 0 if not failed and not pending and not interrupted else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke", "pilot", "core", "full"), default="full")
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--slots-per-gpu", type=int, default=1)
    p.add_argument("--threads-per-job", type=int, default=3)
    p.add_argument("--min-free-mib", type=int, default=12288,
                   help="Do not start a worker unless GPU has at least this much free VRAM")
    p.add_argument("--gpu-job-mib", type=int, default=9000,
                   help="Conservative estimated VRAM for one worker, especially with 2 slots/GPU")
    p.add_argument("--gpu-reserve-mib", type=int, default=3072)
    p.add_argument("--max-idle-gpu-util", type=int, default=50,
                   help="Avoid a GPU with this utilization if no worker from this scheduler uses it")
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--max-wait-minutes", type=int, default=30,
                   help="Exit if all requested GPUs remain unsuitable and no jobs are running; 0=unlimited")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lb", type=int, choices=(0, 1), default=1)
    p.add_argument("--lb-lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--model-dim", type=int, default=128)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--graph-dim", type=int, default=32)
    p.add_argument("--pdm-width", type=int, default=16)
    p.add_argument("--pdm-depth", type=int, default=2)
    p.add_argument("--moving-average", type=int, default=5)
    p.add_argument("--tf32", type=int, choices=(0, 1), default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--data-dir", default="data_proposal96")
    p.add_argument("--raw-root", default=str(ROOT / "raw" / "ASTGNN" / "data"))
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "eval_proposal96")
    p.add_argument("--logs-dir", type=Path, default=ROOT / "logs" / "eval_proposal96")
    p.add_argument("--recipe", type=Path,
                   help="selected_recipe.json written by tune_shared.py; freezes Shared-arm selected hyperparameters")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preflight-tests", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reset-resume", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.slots_per_gpu < 1 or args.threads_per_job < 1 or args.poll_seconds < 1:
        p.error("slots-per-gpu, threads-per-job and poll-seconds must be positive")
    if args.save_every < 1 or args.max_wait_minutes < 0:
        p.error("save-every must be >=1 and max-wait-minutes >=0")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        p.error("List each selected physical GPU exactly once")
    if args.recipe is not None:
        if not args.recipe.is_file():
            p.error(f"Recipe not found: {args.recipe}")
        with args.recipe.open(encoding="utf-8") as f:
            recipe = json.load(f)
        code_sha = train_eval.code_digest()
        data_sha = train_eval.data_digest("PEMS04", args.data_dir)
        if recipe.get("code_digest") != code_sha or recipe.get("data_digest") != data_sha:
            p.error("Recipe was tuned with different code or PEMS04 data; retune before use")
        for name in ("lr", "wd", "dropout"):
            setattr(args, name, recipe["selected"][name])
        print("Frozen Shared-arm recipe:", {k:getattr(args,k) for k in ("lr","wd","dropout")}, flush=True)
    sys.exit(run_all(args))


if __name__ == "__main__":
    main()

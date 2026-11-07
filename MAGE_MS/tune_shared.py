"""Freeze a small hyperparameter search on PEMS04 Shared before the final 43 runs.

Ten short one-seed experiments, validation-only (no test metric computed).
One independent process per selected GPU; each GPU is checked for free memory.
The winning recipe is chosen by minimum validation MAE only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import run_eval
import train_eval

ROOT = Path(__file__).resolve().parent
# Fixed a priori: 10 short trials on one seed, no test-set selection.
TRIALS = (
    (0.005, 1e-4, 0.1), (0.005, 1e-4, 0.3),
    (0.010, 1e-4, 0.1), (0.010, 1e-4, 0.3),
    (0.020, 1e-4, 0.1), (0.020, 1e-4, 0.3),
    (0.010, 1e-3, 0.1), (0.010, 1e-3, 0.3),
    (0.020, 1e-3, 0.1), (0.020, 1e-3, 0.3),
)


def trial_path(args, trial):
    return args.results_dir / f"shared_tune_{trial:02d}.json"


def trial_args(args, trial):
    lr, wd, dropout = TRIALS[trial]
    result = trial_path(args, trial)
    return [
        "--dataset", "PEMS04", "--arm", "shared", "--seed", "0",
        "--seq-len", "96", "--horizon", "12", "--device", "cuda:0",
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--lr", str(lr), "--wd", str(wd), "--dropout", str(dropout),
        "--bs", str(args.bs), "--lb", "1", "--lb-lr", "0.001",
        "--model-dim", "128", "--experts", "16", "--topk", "4",
        "--graph-dim", "32", "--pdm-width", "16", "--pdm-depth", "2",
        "--moving-average", "5", "--tf32", str(args.tf32),
        "--threads", str(args.threads_per_job), "--save-every", "10",
        "--data-dir", str(args.data_dir), "--out", str(result),
        "--validation-only",
    ]


def collect(args, source_sha, data_sha):
    validated, errors = [], []
    for idx in range(len(TRIALS)):
        path = trial_path(args, idx)
        opts = train_eval.make_parser().parse_args(trial_args(args, idx))
        expected = train_eval.run_identity(opts, source_sha, data_sha)["signature"]
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            outcome = json.load(f)
        if (not outcome.get("completed") or outcome.get("signature") != expected
            or outcome.get("protocol") != "96to12_v1_validation_only"):
            errors.append(f"Incompatible or non-validation result: {path}")
            continue
        validated.append({
            "trial": idx, "lr": TRIALS[idx][0], "wd": TRIALS[idx][1],
            "dropout": TRIALS[idx][2], "val_mae": outcome["best_val_mae"],
            "epochs_run": outcome["epochs_run"],
        })
    return validated, errors


def run(args):
    if args.dry_run:
        for idx, (lr, wd, dropout) in enumerate(TRIALS):
            print(f"trial={idx:02d} lr={lr} wd={wd} dropout={dropout}")
        return 0
    data_path = train_eval.data_dir_path(args.data_dir) / "PEMS04"
    if not (data_path / "his.npz").exists():
        raise FileNotFoundError(f"Prepare PEMS04 first: {data_path}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.logs_dir / ".tune_shared.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as ex:
        raise RuntimeError("Another tune_shared process is using these directories") from ex
    try:
        code_sha = train_eval.code_digest()
        data_sha = train_eval.data_digest("PEMS04", args.data_dir)
        validated, errors = collect(args, code_sha, data_sha)
        if errors:
            raise RuntimeError("\n".join(errors))
        completed = {row["trial"] for row in validated}
        pending = queue.Queue()
        for idx in range(len(TRIALS)):
            if idx not in completed:
                pending.put(idx)
        print(f"{len(completed)}/10 validation-only trials complete; {pending.qsize()} pending", flush=True)
        if args.select_existing and pending.qsize():
            raise RuntimeError(f"Cannot select recipe; {pending.qsize()} validation trials are missing")
        if not args.select_existing and not pending.empty():
            stats = run_eval.gpu_status()
            missing_gpu = [idx for idx in args.gpus if idx not in stats]
            if missing_gpu:
                raise RuntimeError(f"GPUs missing: {missing_gpu}")
            stop = threading.Event()
            active = {}
            active_lock = threading.Lock()
            failures = []

            def worker(gpu):
                idle_start = time.monotonic()
                while not stop.is_set():
                    try:
                        idx = pending.get_nowait()
                    except queue.Empty:
                        return
                    gpu_info = None
                    while not stop.is_set():
                        gpu_info = run_eval.gpu_status()[gpu]
                        if (gpu_info["free_mib"] >= args.min_free_mib
                            and gpu_info["util_pct"] <= args.max_idle_gpu_util):
                            break
                        if (args.max_wait_minutes > 0 and
                            time.monotonic() - idle_start >= args.max_wait_minutes * 60):
                            failures.append((idx, gpu, "No suitable GPU before timeout"))
                            pending.task_done()
                            return
                        time.sleep(15)
                    if stop.is_set():
                        return
                    idle_start = time.monotonic()
                    env = dict(os.environ)
                    env.update({
                        "CUDA_VISIBLE_DEVICES": gpu_info["uuid"],
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
                        "OMP_NUM_THREADS": str(args.threads_per_job),
                        "MKL_NUM_THREADS": str(args.threads_per_job),
                        "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
                    })
                    logfile = args.logs_dir / f"shared_tune_{idx:02d}.log"
                    command = [sys.executable, "-u", str(ROOT / "train_eval.py"), *trial_args(args, idx)]
                    print(f"TUNE START trial={idx:02d} gpu={gpu} "
                          f"lr/wd/dropout={TRIALS[idx]}", flush=True)
                    with logfile.open("a", encoding="utf-8") as f:
                        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                        with active_lock:
                            active[gpu] = proc
                        code = proc.wait()
                        with active_lock:
                            active.pop(gpu, None)
                    if code != 0:
                        failures.append((idx, gpu, f"exit {code}; {logfile}"))
                        print(f"TUNE FAIL trial={idx:02d}: {failures[-1][2]}", flush=True)
                    else:
                        print(f"TUNE DONE trial={idx:02d}", flush=True)
                    pending.task_done()

            try:
                with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
                    tasks = [executor.submit(worker, gpu) for gpu in args.gpus]
                    for task in as_completed(tasks):
                        task.result()
            except KeyboardInterrupt:
                stop.set()
                with active_lock:
                    for process in active.values():
                        if process.poll() is None:
                            process.terminate()
                raise
            if failures:
                raise RuntimeError("Failed tuning trials: " + str(failures))

        validated, errors = collect(args, code_sha, data_sha)
        if errors or len(validated) != len(TRIALS):
            raise RuntimeError(f"Tuning incomplete: {len(validated)}/10. {errors}")
        ordered = sorted(validated, key=lambda item: (item["val_mae"], item["trial"]))
        selected = ordered[0]
        recipe = {
            "selection_method": "best PEMS04 Shared validation MAE, seed 0, ten trials, no test data",
            "code_digest": code_sha, "data_digest": data_sha,
            "selected": {k: selected[k] for k in ("lr", "wd", "dropout")},
            "winner": selected, "ranking": ordered,
        }
        out = args.results_dir / "selected_recipe.json"
        out.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        print("Selected:", recipe["selected"], "validation MAE:", selected["val_mae"])
        print("Recipe:", out)
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--threads-per-job", type=int, default=3)
    p.add_argument("--min-free-mib", type=int, default=12288)
    p.add_argument("--max-idle-gpu-util", type=int, default=50)
    p.add_argument("--max-wait-minutes", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--tf32", type=int, choices=(0, 1), default=0)
    p.add_argument("--data-dir", default="data_proposal96")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "tune_shared")
    p.add_argument("--logs-dir", type=Path, default=ROOT / "logs" / "tune_shared")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--select-existing", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        p.error("Supply distinct GPU indexes")
    if args.epochs < 1 or args.patience < 1:
        p.error("epochs and patience must be positive")
    sys.exit(run(args))


if __name__ == "__main__":
    main()

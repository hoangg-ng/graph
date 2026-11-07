"""Safely schedule the additional comparison runs in isolated output folders.

Dry-run is the default. Pass --execute to launch training workers.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_eval
import train_eval

ROOT = Path(__file__).resolve().parent
FOLLOWUP_ARMS = ("grouped", "grouped_no_pdm", "grouped_no_spatial", "grouped_uniform")
DATASETS = ("PEMS04", "PEMS08")


def plan_jobs(datasets=DATASETS, arms=FOLLOWUP_ARMS, seeds=range(5)):
    return [
        run_eval.Job(dataset, arm, seed)
        for dataset in datasets
        for arm in arms
        for seed in seeds
    ]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_recipe(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Selected recipe not found: {path}")
    raw = path.read_bytes()
    recipe = json.loads(raw.decode("utf-8"))
    selected = recipe.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"Recipe has no selected hyperparameters: {path}")
    expected = {"lr": 0.005, "wd": 0.0001, "dropout": 0.3}
    for key, expected_value in expected.items():
        if key not in selected or abs(float(selected[key]) - expected_value) > 1e-12:
            raise ValueError(
                f"Recipe {path} must select {key}={expected_value}; got {selected.get(key)!r}"
            )
    return recipe, hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _result_path(args, job):
    return args.results_dir / f"{job.name}.json"


def _log_path(args, job):
    return args.logs_dir / f"{job.name}.log"


def _worker_args(args, job, out):
    options = run_eval.worker_arguments(args, job, out)
    options += [
        "--recipe-source", str(args.recipe_path),
        "--recipe-digest", args.recipe_digest,
        "--recipe-source-code-digest", args.recipe_source_code_digest or "",
    ]
    if args.validation_only:
        options.append("--validation-only")
    return options


def _prepare(args, jobs):
    data_digests = run_eval._prepare_datasets(
        args, sorted({job.dataset for job in jobs})
    )
    return data_digests


def _manifest(args, code_digest, data_digests, jobs):
    return {
        "study": "ablation96",
        "validation_only": bool(args.validation_only),
        "recipe_source": str(args.recipe_path),
        "recipe_digest": args.recipe_digest,
        "recipe_source_code_digest": args.recipe_source_code_digest,
        "recipe_source_data_digest": args.recipe_source_data_digest,
        "selected_hyperparameters": {
            "lr": args.lr,
            "wd": args.wd,
            "dropout": args.dropout,
        },
        "new_code_digest": code_digest,
        "data_digests": data_digests,
        "workers_record_recipe_metadata": True,
        "execution_requested": bool(args.execute),
        "jobs": [job.name for job in jobs],
    }


def run(args):
    args.results_dir = _resolve(args.results_dir)
    args.logs_dir = _resolve(args.logs_dir)
    args.data_dir = str(_resolve(Path(args.data_dir)))
    args.recipe_path = _resolve(args.recipe)
    recipe, args.recipe_digest = _read_recipe(args.recipe_path)
    args.lr = float(recipe["selected"]["lr"])
    args.wd = float(recipe["selected"]["wd"])
    args.dropout = float(recipe["selected"]["dropout"])
    args.recipe_source_code_digest = recipe.get("code_digest")
    args.recipe_source_data_digest = recipe.get("data_digest")
    jobs = plan_jobs(args.datasets, args.arms, args.seeds)
    code_digest = train_eval.code_digest()

    if args.dry_run:
        print(
            f"DRY RUN: {len(jobs)} jobs; no GPU or training worker will be started.\n"
            f"Recipe: {args.recipe_path} (sha256={args.recipe_digest[:12]})\n"
            f"Recipe source code: {args.recipe_source_code_digest}\n"
            f"Current worker code: {code_digest}\n"
            f"Hyperparameters: lr={args.lr}, wd={args.wd}, dropout={args.dropout}\n"
            f"Results: {args.results_dir}\nLogs: {args.logs_dir}",
            flush=True,
        )
        for job in jobs:
            print(f"{job.name} -> {_result_path(args, job)}")
        return 0

    if args.recipe_source_data_digest:
        current_pems04_digest = train_eval.data_digest("PEMS04", args.data_dir)
        if current_pems04_digest != args.recipe_source_data_digest:
            raise RuntimeError(
                "Selected recipe was tuned on different PEMS04 data; refusing follow-up run."
            )

    data_digests = _prepare(args, jobs)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "ablation_manifest.json"
    new_manifest = _manifest(args, code_digest, data_digests, jobs)

    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            previous = json.load(f)

        identity_keys = (
            "study",
            "validation_only",
            "recipe_digest",
            "recipe_source_code_digest",
            "selected_hyperparameters",
            "new_code_digest",
            "data_digests",
            "jobs",
        )

        mismatched = [
            key for key in identity_keys
            if previous.get(key) != new_manifest.get(key)
        ]

        if mismatched:
            raise RuntimeError(
                f"Existing manifest is incompatible: {mismatched}. "
                "Use a separate --results-dir."
            )

        print(f"Reusing existing manifest: {manifest_path}", flush=True)
    else:
        _atomic_json(manifest_path, new_manifest)

    expected = {
        job: train_eval.run_identity(
            train_eval.make_parser().parse_args(
                _worker_args(args, job, _result_path(args, job))
            ),
            source_hash=code_digest,
            dataset_hash=data_digests[job.dataset],
        )["signature"]
        for job in jobs
    }
    pending = []
    done = 0
    for job in jobs:
        if run_eval.check_result(args, job, expected[job]):
            done += 1
        else:
            pending.append(job)
    print(
        f"Ready: {len(pending)} pending, {done} matching completed; "
        f"current code digest={code_digest}",
        flush=True,
    )
    if not pending:
        print("All requested follow-up jobs are already complete.")
        return 0

    gpu_info = run_eval.gpu_status()
    for index in args.gpus:
        if index not in gpu_info:
            raise RuntimeError(f"Requested GPU {index} is not available: {gpu_info}")
    launch_free = {index: gpu_info[index]["free_mib"] for index in args.gpus}
    active = {}
    failed = []
    lock_path = args.logs_dir / ".followup_scheduler.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"Another follow-up scheduler is using {args.logs_dir}") from error

    interrupted = False

    def shutdown(signum, _frame):
        nonlocal interrupted
        interrupted = True
        for state in active.values():
            if state["proc"].poll() is None:
                try:
                    os.killpg(state["proc"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    previous_int = signal.signal(signal.SIGINT, shutdown)
    previous_term = signal.signal(signal.SIGTERM, shutdown)
    idle_start = time.monotonic()
    try:
        while pending or active:
            for job, state in list(active.items()):
                code = state["proc"].poll()
                if code is None:
                    continue
                state["stream"].close()
                del active[job]
                if code == 0 and run_eval.check_result(args, job, expected[job]):
                    done += 1
                    print(f"DONE {job.name} on GPU {state['gpu']}", flush=True)
                    idle_start = time.monotonic()
                else:
                    failed.append(job)
                    print(f"FAILED {job.name}; inspect {_log_path(args, job)}", flush=True)
            if interrupted:
                break

            gpu_info = run_eval.gpu_status()
            for index in args.gpus:
                if index not in gpu_info:
                    raise RuntimeError(f"Requested GPU {index} disappeared: {gpu_info}")
                if not any(item["gpu"] == index for item in active.values()):
                    launch_free[index] = gpu_info[index]["free_mib"]
            while pending and not interrupted:
                selected = next(
                    (
                        index for index in sorted(
                            args.gpus,
                            key=lambda gpu: (
                                sum(item["gpu"] == gpu for item in active.values()),
                                -gpu_info[gpu]["free_mib"],
                                gpu,
                            ),
                        )
                        if run_eval._available_gpu(
                            args, index, gpu_info[index], active, launch_free
                        )
                    ),
                    None,
                )
                if selected is None:
                    break
                job = pending.pop(0)
                stream = _log_path(args, job).open("a", encoding="utf-8", buffering=1)
                env = dict(os.environ)
                env.update({
                    "CUDA_VISIBLE_DEVICES": gpu_info[selected]["uuid"],
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "OMP_NUM_THREADS": str(args.threads_per_job),
                    "MKL_NUM_THREADS": str(args.threads_per_job),
                    "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
                    "NUMEXPR_NUM_THREADS": str(args.threads_per_job),
                    "PYTHONUNBUFFERED": "1",
                })
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "train_eval.py"),
                    *_worker_args(args, job, _result_path(args, job)),
                ]
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                    )
                except Exception:
                    stream.close()
                    pending.insert(0, job)
                    raise
                active[job] = {
                    "proc": process,
                    "gpu": selected,
                    "stream": stream,
                    "start": time.monotonic(),
                }
                print(f"START {job.name} GPU={selected} PID={process.pid}", flush=True)
                idle_start = time.monotonic()
            if not active and pending and args.max_wait_minutes > 0:
                if time.monotonic() - idle_start > 60 * args.max_wait_minutes:
                    print(
                        f"GPU headroom unavailable for {args.max_wait_minutes} minutes; "
                        f"{len(pending)} jobs remain pending.",
                        flush=True,
                    )
                    break
            if pending or active:
                time.sleep(args.poll_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
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
    print(
        f"Completed/skip={done}, failed={len(failed)}, "
        f"unstarted={len(pending)}, interrupted={interrupted}",
        flush=True,
    )
    return 0 if not failed and not pending and not interrupted else 1


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="execute", action="store_false")
    mode.add_argument("--execute", dest="execute", action="store_true")
    parser.set_defaults(execute=False)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--arms", nargs="+", choices=FOLLOWUP_ARMS, default=list(FOLLOWUP_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=3)
    parser.add_argument("--min-free-mib", type=int, default=12288)
    parser.add_argument("--gpu-job-mib", type=int, default=9000)
    parser.add_argument("--gpu-reserve-mib", type=int, default=3072)
    parser.add_argument("--max-idle-gpu-util", type=int, default=50)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--max-wait-minutes", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lb", type=int, choices=(0, 1), default=1)
    parser.add_argument("--lb-lr", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--graph-dim", type=int, default=32)
    parser.add_argument("--pdm-width", type=int, default=16)
    parser.add_argument("--pdm-depth", type=int, default=2)
    parser.add_argument("--moving-average", type=int, default=5)
    parser.add_argument("--tf32", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--data-dir", default="data_proposal96")
    parser.add_argument("--raw-root", default=str(ROOT / "raw/ASTGNN/data"))
    parser.add_argument("--recipe", type=Path, default=ROOT / "results/tune_shared/selected_recipe.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/ablation_val96")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs/ablation_val96")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-resume", action="store_true")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if (
        args.slots_per_gpu < 1
        or args.threads_per_job < 1
        or args.poll_seconds < 1
        or args.epochs < 1
        or args.patience < 1
        or args.max_wait_minutes < 0
    ):
        parser.error("scheduler counts, epochs, and patience must be positive")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("List each selected physical GPU exactly once")
    args.dry_run = not args.execute
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

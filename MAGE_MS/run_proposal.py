"""Run the proposal's 43 local experiments sequentially (PEMS04 before PEMS08)."""

import argparse
import os
import shlex
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_ARMS = ("shared", "proposed", "reversed", "independent", "uniform")


def experiment_jobs():
    jobs = [
        ("PEMS04", arm, seed)
        for seed in range(5)
        for arm in CORE_ARMS
    ]
    jobs.extend(
        ("PEMS04", arm, seed)
        for seed in range(3)
        for arm in ("mage", "mage_features")
    )
    jobs.extend(
        ("PEMS08", arm, seed)
        for seed in range(3)
        for arm in ("shared", "proposed", "reversed", "uniform")
    )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--data_dir", default="data_proposal96")
    parser.add_argument("--results_dir", default="results/proposal96")
    parser.add_argument("--logs_dir", default="logs/proposal96")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    args = parser.parse_args()
    results_dir = args.results_dir
    logs_dir = args.logs_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(ROOT, results_dir)
    if not os.path.isabs(logs_dir):
        logs_dir = os.path.join(ROOT, logs_dir)
    if not args.dry_run:
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    jobs = experiment_jobs()
    print(f"{len(jobs)} planned experiments; PEMS04 runs before PEMS08")
    for dataset, arm, seed in jobs:
        name = f"{dataset}_{arm}_s{seed}"
        result_path = os.path.join(results_dir, f"{name}.json")
        command = [
            sys.executable,
            os.path.join(ROOT, "train_proposal.py"),
            "--dataset",
            dataset,
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--data_dir",
            args.data_dir,
            "--out",
            result_path,
        ]
        if os.path.isfile(result_path):
            print(f"skip existing result: {name}")
            continue
        if args.dry_run:
            print(shlex.join(command))
            continue
        log_path = os.path.join(logs_dir, f"{name}.log")
        print(f"start {name}; log={log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"done {name}", flush=True)
    if args.dry_run:
        print("External TimeMixer/STID runs and original 12->12 MAGE reproduction are not included.")


if __name__ == "__main__":
    main()

"""Grouped/Slim comparison using the existing resumable GPU scheduler.

Dry-run is the default. Pilot: --validation-only --seeds 0 1 (16 jobs).
Formal: seeds 0..4 (40 jobs), including a new Grouped baseline in this code version.
"""
from pathlib import Path

import run_ablation_eval as scheduler
import run_eval

ROOT = Path(__file__).resolve().parent
ARMS = ("grouped", "grouped_slim", "slim_then_grouped", "grouped_uniform")
OPTIONAL_ARMS = ARMS + ("grouped_no_spatial",)


def plan_jobs(datasets=("PEMS04", "PEMS08"), arms=ARMS, seeds=range(5)):
    return [run_eval.Job(dataset, arm, seed) for dataset in datasets for arm in arms for seed in seeds]


def make_parser():
    parser = scheduler.make_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "arms":
            action.choices = OPTIONAL_ARMS
    parser.set_defaults(arms=list(ARMS), results_dir=None, logs_dir=None)
    return parser


def run(args):
    suffix = "val96" if args.validation_only else "test96"
    args.results_dir = args.results_dir or ROOT / f"results/unified_slim_{suffix}"
    args.logs_dir = args.logs_dir or ROOT / f"logs/unified_slim_{suffix}"
    args.dry_run = not args.execute
    args.study = "unified_slim96"
    return scheduler.run(args, jobs=plan_jobs(args.datasets, args.arms, args.seeds))


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.prepare:
        parser.error("Use existing data_proposal96 inputs; preprocessing is not changed by this study")
    if (args.slots_per_gpu < 1 or args.threads_per_job < 1 or args.poll_seconds < 1
            or args.epochs < 1 or args.patience < 1 or args.max_wait_minutes < 0):
        parser.error("Counts, epochs and patience must be positive; max wait cannot be negative")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("List each GPU exactly once")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.arms)) != len(args.arms):
        parser.error("Datasets and arms must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in range(5) for seed in args.seeds):
        parser.error("Select each seed 0..4 once")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

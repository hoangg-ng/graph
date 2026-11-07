"""Run matched MAGE-vs-Proposed evaluations through the existing four-GPU scheduler.

Install next to run_eval.py and train_eval.py. No existing source file is modified.
Each selected dataset: six arms x five identical random seeds = 30 runs.
Existing matching, completed result JSONs are reused by run_eval.py.
"""
from __future__ import annotations

import run_eval

ARMS = ("mage", "shared", "proposed", "reversed", "independent", "uniform")
PROFILES = {
    "matched04": ("PEMS04",),
    "matched08": ("PEMS08",),
    "matched_both": ("PEMS04", "PEMS08"),
}


def matched_jobs(profile):
    """Return the exact deterministic five-seed matrix for this study."""
    datasets = PROFILES[profile]
    return [
        run_eval.Job(dataset, arm, seed)
        for dataset in datasets
        for seed in range(5)
        for arm in ARMS
    ]


def main():
    original_parser = run_eval.parser
    original_planned_jobs = run_eval.planned_jobs

    def extended_parser():
        p = original_parser()
        profile_option = next(a for a in p._actions if a.dest == "profile")
        profile_option.choices = tuple(profile_option.choices) + tuple(PROFILES)
        profile_option.default = "matched_both"
        return p

    def extended_jobs(profile):
        return matched_jobs(profile) if profile in PROFILES else original_planned_jobs(profile)

    # Reuse run_eval's GPU memory checks, process scheduler, configuration
    # signatures, atomic checkpoints, locking, retry and recipe validation.
    run_eval.parser = extended_parser
    run_eval.planned_jobs = extended_jobs
    run_eval.main()


if __name__ == "__main__":
    main()

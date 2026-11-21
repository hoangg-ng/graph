# Archive: earlier MAGE study (multi-scale proposal, graph collapse, diagnostics)

Code and results from the first phase of this project (2026-10-18 to 2026-11-07), moved here on 2026-11-20 so the top
level holds only the unseen-sensor work. Each item is described in `../../experiment_log.md`, whose entries still name
the original paths (`MAGE_MS/<script>`, `MAGE_MS/results/<dir>`); read them as `MAGE_MS/archive/<script>` and
`MAGE_MS/archive/results/<dir>`.

| Here | Log entry |
|---|---|
| `official_repro/` | Step 1, MAGE reproduction |
| `cost.py`, `results/cost_*.json` | Step 3, cost |
| `post.py`, `analyze.py`, `results/post*` | Steps 4-5, routing and regimes |
| `diag*.py`, `results/diag*.json` | Error anatomy; which MAGE parameters do useful work (`diag_useful.py`, `results/useful/`) |
| `results/runs/`, `results/long/` | Step 2 multi-scale encoder, Step 7 long input windows |
| `precheck_hops.py`, `results/precheck_hops*.json` | Pre-check for the proposal |
| `results/step8/`, `results/pems04/`, `results/space/`, `results/rerun/` | Step 8, original proposal and review follow-up |
| `results/hier/` | Graph-collapse fix + rank hierarchy |
| `run_ab.py`, `results/ab/` | Slim MAGE + structure, methods A / B |
| `others/run_others2.sh` | unreferenced launcher variant of `others/run_others.sh` |

The scripts expect to be run from `MAGE_MS/` (they import `models.py` / `streams.py` and write to `results/`); move a
script back up one level before re-running it.

# Proposal: Scale-Dependent Spatial Mixing for Traffic Forecasting

## Research Question
Does stronger spatial mixing at coarser temporal resolutions improve traffic forecasting compared with uniform mixing, reversed mixing, or independent spatial operators?

## Motivation
MAGE (NeurIPS 2025) learns adaptive spatial relationships, while TimeMixer (ICLR 2024) captures temporal patterns at multiple resolutions. Prior work has combined temporal and spatial mixing or learned separate graphs per scale, but the benefit of explicitly matching **spatial mixing strength to temporal resolution** remains unclear.

We hypothesize that fine-grained fluctuations benefit from weaker mixing, whereas slower trends may benefit from stronger network-wide mixing. This is a testable hypothesis, not an assumption.

## Proposed Method
- **Task:** Predict the next 12 traffic-flow steps (1 hour) from 96 historical steps (8 hours).
- **Temporal encoder:** Average-pool the input into four resolutions (5, 10, 20, and 40 minutes), then use TimeMixer-style multiscale encoding.
- **Spatial encoder:** Use one shared MAGE-style learned spatial operator across all scales, applying different powers to control mixing strength.
- **Prediction:** Combine per-scale prediction heads to produce the final forecast.

| Variant | Mixing schedule (finest → coarsest) | Purpose |
|---|---|---|
| Shared | (1, 1, 1, 1) | Same mixing at every scale |
| **Proposed** | **(1, 2, 4, 8)** | Stronger mixing at coarser scales |
| Reversed | (8, 4, 2, 1) | Test the opposite hypothesis with equal compute |
| Independent | Separate operator per scale | Test flexibility versus parameter efficiency |
| Uniform | Global mean mixing | Test whether learned spatial structure is necessary |

## Experimental Design
- **Datasets:** PEMS04 (primary) and PEMS08 (confirmation), using traffic flow.
- **Protocol:** Chronological 60/20/20 split; identical 96 → 12 forecasting setup for controlled comparisons.
- **Reference baselines:** MAGE, TimeMixer, STID, and unchanged MAGE with engineered temporal features.
- **Metrics:** MAE (primary), RMSE, MAPE, parameter count, runtime, and memory; report masked and unmasked errors where applicable.
- **Reliability:** Five paired random seeds for core variants on PEMS04; three seeds for confirmation on PEMS08.
- **Key comparison:** Proposed versus Reversed isolates the *direction* of scale-dependent mixing because both have equal parameters and propagation counts.

## Expected Contribution
A controlled empirical test of whether temporal resolution should determine the strength of learned spatial mixing. The approach reuses one spatial operator without adding scale-specific graph parameters. Even if the proposed schedule does not win, the comparisons can establish whether scale-specific mixing matters, whether its direction matters, and whether simpler alternatives suffice.

## Timeline (8 Weeks)
- **Weeks 1–2:** Data pipeline and MAGE reproduction.
- **Weeks 3–4:** Multiscale model, controlled variants, and implementation checks.
- **Weeks 5–6:** Main experiments on PEMS04; confirmation on PEMS08.
- **Weeks 7–8:** Diagnostics, statistical comparisons, and final report.

## Key References
MAGE (NeurIPS 2025); TimeMixer (ICLR 2024); MobiMixer (IEEE TMC 2025); MSGNet (AAAI 2024); STID (CIKM 2022).

## Additive implementation

The proposal-specific implementation lives in `MAGE_MS/proposal_model.py`,
`MAGE_MS/prep_proposal.py`, `MAGE_MS/train_proposal.py`, and `MAGE_MS/run_proposal.py`.
It does not replace the repository's existing MAGE code or official sources. The
PDM is a TimeMixer-style adaptation; it is not claimed to be a verbatim copy of
the official TimeMixer implementation. Suggested PDM width/depth and
moving-average values are configurable starting points.

Run the focused checks from `MAGE_MS`:

```powershell
python -m unittest test_proposal test_proposal_prep
python prep_proposal.py --datasets 04 08 --out data_proposal96
python train_proposal.py --dataset PEMS04 --arm proposed --data_dir data_proposal96 --device cpu --epochs 1 --patience 1 --out results/proposal_smoke/PEMS04_proposed_s0.json
python run_proposal.py --device cpu --dry_run
```

The experiment runner covers the five core variants and two MAGE baselines. It
does not implement external TimeMixer/STID training or the original 12-to-12
MAGE reproduction. A one-epoch smoke run is only an execution check, not a
performance result. Verify the raw data, preprocessing, split consistency,
power-distance diagnostics, and full-seed experiments before drawing conclusions.

# Lightweight Multi-Scale Temporal Encoding for Efficient Adaptive Graph Forecasting

## Idea

We study whether a lightweight multi-scale temporal encoder, built from parallel nonlinear branches over pooled views of the input window, improves linear-complexity adaptive graph models such as MAGE, which compress time with a single linear projection, while keeping their cost linear in the number of nodes.

## Motivation

- MAGE (Ma et al., NeurIPS 2025) maps each node's input window to a hidden vector with one linear projection. After this layer, every block operates across nodes or channels and none operates across time.
- Average pooling is a linear map. Concatenating pooled views before MAGE's projection gives another linear function of the raw window, which the original projection can already express. Multi-scale information therefore cannot be added through the input alone. It needs scale-specific nonlinear processing before fusion.
- MAGE's router selects graph experts from the node representation. Temporal distinctions missing from that representation cannot influence which graphs a node uses.

## Approach

Replace MAGE's input layer with:

```
H = X W0 + sum_m g_m(node) * f_m(pool_m(X))
```

- `X W0`: MAGE's original linear projection, kept as a residual path (recovers full-window context lost by pooling).
- `pool_m`: average pooling with factor 2^m, scales m in {0, 1, 2} (window lengths 12, 6, 3).
- `f_m`: shallow MLP per scale (Linear, GELU, Linear) mapping to the hidden size d.
- `g_m(node)`: softmax gate over scales, computed from MAGE's node embedding, so each sensor weights scales differently.
- Calendar embeddings, node embeddings, MAGE blocks and the output layer stay unchanged.

Encoder design follows MSMixer (pooled branches, gating, linear shortcut), adapted to a 12-step window, a graph backbone and per-node gating.

## Complexity

- Encoder: O(N·P·d + (M+1)·N·d²)
- MAGE block: O(K_G·N·d·d_G + N·d²)
- Full model stays linear in N and in P. No N² term. Overhead is a constant factor of about M extra d×d layers per node.

## Data

| Dataset | Nodes | Interval | Input / horizon | Role |
|---|---|---|---|---|
| PeMS04 | 307 | 5 min | 12 / 12 | Main |
| PeMS08 | 170 | 5 min | 12 / 12 | Main |
| LargeST-SD (optional) | 716 | 5 min | 12 / 12 | Efficiency scaling |

Follow MAGE's official preprocessing, split and normalization so results are directly comparable.

## Models

| # | Model | Question |
|---|---|---|
| 1 | Original MAGE (reproduced) | Reference |
| 2 | MAGE + pooled views as extra input channels, linear projection | Does multi-scale input alone help? Expected: no |
| 3 | MAGE + single-scale nonlinear MLP, parameter-matched to 5 | Does nonlinearity or parameter count alone help? |
| 4 | MAGE + multi-scale branches, global gate | Does scale structure help? |
| 5 | MAGE + multi-scale branches, node-conditioned gate (proposed) | Does per-node scale weighting help? |

Optional: model 5 without time-of-day / day-of-week embeddings, to check overlap between the branches and calendar information.

## Protocol

- Reproduce MAGE with the official code (github.com/PoorOtterBob/MAGE, the link in the NeurIPS paper) before any modification.
- Same training settings for all models. Fix scales and fusion before looking at test results.
- 5 seeds per model, report mean and standard deviation.
- Accuracy: MAE, RMSE, MAPE, averaged and at horizons 3, 6, 12.
- Efficiency: parameter count, training time per epoch, peak GPU memory, inference time per batch (same GPU and batch size).

## Analysis

- Routing: compare graph-expert assignment distributions between models 1 and 5.
- Regime breakdown: errors under free flow, congestion onset, recovery, and high- vs. low-volatility windows.
- Scaling (optional): time and memory against N on LargeST.

## Extensions if the first results are positive

- Apply the same encoder to other linear-complexity backbones (BigST, STID, RAGL).
- Longer input windows (e.g. 288 steps) to test the encoder where temporal scales differ more.

## Closest prior work

MSMixer (arXiv 2605.02689), MobiMixer (IEEE TMC 2025), TimeMixer (ICLR 2024), SGP (AAAI 2023), HieraMix (IJCAI 2026), HD-TTS (ICML 2024), STGMS (Sci Rep 2025), STID (CIKM 2022).
---

# Summary (2026-10-18)

**Question.** Does a multi-scale nonlinear temporal encoder improve MAGE while keeping its cost linear in the number of nodes?

**Answer: no, not on 12-step windows.** The proposed encoder (model 5, per-node gate) is worse than MAGE on PEMS08 and ties it on PEMS04. Multiple scales add nothing beyond a same-size MLP. The per-node gate is the harmful part. Cost stays linear, at +8–9 % time and +8 % memory.

## What was done

1. **Setup.** Read the MAGE (NeurIPS 2025) and TimeMixer (ICLR 2024) papers and audited the official MAGE code. Built a fresh env, since miniconda is gone from this host (uv, torch 2.8+cu128). Prepared PEMS04/08 with MAGE's protocol (values + time-of-day + day-of-week, chronological 6:2:2).
2. **Found two problems in the official code.**
   - The released training engine crashes at the first validation pass: a debug `np.save` to `/data/mjm1/...` followed by `sys.exit(0)`.
   - The MAGE block's einsum is **quadratic in N** unless `opt_einsum` is installed. We re-wrote it as an exactly equivalent linear-order contraction (max relative difference 1.4e-6).
3. **Reproduced MAGE** with the official code (only the crashing lines removed), 3 seeds per dataset. It matches our own model-1 numbers on both datasets.
4. **Implemented and trained the 5 models from the design.**
   - 1: MAGE.
   - 2: pooled views → linear.
   - 3: single-scale MLP, parameter-matched.
   - 4: multi-scale with a global gate.
   - 5: multi-scale with a node-conditioned gate.

   Each trained for 5 seeds on PEMS04 and PEMS08 (50 runs, plus 6 reproduction runs) with the paper's hyperparameters.
5. **Efficiency benchmark** on an idle A6000: real dataset sizes, plus scaling from 250 to 16 000 nodes.
6. **Diagnostics** from saved checkpoints:
   - error by traffic regime (free flow, congestion onset, recovery, congested, low/high volatility);
   - graph-expert routing;
   - the learned scale gates.

The decision rule was fixed before any test result: a model counts as helping only if it beats model 1 on mean MAE and in at least 4 of 5 paired seeds.

## Main results (test, mean ± sd over 5 seeds, T = 12 → 12)

| # | model | PEMS08 MAE | PEMS08 RMSE | PEMS04 MAE | PEMS04 RMSE | encoder params |
|---|---|---|---|---|---|---|
| 1 | MAGE (linear encoder) | **13.96 ± 0.03** | 23.30 ± 0.12 | 18.25 ± 0.05 | 30.45 ± 0.20 | 7.8k |
| 2 | pooled views → linear | 13.97 ± 0.02 | **23.21 ± 0.09** | 18.30 ± 0.06 | 30.59 ± 0.28 | 13.8k |
| 3 | single-scale MLP (param-matched) | 13.98 ± 0.04 | 23.38 ± 0.09 | **18.21 ± 0.04** | **30.22 ± 0.16** | 71.6k |
| 4 | multi-scale, global gate | 13.98 ± 0.05 | 23.29 ± 0.12 | **18.21 ± 0.04** | 30.27 ± 0.18 | 71.2k |
| 5 | multi-scale, node gate (proposed) | 14.06 ± 0.02 | 23.36 ± 0.16 | 18.25 ± 0.02 | 30.38 ± 0.05 | 71.6k |
| – | official MAGE code (3 seeds) | 13.95 | 23.23 | 18.29 | 30.61 | |
| – | MAGE paper (reported) | 13.66 | 23.04 | 18.16 | 30.16 | |

**Head-to-head comparisons** (paired by seed: ΔMAE, then seeds better):

| comparison | question | PEMS08 | PEMS04 | verdict |
|---|---|---|---|---|
| 2 vs 1 | pooled views through a linear layer only? | +0.01, 3/5 | +0.05, 2/5 | no effect (as expected) |
| 3 vs 1 | nonlinear MLP path? | +0.02, 2/5 | **−0.04, 5/5** | small gain on PEMS04 only |
| 4 vs 3 | multiple scales beyond a same-size MLP? | −0.005, 3/5 | −0.003, 3/5 | **no** |
| 5 vs 4 | per-node gate over a global gate? | **+0.08, 0/5** | **+0.045, 0/5** | **harmful** |
| 5 vs 1 | proposed encoder vs MAGE? | **+0.10, 0/5** | +0.005, 3/5 | **no** |

**Cost** (one A6000):

| | MAGE | proposed (model 5) |
|---|---|---|
| train ms/step, batch 64, PEMS08 / PEMS04 size | 22.3 / 29.4 | 23.4 / 31.4 |
| peak memory MB, PEMS08 / PEMS04 size | 810 / 1280 | 864 / 1378 |
| at 16 000 nodes, batch 16: train ms / peak MB | 324.9 / 15 133 | 352.6 / 16 372 (+9 % / +8 %) |
| log–log slope vs N, time / memory | 0.93 / 0.97 | 0.92 / 0.97 |
| released MAGE einsum, without `opt_einsum` | slope 1.63 / 1.39; 6.6× slower at 8k nodes; out of memory at 16k | |

**Diagnostics:**
- **No variant reliably helps in congestion onset, recovery, or congested periods.** These are where multi-scale context was expected to matter.
- **Model 5's losses sit exactly in those regimes.** On PEMS04 (paired ΔMAE, seeds better): onset +0.14 (0/5), recovery +0.26 (0/5), congested +0.34 (0/5), while free flow ties.
- **Model 3/4's PEMS04 gain is in free flow** (−0.04), not in transitions.
- **The encoder does not change graph-expert routing.** The model 1 vs 5 difference equals the seed-to-seed difference of model 1.
- **The per-node gate pushes some sensors to almost one-hot coarse scales.** Normalised gate entropy goes as low as 0.18. This plausibly explains the loss at fast transitions: the right scale depends on the current traffic, not on which sensor it is.

## Take-aways

- **Do not pursue** the node-gated multi-scale encoder as a contribution for short (12-step) windows.
- **Any future multi-scale claim must beat model 3,** a plain nonlinear MLP path, not just MAGE. Scale selection should depend on the input window, not on node identity.
- **The only promising direction left** is long input windows (e.g. 288 steps), where the scales actually differ.
- **The quadratic-einsum issue in the released MAGE code is a separate, reportable finding.**

Details, per-seed numbers, horizon, regime and routing tables follow below.

# Implementation (as built)

Code: `graph/MAGE_MS/`. `models.py` holds the encoders and imports the official MAGE backbone unchanged. `train.py` is the training loop, `run_grid.py` the job launcher, `prep.py` the data preparation, `cost.py` the cost-vs-N benchmark, and `analyze.py` builds the tables. Python env: `MAGE_MS/.venv` (uv, torch 2.8+cu128).

| # | code name | encoder | params (PEMS04/08) |
|---|---|---|---|
| 1 | `linear` | official `Linear(12·5 → 128)` | 7.8k |
| 2 | `mslin` | pooled views (×1, ×2, ×4) as extra channels → one linear projection | 13.8k |
| 3 | `m3` | `X·W0` + one scale-1 MLP, hidden widened to 335 to match model 5 | 71.6k |
| 4 | `m4` | `X·W0` + Σ_m softmax(θ)_m · MLP_m(pool_m X), one gate shared by all nodes | 71.2k |
| 5 | `m5` | `X·W0` + Σ_m softmax(W_g·e_n)_m · MLP_m(pool_m X), with e_n = MAGE's spatial node embedding | 71.6k |

The full model has about 1.8M parameters (PEMS08), so the encoder change is about 3% of it. Gates are zero-initialised, so they start uniform. Model 3 keeps the `X·W0` residual, so models 3–5 differ only in the branches and gating.

**Data.** PEMS04/08 come from the ASTGNN repo. Inputs are values plus time-of-day plus day-of-week (C = 5). The split is chronological 6:2:2 over sample indices, as in MAGE/LargeST. All value channels are z-scored with training-span statistics. LargeST normalises channel 0 only; that is fine for PEMS03/07 but not for PEMS04/08, whose three channels have different scales. Labels are raw flow, and zeros are masked.

**Training.** Paper settings for all of models 1–5: d=128, d_G=32, 16 experts, top-4, 3 blocks; Adam lr 0.02, wd 1e-4; MultiStepLR(0.5) at the official milestones; gradient clip 5; masked L1 loss; batch size 64; at most 300 epochs, patience 30 on validation MAE. The load-balancing bias is on (lr 1e-3, sign rule), as the paper describes.

**Problems found in the official code (relevant to the protocol):**
1. **The released training engine cannot finish a run.** `src/engines/MAGE_enine.py::evaluate` saves to a hard-coded `/data/mjm1/...` path and then calls `sys.exit(0)`, so it dies at the first validation pass. The load-balancing update is also commented out. Reproduction uses `MAGE_MS/official_repro/`, a copy with only those 4 debug lines removed; everything else is as released (lr 0.01 default in `main.py`, no load balancing, threaded dataloader). Models 1–5 use our `train.py`, which follows the paper's settings.
2. **The official MAGE block is quadratic in N unless `opt_einsum` is installed.** The released block uses `einsum('bik,e,ekid,ekdj,bjf,fc->bic')`. Torch's default left-to-right contraction builds a (b, N, K, 2, N) tensor, so the linear complexity the paper claims depends on an optional package. `models.py` contracts in an explicit linear order: (E2·X) first, then a batched matmul with route·E1. The output is the same function: max relative difference 1.4e-6 vs the official einsum. All our runs use it; `cost.py` measures both.

# Plan and status

| step | what | status |
|---|---|---|
| 0 | env, data, read MAGE + TimeMixer papers, inspect official code | done 2026-10-18 |
| 1 | Reproduce model 1 with the official code: PEMS04/08 × 3 seeds | done; matches our model 1 on both datasets |
| 2 | Models 1–5 × PEMS04/08 × 5 seeds (50 runs, 4 per GPU on GPUs 0/2/3) | done 14:24, 50/50, no crashes |
| 3 | Efficiency: params, s/epoch, peak memory, inference ms per batch on an idle GPU; cost vs N (250 → 16k nodes) | done 14:55 (`MAGE_MS/cost.py`, `results/cost_*.json`) |
| 4 | Routing: expert-usage distribution, models 1 vs 5 (best checkpoints saved in `results/ckpt/`) | done 15:20 (`MAGE_MS/post.py`, `results/post_summary.md`) |
| 5 | Regime breakdown: free flow / congestion onset / recovery; high- vs low-volatility windows | done 15:20 (same script) |
| 6 | Optional: model 5 without calendar embeddings; LargeST-SD (needs the Kaggle `archive.zip`) | not started |
| 7 | **Long input windows** (follow-up, see below): models 1, 3, 4, 5 × T ∈ {12, 96, 288} × PEMS08, PEMS04 × 5 seeds (120 runs) | done (120/120, no crashes); results in "Long input windows" below |

**Step 7 design (fixed before any long-window result).** Question: did multi-scale fail at T = 12 because a 12-step window leaves nothing for coarse scales to add?
- **Data:** splits rebuilt for a 288-step maximum window (`MAGE_MS/data_w288/`), so T = 12, 96 and 288 are all scored on identical test targets. T = 12 is re-run on this split as the within-study reference. These numbers are not mixed with the T = 12 table above.
- **Scales:** T ≤ 48 keeps factors {1, 2, 4}. T ≥ 96 uses {1, 4, 16}: 5-, 20- and 80-minute resolution, with coarsest views of 6 points at T = 96 and 18 at T = 288.
- **Models:** model 2 is dropped, since a linear map of pooled views stays linear at any T. Model 3 stays parameter-matched to model 5 at each T. Encoder parameters for models 1 / 3–5: T = 96 61.6k / 192k, T = 288 184k / 476k.
- **Training:** hyperparameters unchanged.
- **Read-out:** multi-scale is credited at a window length if model 4 or 5 beats model 3 in ≥ 4/5 paired seeds with a lower mean MAE. Also check whether model 1 itself improves with T, i.e. whether MAGE can use a longer history at all.

**Decision rule (fixed before any test result):** a model "helps" on a dataset if its mean test MAE beats model 1 and the per-seed paired difference has the same sign in at least 4 of 5 seeds. Multi-scale structure is credited only if model 4 or 5 also beats model 3. Per-node gating is credited only if model 5 beats model 4.

## Upcoming plan (added 2026-10-18; starts after step 7 finishes)

### Prior-work landscape (first pass, not yet a full review)

| idea | already done by |
|---|---|
| Linear-cost adaptive graph learning on large graphs | MAGE (NeurIPS 2025), BigST (VLDB 2024), RAGL (arXiv 2506.07179), GraphSparseNet (VLDB), FaST |
| Input-conditioned mixture of encoders at the input layer | **FaST** (arXiv 2601.05174, KDD 2026): dense softmax router `softmax(X W + R_node + R_tod + R_dow)` over GLU experts; "LinearInput" ablation is worse. No multi-scale, no MAGE comparison, LargeST only (96 → 48…672, 15-min). |
| Multi-scale + MoE + graph | STM3 (arXiv 2508.12247): ≤ 307 nodes, one graph shared by all scales, static node-embedding router, horizon ≤ 96 |
| Separate graph per frequency band | WavGCRN: wavelet split, recurrent, small graphs |
| Coarse-to-fine forecasting | NHITS, TimeMixer (no graph); MBAF-GCN (small graphs) |
| Week-ahead forecasting on LargeST | FaST (single-resolution) |
| Propagation delay / lagged directed spatial dependency | PDFormer (AAAI 2023; delay-aware attention, quadratic in N), STDDE (arXiv 2402.01231; delay differential equations, learnable per-edge delays), TraverseNet (2021; lagged cross-correlation as motivation, space-time message passing), CDGNet (2021; cross-time-slice graph), DSTCGCN (2023), Traffic-DiMAGNet (Future Transportation 2026; messages shifted by estimated lead-lag delays on a directed graph), DADGN (IJMLC 2026), A-STFGCN (arXiv 2607.24885). All on small graphs (≤ ~900 nodes) as far as found; none linear-cost on LargeST. |
| Exploiting spatially/temporally correlated residuals | **SAEA** (Liu et al., Information Fusion 2026): models forecast errors as a VAR process with a road-structured sparse coefficient matrix, plugs onto STGCN/GWNet/AutoSTG etc., test-time adjustment; PeMS-228 and METR-LA-207; gains mostly at 45 min (RMSE −5–10 %), VAR(1) beats VAR(2). This is essentially our `diag_space.py` finding turned into a method. |

Consequence: the "input-conditioned scale gate" follow-up (option B) is largely taken by FaST and is only a component, not a contribution.

**Side-by-side (from the papers; not yet verified by running their code).**

| | MAGE (NeurIPS 2025) | FaST (KDD 2026) | CrossGNN (NeurIPS 2023) | RAGL (arXiv 2506.07179) | Res-RGNN (AAAI 2019) |
|---|---|---|---|---|---|
| Task / data | ST graphs; PEMS, LargeST | ST graphs; LargeST only (≤ 8.6k nodes) | general multivariate LTSF (ETT, Weather, Traffic 862, Electricity); no road graphs | ST graphs; LargeST (≤ 8.6k) | traffic, small graphs |
| Input → horizon | 12 → 12 | 96 → 48…672 (15-min) | 96 → 96…720 | 12 → 12 | short |
| Temporal encoding | one linear projection + calendar/node prompts | input MoE (dense softmax router over GLU experts) + calendar/node embeddings | **multi-scale**: FFT top-S periods → avg-pool per period → time-node graph within/across scales (learned, top-k, trend edges) | one FC projection + calendar embeddings | recurrent (GRU-style) with residuals and periodic hops |
| Spatial / variable mixing | per-node top-4 of 16 learned low-rank *differential* (signed) graphs, input-routed | 32 learned agent tokens (attention node→agent→node) | one learned variable graph with **positive and negative (heterogeneous) edges**, top-K | learned cosine graph from node embeddings, stochastic-embedding regularisation | fixed/predefined graph conv inside RNN |
| Graph changes with input? | selection yes, graphs static | yes (attention) | no (static; dynamic graphs named as future work) | no | no |
| Cost in N | O(N) (official code O(N²) without opt_einsum) | O(N·a) | linear in **L** only; variable graph is a dense D×D matrix + top-K → O(D²) | O(N) | ≥ O(E) per step, sequential over time |
| Multi-scale | no | no | yes (temporal only; one graph for variables) | no | periodic hops only |
| Output | linear decoder | single head | direct MLP head | linear head | recurrent |

**Gaps this table shows.**
- **G-a. Multi-scale temporal modelling + O(N) adaptive graphs on large networks.** CrossGNN is multi-scale but dense in variables and evaluated on ≤ 862 series without road structure; MAGE/FaST/RAGL are O(N) but single-scale. Caveat: at T = 12 our multi-scale encoder did not help MAGE; whether this gap has signal depends on step 7 (long windows), matching the 96-step setting where CrossGNN reports gains.
- **G-b. Scale-aware spatial mixing.** In CrossGNN scales interact only along time, then all scales share one variable graph; MAGE/FaST have no scales. Nobody conditions graph choice on scale at large N (see gap 1 caveats: crowded "more graphs" line, needs step 7b first).
- **G-c. Negative / heterogeneous edges at scale — withdrawn (2026-10-18).** On checking the official code, MAGE already uses *differential* graph experts: each expert is `softmax(E1[0])softmax(E2[0]) − λ·softmax(E1[1])softmax(E2[1])` (Differential-Transformer style, `cal_coefficient`), so edges can be negative. Not a gap. (Side note: `cal_outer` returns `w·I + (1−w)·I = I`, so `outer1` is unused in the release.)
- **G-d. Robustness to noise / missing sensors on large graphs.** CrossGNN's motivation (noise) is untested on LargeST-scale traffic, where missing data is common; MAGE/FaST/RAGL do not study it.
- **G-e. Output side.** All five use a single direct head; coarse-to-fine heads for long horizons (gap 2) remain open.
- **G-f. Evaluation.** MAGE, FaST and RAGL never compare against each other; no common benchmark of the O(N) models at long horizons on PEMS and LargeST.

Sources: [CrossGNN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract.html), [Res-RGNN](https://mlanthology.org/aaai/2019/chen2019aaai-gated), [RAGL](https://arxiv.org/html/2506.07179v1). "RGNN" was read as Res-RGNN (AAAI 2019); RAGL was added as the large-scale adaptive-graph model with a similar name.

### Candidate gaps

1. **Scale-specific graphs at linear cost on large graphs (demoted 2026-10-18 to an ablation inside 2).** Keep M scale tokens per node through the MAGE blocks, each routed to its own graph experts. Demoted because "more graphs per case" is a crowded line: MAGE already gives each node a per-sample top-4 mixture of 16 learned low-rank graphs (our per-node routing entropy is 0.86–0.98, so nodes do switch graphs over time), and multi-graph / dynamic / time-varying / MoE-of-graphs / per-frequency graphs are covered by ST-MGCN, DGCRN, D2STGNN, MegaCRN, TESTAM, WavGCRN. Also, there is no evidence yet that the spatial module is the bottleneck (step 7b).
2. **Coarse-to-fine output for long-horizon graph forecasting (primary, pairs with 1).** Per-scale heads (coarse daily-trend head + finer corrections, NHITS/TimeMixer-FMM style) instead of FaST/MAGE's single linear head from one d-vector. Attacks FaST on its own setting (96 → 672).
3. **Input-conditioned scale selection (component).** Model 5 shows a static per-node gate hurts at onset/recovery/congestion; scale choice must depend on the current window. Used inside 1/2, not claimed alone.
4. **Analysis contribution: when does multi-scale help graph forecasters?** Controlled paired-seed study with a parameter-matched non-multi-scale baseline (model 3), over window length, horizon, regime, and with/without calendar embeddings. Our T = 12 negative result plus step 7 is the start.
5. **Efficiency finding.** Released MAGE block is O(N²) without `opt_einsum` (slope 1.63, 6.6× slower at 8k nodes, OOM at 16k); the linear contraction fixes it.

### Candidate method (2026-10-18): scale- and state-dependent spatial propagation

Motivated by the three diagnostics above (error anatomy, leftover spatial signal, propagation by scale/state). Working idea, not yet checked against the literature in full:
- **Split each node's input** into a calendar part (existing embeddings), a slow anomaly and a fast fluctuation (fixed pooling or learned decomposition).
- **Slow anomaly → broad, instantaneous mixing:** MAGE-style low-rank graph experts, O(N).
- **Fast fluctuation → local, directed, delayed mixing:** message passing on road edges only (O(E) = O(N)) from neighbours' last few steps, separate weights per direction and lag.
- **Traffic state gates the fast operator:** direction/lag weights conditioned on the current state (e.g. speed, occupancy), so it can learn forward propagation in free flow and backward in congestion.
- Cost stays linear in N; the novelty claim is the coupling (time scale and traffic state choose the spatial operator), not delay-awareness or per-scale graphs alone.
- Closest prior work to check: WavGCRN (per-frequency graphs, no delay, small), PDFormer (delay, quadratic), STDDE (static learnable delays), Traffic-DiMAGNet (static lead-lag priors), LIFT (time-varying lead-lag, generic MTS, all-pairs), physics-guided traffic GNNs (kinematic-wave/LWR based).
- Expected gains at 1 h on PEMS are bounded by the noise floor; evaluate on transitions, longer horizons and LargeST.
- **Refinement proposed by the user (2026-10-18): separate time experts and space experts that "discuss".** Time experts = temporal operators (e.g. slow/instant, fast/forward-delayed, fast/backward-delayed, own-history only); space experts = MAGE's low-rank graphs plus road-edge operators. Instead of choosing each set independently, run R rounds (R = 2–3) per block: (1) each node picks/weights time experts from its own state; (2) space experts mix the chosen time-expert outputs over the graph; (3) the mixed spatial context is fed back to re-weight the node's time experts ("what my neighbours are doing changes how I read my own history"), and vice versa. Cost O(R·N). Prior work: TESTAM (ICLR 2024; time-only / static-graph / dynamic-graph experts run in parallel, top-1 selected by a gate; no exchange between experts), TESTAM+ (2025), HiMoE (arXiv 2412.00316), MAGE (graph experts only), FaST (input experts only). Exchange between time and space experts during processing was not found in a first search; needs a full check. First test: a 2-round version on MAGE, with the round-2 time-expert weights logged by regime to see whether the exchange actually changes the temporal operator during congestion.

### How step 7 feeds in

| step-7 outcome | supports |
|---|---|
| Model 1 improves with T, and model 4/5 beats model 3 | gaps 1 and 3 have direct signal |
| Model 1 does **not** improve with T | the single per-node vector is the bottleneck (shared by FaST) → gaps 1 and 2 |
| Model 1 improves but model 4 ≈ model 3 | longer history helps but not via scales → lean on gap 2 (output side) and gap 4 |

### Steps

| step | what | status |
|---|---|---|
| 7b | **Graph-ablation diagnostic** (cheap, on existing data): MAGE with graph mixing removed (each block's graph term replaced by identity / node-embedding only, STID-like) vs full MAGE, PEMS08/04 × 5 seeds at T = 12 and the best step-7 T; also per-regime routing (does expert choice change with free flow vs congestion?). If removing the graph costs little, spatial structure is not where gains are, and new graph designs are dropped. | not started |
| 8 | Full literature check: read FaST, STM3, WavGCRN in full; targeted search (hierarchical / multi-resolution graph + LargeST + long horizon, 2025–2026) | not started |
| 9 | Data: LargeST-SD (716 nodes; Kaggle `archive.zip`), 15-min aggregation as in FaST, plus PEMS08 | not started |
| 10 | Reproduce FaST (github.com/yijizhao/FaST) and MAGE on SD at 96 → {48, 96, 192, 672}; check FaST's reported numbers | not started |
| 11 | Diagnose FaST/MAGE failures: error by horizon, by regime (`post.py`), by node; does error concentrate at long horizons or transitions? | not started |
| 12 | Build the gap-1/2 model (scale tokens + scale-specific graph experts + coarse-to-fine heads), designed against the failures found in step 11; ablations: shared vs per-scale graphs, single vs per-scale heads, static vs input-conditioned gate; baselines must include FaST, MAGE, and a param-matched single-scale variant | not started |
| 13 | Efficiency vs N (reuse `cost.py`), confirm O(N) | not started |

Go/no-go after step 11: proceed with step 12 only if FaST/MAGE errors show a pattern that per-scale graphs or coarse-to-fine heads plausibly address (e.g. error growing with horizon faster than a coarse-trend baseline, or concentrated at transitions); otherwise write up gap 4 + 5 as an analysis paper.

Sources: [FaST](https://arxiv.org/html/2601.05174v1), [STM3](https://arxiv.org/html/2508.12247v1), [RAGL](https://arxiv.org/html/2506.07179v1), [WavGCRN review](https://themoonlight.io/review/wavelet-inspired-multiscale-graph-convolutional-recurrent-network-for-traffic-forecasting), [MS-STDT (IJCAI 2026)](https://www.ijcai.org/proceedings/2026/0311.pdf).

# Progress log

- **2026-10-18 09:45.** First grid launched with a different encoder set (pre-design). It was stopped at 09:53 when this design was adopted. Its PEMS08 runs of `linear` (seeds 0–2) and `mslin` (seeds 0–1) are identical to models 1 and 2 here, so they were kept. No other results from that grid are used.
- **2026-10-18 09:55.** Grid for this design launched (models 1–5 plus the official reproduction).
- **2026-10-18 11:41.** All 25 PEMS08 runs done, and the 3 official-code reproduction runs too. No crashes. PEMS04 started.
  On PEMS08, no encoder variant improves on MAGE's linear projection, and model 5 is worse in 5 of 5 seeds (details in Results).
- **2026-10-18 14:24.** Grid complete: all 50 runs plus the 6 reproduction runs, no crashes. PEMS04 results are added below.
- **2026-10-18 14:30.** Efficiency benchmark started on idle GPU 3. The routing/regime analysis needs checkpoints, but 5 PEMS08 runs predate checkpoint saving: model 1 seeds 0–2 and model 2 seeds 0–1, all launched before 09:55. Those 5 are being re-trained with identical settings into `results/rerun/`, only to obtain checkpoints. The Results tables keep the original runs.
- **2026-10-18 14:54.** The 5 checkpoint re-runs finished. Their test MAE vs the originals: model 1 seed 0 13.971 → 13.971, seed 1 13.937 → 13.947, seed 2 13.965 → 13.908; model 2 seed 0 13.965 → 13.945, seed 1 13.961 → 13.961. Two reproduce exactly. Three drift by 0.01–0.06 (GPU non-determinism under concurrent jobs), which is the same size as the seed spread. This does not change any comparison. Steps 4–5 use the re-run checkpoints for these 5 and the originals for the other 45.
- **2026-10-18 14:55.** Efficiency benchmark done; results added.
- **2026-10-18 15:20.** Routing and regime analyses done; results added.
- **2026-10-18 (prior-work check).** FaST (Zhao et al., arXiv 2601.05174, listed as KDD 2026; LargeST only, 96 → 48…672 steps, 15-min data) already does **input-conditioned expert mixing at the input layer**: a dense softmax router `softmax(X W + R_node + R_tod + R_dow)` over GLU experts replaces the linear input projection, and its "LinearInput" ablation is worse. So the "input-conditioned gate" follow-up (option B) is largely taken. FaST has no multi-scale temporal component (no pooling or patching), does not compare against MAGE, and is not evaluated on PEMS04/08. Our remaining novelty is in (a) the experts being *pooled views at different scales* and (b) keeping scales separate through the graph blocks (option C). FaST also becomes a required baseline for any long-window or LargeST claim.
- **2026-10-18 evening (error anatomy).** `MAGE_MS/diag.py` on the T = 12 model-1 checkpoints (5 seeds): MAGE is within ~2 MAE (PEMS08) and ~0.3 MAE (PEMS04) of an estimated measurement-noise floor; residuals are white in time and across nodes. See Results → Error anatomy. This reframes the bottleneck: on PEMS at 1 h ahead the benchmark, not the encoder, is the limit.

# Results

Reference: MAGE paper Table 5 (5-run mean MAE / RMSE / MAPE): PEMS04 18.16 / 30.16 / 12.64; PEMS08 13.66 / 23.04 / 9.09.

## PEMS08 (complete: 5 seeds per model)

Test metrics, mean ± sd over seeds 0–4. ΔMAE is paired by seed against model 1. "better" counts the seeds where the model beats model 1.

| # | model | MAE | RMSE | MAPE % | ΔMAE vs 1 | better | epochs | encoder params |
|---|---|---|---|---|---|---|---|---|
| 1 | MAGE, linear encoder | **13.96 ± 0.03** | 23.30 ± 0.12 | **9.20 ± 0.06** | | | 136 | 7.8k |
| 2 | pooled views → linear | 13.97 ± 0.02 | **23.21 ± 0.09** | 9.25 ± 0.04 | +0.01 | 3/5 | 154 | 13.8k |
| 3 | 1-scale MLP (param-matched) | 13.98 ± 0.04 | 23.38 ± 0.09 | 9.34 ± 0.22 | +0.02 | 2/5 | 129 | 71.6k |
| 4 | multi-scale, global gate | 13.98 ± 0.05 | 23.29 ± 0.12 | 9.28 ± 0.11 | +0.02 | 1/5 | 126 | 71.2k |
| 5 | multi-scale, node gate (proposed) | 14.06 ± 0.02 | 23.36 ± 0.16 | 9.40 ± 0.34 | **+0.10** | **0/5** | 112 | 71.6k |

Horizon MAE @3 / @6 / @12: model 1 13.07 / 13.97 / 15.27, model 2 13.08 / 13.97 / 15.28, model 3 13.12 / 13.99 / 15.28, model 4 13.09 / 13.99 / 15.26, model 5 13.17 / 14.07 / 15.36. Model 5 is behind at every horizon, so the gap is not specific to long-range prediction.

Per-seed test MAE (seeds 0–4):
- model 1: 13.971, 13.937, 13.965, 14.009, 13.931
- model 2: 13.965, 13.961, 13.951, 13.960, 14.008
- model 3: 13.910, 14.002, 14.018, 13.983, 14.009
- model 4: 14.017, 13.899, 13.977, 14.012, 13.993
- model 5: 14.065, 14.041, 14.078, 14.045, 14.069

**Reproduction check (official code, 3 seeds):** MAE 13.90 / 13.97 / 13.98 (mean 13.95), RMSE mean 23.23, MAPE mean 9.35 %. This matches our model 1 (13.96). Both are about 0.3 MAE above the paper's 13.66. Likely reasons: different PEMS08 copy/preprocessing (ASTGNN data, our split indices), and the paper's numbers come from its own LargeST pipeline, which we could not run as released. The comparison between models 1–5 is unaffected, since all use the same data and loop.

**Reading (PEMS08):**
- Applying the decision rule, none of models 2–5 helps. Models 2–4 are within seed noise of model 1 (|Δ| ≤ 0.02, sd ≈ 0.03–0.05, mixed signs).
- Model 5 is consistently *worse*: +0.10 MAE, 0/5 seeds better, outside the seed spread. It also stopped earliest (112 epochs on average).
- Adding nonlinearity (model 3) or multi-scale structure (model 4) buys nothing over the linear projection on a 12-step window. Making the scale gate depend on the node embedding hurts. A plausible mechanism: the gate reads MAGE's spatial prompt, which is also an additive input to every block, so the gate's gradients pull on a shared embedding. This is not tested yet; the routing analysis in step 4 can check it.

## PEMS04 (complete: 5 seeds per model)

| # | model | MAE | RMSE | MAPE % | ΔMAE vs 1 | better | epochs |
|---|---|---|---|---|---|---|---|
| 1 | MAGE, linear encoder | 18.25 ± 0.05 | 30.45 ± 0.20 | **12.47 ± 0.07** | | | 179 |
| 2 | pooled views → linear | 18.30 ± 0.06 | 30.59 ± 0.28 | 12.57 ± 0.06 | +0.05 | 2/5 | 210 |
| 3 | 1-scale MLP (param-matched) | **18.21 ± 0.04** | **30.22 ± 0.16** | 12.52 ± 0.10 | −0.04 | **5/5** | 175 |
| 4 | multi-scale, global gate | **18.21 ± 0.04** | 30.27 ± 0.18 | 12.56 ± 0.06 | −0.04 | 4/5 | 154 |
| 5 | multi-scale, node gate (proposed) | 18.25 ± 0.02 | 30.38 ± 0.05 | 12.62 ± 0.07 | +0.01 | 3/5 | 180 |

Horizon MAE @3 / @6 / @12: model 1 17.53 / 18.28 / 19.35, model 2 17.57 / 18.33 / 19.41, model 3 17.49 / 18.24 / 19.34, model 4 17.47 / 18.24 / 19.32, model 5 17.52 / 18.29 / 19.38.

Per-seed test MAE (seeds 0–4):
- model 1: 18.184, 18.217, 18.260, 18.267, 18.311
- model 2: 18.377, 18.340, 18.256, 18.271, 18.235
- model 3: 18.163, 18.187, 18.203, 18.261, 18.241
- model 4: 18.263, 18.165, 18.169, 18.200, 18.245
- model 5: 18.273, 18.248, 18.249, 18.217, 18.278

**Reproduction check (official code, 3 seeds):** MAE 18.32 / 18.34 / 18.22 (mean 18.29), RMSE mean 30.61, MAPE mean 12.46 %. This is consistent with our model 1 (18.25) and close to the paper's 18.16.

## Decision-rule comparisons (paired by seed, ΔMAE mean ± sd, seeds better)

| comparison | question | PEMS08 | PEMS04 |
|---|---|---|---|
| 3 vs 1 | does nonlinearity / capacity help? | +0.022 ± 0.062, 2/5 | **−0.037 ± 0.026, 5/5** |
| 4 vs 1 | does the multi-scale encoder help? | +0.017 ± 0.039, 1/5 | −0.040 ± 0.067, 4/5 |
| 5 vs 1 | does the proposed encoder help? | **+0.097 ± 0.038, 0/5** | +0.005 ± 0.056, 3/5 |
| 4 vs 3 | does scale structure add anything beyond the MLP? | −0.005 ± 0.079, 3/5 | −0.003 ± 0.062, 3/5 |
| 5 vs 4 | does per-node gating add anything? | **+0.080 ± 0.043, 0/5** | **+0.045 ± 0.035, 0/5** |
| 2 vs 1 | do pooled views help through the linear input alone? | +0.01, 3/5 | +0.05, 2/5 |

Verdicts under the pre-registered rule:
- **Multi-scale input alone (model 2): no**, on both datasets. This is what the design expected: pooling is linear, so the original projection can already express it.
- **Nonlinearity (model 3): helps on PEMS04 only, by 0.04 MAE (0.2 %, 5/5 seeds); no effect on PEMS08.**
- **Multi-scale structure (model 4 vs 3): not credited.** It ties the param-matched single-scale MLP on both datasets (|Δ| ≤ 0.005, sd 0.06–0.08). The model 4 gain over model 1 on PEMS04 is the same size as model 3's, so it comes from the nonlinear branch, not from the scales.
- **Per-node gating (model 5 vs 4): rejected.** It is worse on both datasets in 10/10 paired seeds (+0.08 and +0.045 MAE). Model 5 does not beat model 1 on either dataset.

## Efficiency (step 3)

Measured alone on one RTX A6000 (GPU 3, nothing else running on it). One training step = forward + backward + Adam; inference = one forward pass. Times are the mean of 10 repetitions after 3 warm-ups. All models use our linear-order MAGE contraction unless marked "official einsum".

**At the real dataset sizes (batch 64, T=12):**

| model | params PEMS08 / PEMS04 | train ms/step N=170 / 307 | infer ms/batch N=170 / 307 | peak MB N=170 / 307 |
|---|---|---|---|---|
| 1 MAGE linear | 1.815M / 2.674M | 22.3 / 29.4 | 7.4 / 8.9 | 810 / 1280 |
| 2 pooled → linear | 1.821M / 2.680M | 23.4 / 29.9 | 8.0 / 9.2 | 813 / 1283 |
| 3 1-scale MLP | 1.879M / 2.738M | 24.0 / 30.4 | 7.8 / 9.4 | 842 / 1338 |
| 4 MS global gate | 1.878M / 2.738M | 23.3 / 31.4 | 6.6 / 9.8 | 864 / 1378 |
| 5 MS node gate | 1.879M / 2.738M | 23.4 / 31.4 | 6.6 / 9.8 | 864 / 1378 |
| 1 with official einsum | 1.815M / 2.674M | 20.7 / 29.9 | 5.8 / 9.2 | 646 / 1169 |

At these sizes, all differences are within about 7 % (time) and 8 % (memory). Small-N timings are dominated by kernel-launch overhead, so ±1 ms is noise.

**Scaling with N (batch 16, synthetic data, N = 250 … 16 000):**

| N | model 1 train ms / MB | model 5 train ms / MB | model 5 overhead | model 1, official einsum, train ms / MB |
|---|---|---|---|---|
| 1 000 | 24.6 / 1 028 | 27.4 / 1 109 | +11 % / +8 % | 36.6 / 1 183 |
| 4 000 | 86.0 / 3 861 | 93.6 / 4 178 | +9 % / +8 % | 308.2 / 6 937 |
| 8 000 | 165.3 / 7 604 | 179.0 / 8 221 | +8 % / +8 % | 1 084.0 / 21 625 |
| 16 000 | 324.9 / 15 133 | 352.6 / 16 372 | +9 % / +8 % | **out of memory (46 GB)** |

Log–log slope over N ≥ 1000 (1.0 = linear):

| model | time | memory |
|---|---|---|
| models 1–5 (linear contraction) | 0.92–0.94 | 0.97 |
| model 1, official einsum | **1.63** | **1.39** |

**Reading:**
- The multi-scale encoder keeps MAGE linear in N, as the complexity analysis says, at a constant overhead of about 8–9 % time and 8 % memory (model 4/5 vs model 1). Models 3–5 cost the same, within 3 %.
- The bigger efficiency issue is in MAGE itself. **The released MAGE block is not linear in N on a stock PyTorch install**: torch contracts its einsum left to right and materialises a (b, N, K, 2, N) tensor. At N = 8000 it is 6.6× slower and needs 2.8× the memory of the same function contracted in linear order, and at N = 16 000 it does not fit on a 46 GB GPU. The paper's linear scaling presumably relies on `opt_einsum` being installed, which the README does not mention.
- Parameter count grows with N for every model (the expert embeddings E1/E2 are N × d_G per expert: about 6.3k parameters per node), so the 64k-parameter encoder is 3.5 % of the model at N = 170 and <0.1 % at N = 16 000.

## Regime breakdown (step 5)

Horizon-averaged masked test MAE per regime, mean over 5 seeds. Brackets: paired Δ vs model 1, then seeds better.
Regimes are per (window, sensor), using the raw speed channel with a 40 mph threshold. Free flow: input and target speeds all ≥ 40. Onset: last input ≥ 40 and some target < 40. Recovery: the reverse. Congested: everything < 40. Volatility: std of input-flow first differences, bottom / top quartile with thresholds from training windows. The target window is used only to stratify errors. Sample counts (PEMS08 / PEMS04): free flow 573k / 965k, onset 10.5k / 23.8k, recovery 7.5k / 17.4k, congested 4.1k / 9.1k, low / high volatility 163k / 139k and 254k / 258k.

**PEMS08**

| model | free flow | onset | recovery | congested | low vol. | high vol. |
|---|---|---|---|---|---|---|
| 1 | 13.12 | 31.81 | 29.35 | 28.12 | 4.98 | 24.56 |
| 2 | +0.02, 2/5 | −0.10, 3/5 | −0.02, 3/5 | −0.12, 4/5 | +0.07, 0/5 | −0.02, 3/5 |
| 3 | +0.03, 2/5 | −0.01, 2/5 | +0.32, 0/5 | −0.13, 3/5 | +0.07, 1/5 | +0.06, 2/5 |
| 4 | +0.03, 1/5 | −0.04, 3/5 | +0.09, 1/5 | −0.11, 4/5 | +0.05, 1/5 | +0.01, 2/5 |
| 5 | **+0.10, 0/5** | +0.05, 2/5 | **+0.26, 0/5** | +0.25, 1/5 | **+0.14, 0/5** | **+0.11, 0/5** |

**PEMS04**

| model | free flow | onset | recovery | congested | low vol. | high vol. |
|---|---|---|---|---|---|---|
| 1 | 17.06 | 34.64 | 34.01 | 35.96 | 5.71 | 33.51 |
| 2 | +0.06, 1/5 | −0.01, 2/5 | −0.10, 4/5 | −0.05, 3/5 | +0.05, 0/5 | +0.01, 3/5 |
| 3 | **−0.04, 5/5** | −0.00, 2/5 | +0.01, 3/5 | +0.00, 3/5 | −0.01, 2/5 | −0.02, 3/5 |
| 4 | −0.04, 4/5 | −0.02, 3/5 | +0.03, 2/5 | −0.04, 3/5 | +0.01, 3/5 | −0.03, 3/5 |
| 5 | −0.01, 3/5 | **+0.14, 0/5** | **+0.26, 0/5** | **+0.34, 0/5** | −0.03, 4/5 | +0.04, 2/5 |

**Reading:**
- No variant reliably improves the hard regimes (onset, recovery, congestion). Multi-scale context was supposed to help there, and the per-regime deltas for models 2–4 are mixed-sign and within noise.
- The PEMS04 gain of models 3/4 sits in **free flow** (−0.04, 5/5 and 4/5), which is 94 % of the samples. It is not a transition effect.
- **Model 5's loss is concentrated in the transition regimes.** On PEMS04, onset +0.14, recovery +0.26 and congested +0.34 are all 0/5 seeds better, while free flow ties. On PEMS08 it is worse almost everywhere.

## Error anatomy (where model 1's error comes from; `MAGE_MS/diag.py`, `results/diag.json`)

Model 1 (MAGE), T = 12, original split, test set, seed 0 unless noted. Noise floor = variogram estimate of white measurement noise on the test span (`γ(1), γ(2)` of first/second differences, linear-drift approximation, Gaussian MAE = 0.8σ); "interpolation" = |y_t − (y_{t−1}+y_{t+1})/2|, which uses the future neighbour and is shown only as a reference.

| | PEMS08 | PEMS04 |
|---|---|---|
| MAGE MAE, avg over 12 steps | 13.97 | 18.18 |
| MAGE MAE at h = 1 / 3 / 6 / 12 | 12.26 / 13.07 / 13.98 / 15.30 | 16.63 / 17.49 / 18.21 / 19.27 |
| Persistence (last value), avg / h = 12 | 25.29 / 34.70 | 31.59 / 42.40 |
| Historical average (node × slot of week), avg | 23.23 | 26.07 |
| **Estimated noise floor (MAE)** | **≈ 12.0** | **≈ 17.9** |
| Interpolation with future neighbour (reference) | 13.2 | 17.9 |
| 5-seed ensemble of MAGE | 13.75 (−0.22) | 18.06 (−0.12) |
| Mean |sd| of predictions across seeds | 2.8 | 3.1 |
| Residual correlation across nodes (h = 1) | 0.05 | 0.02 |
| Residual lag-1 autocorrelation (h = 1) | 0.04 | 0.05 |
| Share of error from the largest 10 % of errors | 40 % | 41 % |
| Share of error from the worst 10 % of nodes | 20 % | 22 % |
| Free flow: share of points / of error / MAE | 94.7 % / 89.1 % / 13.1 | 92.8 % / 86.6 % / 17.0 |
| Onset + recovery + congested: share of points / of error | 3.7 % / 7.9 % | 4.8 % / 9.3 % |

Reading:
- **The 1-hour PEMS task is near its noise floor.** Already at 5 minutes ahead MAGE's MAE (12.3 / 16.6) is ~88–91 % of its 12-step average, and the estimated noise floor (12.0 / 17.9) is close to the average itself. Reducible error is roughly ≤ 2 MAE on PEMS08 and a few tenths on PEMS04. This explains why every encoder change moves MAE by ≤ 0.1 and why published gains on these benchmarks are small. The noise estimate is rough (assumes white noise plus locally linear signal) but agrees with the next point.
- **Residuals look like white noise over time, and on average across nodes — but not between road neighbours** (corrected after `diag_space.py`, next subsection): the all-pairs average (0.02–0.05) hides a road-neighbour correlation of 0.11–0.19 with a downstream lag. So part of what the variogram counts as "noise" is spatially coherent traffic disturbance, and the noise-floor estimate is an upper bound on truly irreducible error.
- **Transitions are hard but small.** Onset/recovery/congested cost 2× the free-flow MAE but are < 5 % of points and < 10 % of total error; free flow (mostly noise) dominates the average.
- **Variance is a bigger lever than architecture here.** Averaging 5 seeds gains −0.22 / −0.12 MAE, more than any encoder variant.
- **Where reducible error exists:** error grows with horizon (h = 1 → 12: +25 % / +16 %) while the noise floor does not, so the share of reducible error rises with the horizon. Long horizons (FaST's setting) and transitions are where models can still differ; average MAE at 1 h on PEMS is not.

## Space: value of neighbours and leftover spatial signal (`MAGE_MS/diag_space.py`, `results/diag_space.json`)

Neighbours from the ASTGNN road edge list (both directions; mean degree ≈ 2–3). (1) Per-node ridge regression on the data itself: own last 12 steps + own historical-average value for the target slot, vs the same plus neighbours' last 12 steps. (2) Correlation of MAGE's test residuals (model 1, seed 0) for road-edge pairs vs random pairs; "lag k" = residual of the first node at target time t vs the second at t + k; edge direction as given in the file (from → to).

| | PEMS08 | PEMS04 |
|---|---|---|
| Ridge, own history + calendar avg: MAE h = 1 / 6 / 12 | 13.79 / 17.49 / 19.67 | 18.04 / 21.94 / 24.05 |
| Ridge, + road neighbours: MAE h = 1 / 6 / 12 | 13.11 / 16.97 / 18.92 | 17.38 / 21.54 / 23.64 |
| Gain from neighbours (linear) | 0.68 / 0.52 / 0.75 | 0.66 / 0.40 / 0.41 |
| MAGE MAE h = 1 / 6 / 12 (for reference) | 12.26 / 13.98 / 15.30 | 16.63 / 18.21 / 19.27 |
| Residual corr, h = 1, lag 0: road edge / random | 0.11 / 0.04 | 0.12 / 0.01 |
| Residual corr, h = 1, lag 1: from→to / to→from / random | **0.085 / 0.009** / 0.00 | **0.129 / 0.018** / 0.00 |
| Residual corr, h = 12, lag 0: road edge / random | 0.16 / 0.06 | 0.19 / 0.03 |
| Residual corr, h = 12, lag 1: from→to / to→from / random | 0.135 / 0.092 / 0.03 | 0.196 / 0.125 / 0.02 |

Reading:
- **Space matters, modestly, for a linear model** (−0.4 to −0.75 MAE, 2–4 %). MAGE beats ridge-with-neighbours by 1.5 MAE at h = 1 and by 3.6–4.4 at h = 12, so its advantage grows with horizon; whether that comes from the graph or from nonlinearity/embeddings is what step 7b (graph ablation) separates.
- **MAGE leaves spatially coherent error along roads,** 3–8× the random-pair level, and larger at h = 12 (0.16–0.19) than at h = 1. These are unpredicted disturbances shared by adjacent sensors, not independent sensor noise.
- **The leftover error propagates in one direction with a delay.** At h = 1, the error of a sensor at time t correlates with the error of its `to` neighbour 5 min later (0.085 / 0.129) but hardly the reverse (0.009 / 0.018): a disturbance travels along the road and MAGE predicts neither end. MAGE's graph experts mix nodes at a single time step through learned low-rank kernels, with no explicit direction or delay.
- Caveat: this lagged signal is not directly usable at forecast time (the upstream disturbance at t is itself in the future at the forecast origin for h = 1); it shows that errors are travelling waves, which may be visible further upstream or earlier. Delay-aware spatial modelling exists (e.g. PDFormer, AAAI 2023, quadratic attention); whether linear-cost delay/direction-aware graphs help is open.

## Propagation by time scale and traffic state (`MAGE_MS/diag_lag.py`, `results/diag_lag.json`)

Data only, training span. Flow minus its per-node slot-of-week mean, split into slow (1-h moving average) and fast (remainder) parts. Mean correlation over road edges (from → to) of `from` at t with `to` at t + k. Fast band also split by the `from` sensor's state (speed ≥ 40 / < 40 mph; congested 2–3 % of time). Median edge length 0.33 / 0.37 km.

| lag (min) | −15 | −10 | −5 | 0 | +5 | +10 | +15 |
|---|---|---|---|---|---|---|---|
| PEMS08 slow | 0.30 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 | 0.30 |
| PEMS08 fast | −0.02 | −0.01 | 0.00 | 0.05 | **0.09** | 0.00 | −0.01 |
| PEMS08 fast, free flow | −0.03 | −0.02 | 0.00 | 0.07 | **0.12** | 0.00 | −0.02 |
| PEMS08 fast, congested | 0.02 | 0.04 | 0.06 | **0.09** | 0.06 | 0.02 | −0.03 |
| PEMS04 slow | 0.52 | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 | 0.52 |
| PEMS04 fast | −0.02 | −0.01 | 0.00 | 0.07 | **0.12** | 0.01 | −0.02 |
| PEMS04 fast, free flow | −0.03 | −0.02 | 0.00 | 0.07 | **0.14** | 0.01 | −0.02 |
| PEMS04 fast, congested | 0.03 | 0.03 | 0.03 | **0.06** | 0.04 | 0.01 | −0.01 |

Reading:
- **Slow anomalies (how today differs from the usual week) are shared broadly and without delay** (0.3–0.5, flat over ±30 min).
- **Fast fluctuations are local, directed and delayed:** weak overall but peaked at +5 min from → to, near zero in reverse.
- **In congestion the pattern changes:** the peak moves to lag 0 and spreads to negative lags (`to` leading `from`), i.e. influence also runs backwards, as expected for queues growing upstream.
- So the right spatial operator depends on the time scale (broad/instant vs local/directed/delayed) and on the traffic state (forward vs backward). Models that compress time first and then mix space with one operator (MAGE, FaST, RAGL) cannot express this.
- Caveats: a 5-min lag over ~0.35 km is far slower than vehicle travel time, so the +5 min peak may partly reflect detector aggregation/timestamp alignment rather than physical travel; it is nonetheless an exploitable lead-lag. The fast-band correlations are small (≤ 0.14) and congestion is rare, so expected accuracy gains on 1-h PEMS are modest; transitions and longer horizons are where it should show.

## Long input windows (step 7; `MAGE_MS/run_long.py`, `results/long/`)

Test MAE, mean ± sd over 5 seeds. All windows are scored on the same `data_w288` test targets. Model 3 is the same-size MLP, model 4 is multi-scale with a global gate, and model 5 is multi-scale with a per-node gate.

| | T | linear (m1) | m3 | m4 | m5 |
|---|---|---|---|---|---|
| PEMS08 | 12 | **13.880 ± 0.010** | 13.977 ± 0.059 | 13.904 ± 0.033 | 13.981 ± 0.029 |
| PEMS08 | 96 | **13.802 ± 0.044** | 13.990 ± 0.039 | 13.854 ± 0.039 | 14.024 ± 0.088 |
| PEMS08 | 288 | **13.774 ± 0.065** | 14.345 ± 0.063 | 14.129 ± 0.084 | 14.157 ± 0.078 |
| PEMS04 | 12 | 18.345 ± 0.051 | **18.211 ± 0.047** | 18.222 ± 0.049 | 18.253 ± 0.023 |
| PEMS04 | 96 | 18.270 ± 0.081 | 18.331 ± 0.031 | **18.221 ± 0.020** | 18.293 ± 0.051 |
| PEMS04 | 288 | **18.248 ± 0.038** | 18.795 ± 0.055 | 18.590 ± 0.104 | 18.649 ± 0.051 |

Paired seeds better (out of 5):

| | T | m4 < m3 | m5 < m4 | m4 < linear | linear(T) < linear(12) |
|---|---|---|---|---|---|
| PEMS08 | 12 / 96 / 288 | 5 / 5 / 5 | 0 / 0 / 1 | 1 / 1 / 0 | – / 5 / 5 |
| PEMS04 | 12 / 96 / 288 | 2 / 5 / 5 | 0 / 0 / 1 | 5 / 4 / 0 | – / 4 / 5 |

**Reading:**
- **A longer window helps linear MAGE a little:** −0.08 to −0.11 MAE (about 0.6 %) on both datasets.
- **Multi-scale structure beats a same-size MLP once the window is long.** m4 beats m3 5/5 at T = 96 and 288 on both datasets.
- **Multi-scale beats linear MAGE only once, and only narrowly:** PEMS04 at T = 96, by −0.05 in 4/5 seeds. At T = 288 every nonlinear encoder is clearly worse, which is consistent with overfitting about 5 weeks of training data.
- **The per-node gate never helps** (m5 < m4 in at most 1 of 5 seeds anywhere).
- **Caveat:** linear MAGE hit the 300-epoch cap in 4/5 (PEMS08) and 5/5 (PEMS04) runs at T = 96 and 288, so the long-window baselines may be slightly under-trained.

## Routing and scale gates (step 4)

Graph-expert usage on the test set (top-4 of 16 experts per node and block), mean over 5 seeds. Values are normalised entropy of global usage / mean per-node entropy / largest expert share.

| | block 1 | block 2 | block 3 |
|---|---|---|---|
| PEMS08 model 1 | 0.993 / 0.915 / 0.089 | 0.993 / 0.860 / 0.080 | 0.981 / 0.974 / 0.114 |
| PEMS08 model 5 | 0.993 / 0.922 / 0.094 | 0.980 / 0.878 / 0.090 | 0.987 / 0.978 / 0.094 |
| PEMS04 model 1 | 0.973 / 0.923 / 0.100 | 0.987 / 0.894 / 0.089 | 0.979 / 0.975 / 0.106 |
| PEMS04 model 5 | 0.974 / 0.918 / 0.119 | 0.989 / 0.879 / 0.085 | 0.980 / 0.966 / 0.107 |

Jensen–Shannon divergence of global expert usage, per block:

| | PEMS08 | PEMS04 |
|---|---|---|
| model 1 vs model 5 (same seed) | 0.011 / 0.023 / 0.021 | 0.033 / 0.020 / 0.033 |
| model 1 vs model 1 (different seed) | 0.009 / 0.011 / 0.027 | 0.040 / 0.017 / 0.034 |

Learned scale gates (scales ×1, ×2, ×4):

| | PEMS08 | PEMS04 |
|---|---|---|
| model 4 (global) | 0.15 / 0.42 / 0.43 | 0.13 / 0.35 / 0.52 |
| model 5 (per node), mean | 0.13 / 0.31 / 0.56 | 0.17 / 0.33 / 0.50 |
| model 5, sd across nodes | 0.09 / 0.19 / 0.19 | 0.09 / 0.17 / 0.19 |
| model 5, gate entropy mean / min (1 = uniform) | 0.73 / 0.18 | 0.82 / 0.25 |

**Reading:**
- **The encoder does not change which graphs nodes use.** The difference in expert usage between models 1 and 5 is the same size as the difference between two seeds of model 1. Load balancing works in both: global usage entropy ≥ 0.97, and no expert above 12 %. The design's premise that "temporal distinctions missing from the representation cannot influence which graphs a node uses" is not contradicted, but adding them does not shift routing measurably either.
- **Both gates move weight away from the full-resolution branch** (×1 gets 0.13–0.17) toward the coarse ×2/×4 views. This is consistent with the linear residual `X·W0` already carrying the full-resolution signal.
- **The per-node gate does specialise.** Gates differ across sensors (sd up to 0.19), and some sensors become nearly one-hot (min entropy 0.18–0.25). Together with the regime breakdown, this suggests a mechanism for model 5's loss: sensors that commit to coarse views lose short-term detail exactly where traffic changes fast (onset, recovery, congestion). The gate is a static per-sensor choice, but the right scale depends on the current regime. This is a hypothesis consistent with the data, not a tested causal claim.

# Conclusion

*Final for the planned steps 0–5 (2026-10-18). Step 6 (optional) not run.*

1. **The proposed encoder (model 5) does not improve MAGE.** It is reliably worse on PEMS08 (+0.10 MAE, 0/5 seeds better) and ties MAGE on PEMS04. Per-node scale gating is the harmful part: it loses to the same encoder with a global gate in 10 of 10 paired runs.
2. **Multi-scale structure adds nothing measurable on a 12-step window.** With a global gate (model 4), it ties a parameter-matched single-scale MLP (model 3) on both datasets. Multi-scale input without nonlinearity (model 2) also does nothing, as predicted.
3. **What does help, slightly and only on PEMS04, is adding a nonlinear MLP path to MAGE's linear input projection:** −0.04 MAE (0.2 %), 5/5 seeds. On PEMS08 it has no effect. This is a small, dataset-dependent capacity effect, not a multi-scale effect.
4. **Cost stays linear, as intended:** all variants scale with slope ≈ 0.93 in N, and the multi-scale encoder costs about 8–9 % more time and 8 % more memory than MAGE at every N. Separately, the *released* MAGE block is quadratic in N unless `opt_einsum` is installed (slope 1.63, out of memory at 16k nodes). That is worth reporting regardless of the encoder result.
5. **The motivation's premise holds in principle but not in effect size.** MAGE's three blocks with SwiGLU FFNs and the hyper-residual seem to recover most of what a nonlinear temporal encoder adds, at least for 12 input steps, where pooling ×4 leaves only 3 points.
6. **Diagnostics agree with the accuracy numbers.** The encoder does not shift graph-expert routing beyond seed noise. No variant improves the hard regimes (congestion onset, recovery, congested). The per-node gate specialises sensors to coarse scales, and model 5's losses fall exactly in those fast-changing regimes. A static per-sensor scale choice appears to be the wrong inductive bias; any scale selection should depend on the input window (current regime), not on node identity.
7. **Interpretation limits:** two datasets, one window length (12), PEMS data from the ASTGNN copy. Our absolute numbers sit 0.1–0.3 MAE above the paper's, but the official code gives the same numbers on this data.

**Recommendation.** Do not pursue the node-gated multi-scale encoder as a contribution on short windows. If the line continues, the one place multi-scale could plausibly matter is long input windows (the "Extensions" item, e.g. 288 steps), where scales actually differ. Model 3 (a plain nonlinear MLP path) should be the baseline any multi-scale claim must beat.

# Step 8: scale-dependent diffusion depth (setup, 2026-10-27)

*Rewritten from the draft spec added on 2026-10-27. Cheap check launched 2026-10-31 01:28 (see "Step 8 runs" below). The validity review at the end changes four details of the draft: MAGE's block wiring is kept, two sanity arms are added, the eigenvalue diagnostic is computed differently, and a data-only pre-check runs first.*

**Hypothesis.** Coarse temporal views carry slow, spatially broad variation, and fine views carry fast, local variation. So each temporal scale should mix over a different spatial range. Concretely, one shared graph is used per block, and scale m applies it k_m times, with coarser scales applying it more often. Evidence so far: the slow component is correlated with road neighbours at every lag (0.31 / 0.53), while the fast component is weaker (0.09 / 0.12) and lagged by 5 min. Cost stays linear in N and in T.

**Baseline.** Linear MAGE at T = 96 from step 7: PEMS08 13.802 ± 0.044, PEMS04 18.270 ± 0.081.

## Model

Notation: input X (T = 96 steps, N nodes, C channels).

1. **Scales.** Average-pool X by factors s ∈ {1, 2, 4, 8}, giving lengths 96 / 48 / 24 / 12 (5, 10, 20 and 40 min per step).
2. **Encoders.** One linear projection per scale gives H_m = X_m W_m. MAGE's prompt (node, time-of-day and day-of-week embeddings) is added, shared across scales. This is the same encoder type as linear MAGE: no MLP, because step 7 showed nonlinear encoders overfit at T = 96.
3. **Graph operators per block ℓ.** These use MAGE's existing parameters, never form an N × N matrix, and are shared by all scales:
   - **Routing.** Computed once per block from the finest scale: r = normalised top-4 sigmoid route of LN(H_1). It is reused for every scale. Load-balancing bias and the bincount also come from the finest scale.
   - **Positive and negative graphs.** S⁺ = Σ_e diag(r_e) softmax(E1⁺_e) softmax(E2⁺_e), and S⁻ likewise from E1⁻ / E2⁻. Each is row-stochastic, because both softmaxed factors and the normalised route are. λ is MAGE's per-block `cal_lambda`.
   - **Lazy operator.** P = (I + S⁺)/2.
   - **Per-scale operator.** S_m = P^{k_m} − λ S⁻, with the contrast applied once. P^k·Z is computed by applying (Z + S⁺Z)/2 k times, each pass in the factorised order of `linear_mage_block_forward`.
4. **Block.** Per scale, U = H + S_m(LN H), then H ← U + FFN(LN U). This is MAGE's block with only the operator changed. LN, the SwiGLU FFN, the router, E1/E2 and λ are shared across scales.
5. **Wiring, kept from MAGE per scale.** z_m = B1(h_m), x_m = B2(z_m), x_m = B3(h_m − x_m). Each scale has its own decoder pair, D_m(x_m) + D′_m(z_m). The forecast is the sum over scales.
6. **Cost per block.** There are Σ_m k_m applications of S⁺ plus one of S⁻ per scale, each O(N · 16 · 32 · d). So the cost is linear in N. It is also linear in T, because only the encoders see T.

## Arms (`--arm`)

| arm | k per scale (×1, ×2, ×4, ×8) | operator | what it tests |
|---|---|---|---|
| `mage` | one scale (×1) | S⁺ − λS⁻ (MAGE exactly) | code check: must reproduce 13.802 ± 0.044 |
| `lazy1` | one scale (×1), k = 1 | P − λS⁻ | effect of the lazy operator alone |
| `shared` | 1, 1, 1, 1 | P − λS⁻ | four scale streams, same mixing |
| `proposed` | 1, 2, 4, 8 | P^k − λS⁻ | coarse → broad mixing |
| `reversed` | 8, 4, 2, 1 | P^k − λS⁻ | wrong direction; same compute as `proposed` |
| `uniform` | — | (1/N)·11ᵀ − λS⁻ | broad mixing with no graph structure |
| `nomix` | — | S_m = 0 | no spatial mixing (also the step-7b graph ablation at T = 96) |
| `learned` | Σ_{k=0..8} softmax(γ_m)_k P^k | − λS⁻ | does the model choose depth rising with scale? γ_m zero-initialised, 9 per scale |
| `road` (optional) | 1, 2, 4, 8 | P = (I + D⁻¹A_road)/2 | the only arm where k hops means physical distance |

## Protocol

- **Data.** PEMS08 (then PEMS04 if the go rule passes), 96 → 12. Use `data_w288` with the same split, scaler and test origins as the step-7 T = 96 runs, so the baseline numbers are reused directly.
- **Training.** Same recipe as linear MAGE: lr 0.02, wd 1e-4, batch 64, patience 30, 300-epoch cap. Seeds 0–4 for every arm.
- **Log.** Masked MAE / RMSE / MAPE, both averaged and at horizons 3 / 6 / 12. Also params, s/epoch, peak GPU memory and `epochs_run`.
- **Epoch-cap caveat.** The 300-epoch cap was hit by 4/5 linear T = 96 runs on PEMS08 and 5/5 on PEMS04. The cap is kept for comparability, and arms that hit it are flagged.
- **Staging.**
  1. Pre-check (below).
  2. Unit tests.
  3. Seeds 0–1 for all arms.
  4. Seeds 2–4 only if `proposed` is not clearly dead (worse than `shared` on both seeds by more than 0.1).

**Pre-check (data only, about 10 min, before any training).** Pool the de-seasonalised flow at ×1 / ×2 / ×4 / ×8. For each scale, compute the mean correlation between sensor pairs as a function of road hop distance (1 to 8). The hypothesis predicts that coarser scales decay more slowly with hops. If the four decay curves lie on top of each other, the premise is weak and step 8 should be rethought before using GPUs. `diag_lag.py` only measured direct neighbours (one hop), so it does not answer this.

**Decision rule (fixed before any result).** "A beats B" means a lower mean test MAE and the same-sign paired difference in ≥ 4/5 seeds. Seed sd at T = 96 is 0.04–0.09, so differences below about 0.05 MAE are not interpretable.

| comparison | question |
|---|---|
| C1 `proposed` vs `shared` | does scale-dependent depth help beyond having four scale streams? |
| C2 `proposed` vs `reversed` | is the direction (coarse → broad) what matters? This comparison is compute-matched. |
| C3 `proposed` vs `uniform` | does depth 8 still use graph structure, or is it a global mean? |
| C4 best arm vs linear MAGE T = 96 | is it useful at all? |
| `learned` | does mean depth d_m rise with coarseness in ≥ 4/5 seeds? |

Go to PEMS04 and LargeST only if C1, C2 and C4 pass. If C1 and C2 pass but C4 fails, the mechanism exists but is not useful yet. Record that, and do not build on it.

**Diagnostics.**
- **`proposed`: how different the powers are.** Report ‖P^k − P^{k′}‖_F / ‖P^{k′}‖_F for (k, k′) ∈ {(1,2), (2,4), (4,8)}, and ‖P^8 − P^∞‖_F / ‖P^∞‖_F, with P^∞ ≈ P^64.
  - These are computed densely on 20 test batches × 5 seeds, per block. That is cheap at N ≤ 307.
  - P depends on the routing, so the values are per sample and are reported as mean and sd.
- **`proposed`: spectrum.** Eigenvalues μ of S⁺ (dense, same samples), giving the eigenvalues of P as ν = (1 + μ)/2.
  - Report |ν₂| and the mixing time 1/(1 − |ν₂|).
  - P always has eigenvalue 1/2 with multiplicity N − rank(S⁺), so |ν₂| ≥ 1/2 whenever S⁺ is rank-deficient.
  - The r × r shortcut (eigenvalues of VU instead of UV) only pays when r < N. Here r = 16 experts × 32 = 512 > N, so it is used only for LargeST.
- **`learned`.** d_m = Σ_k k · softmax(γ_m)_k per scale, block and seed, plus the entropy of softmax(γ_m).
- **All arms.** Expert-usage entropy, as in step 4.

**Unit tests.**
1. Rows of S⁺ and P sum to 1, and rows of S_m sum to 1 − λ (random input, atol 1e-5).
2. The factorised P^k·Z equals the dense matrix power times Z (N = 50, float64, k ≤ 8).
3. Peak memory and time grow linearly in N up to 16k nodes, using `cost.py`. This catches an accidental N × N allocation.
4. With the same weights, the `mage` arm's forward pass equals `MAGEX(encoder='linear')` to float precision. One seed of training must land within the step-7 range.

**Output.**
- A table of arm × (MAE / RMSE / MAPE mean ± sd, horizons 3 / 6 / 12, params, s/epoch, memory).
- Paired ΔMAE versus linear MAGE T = 96 and versus `shared`, as mean ± sd with the count of seeds better.
- Figures: depth d_m per scale for `learned`, and the ‖P^k − P^∞‖ curve.

**Budget estimate (to be measured in the unit-test run).**
- Linear MAGE at T = 96 takes about 15 s/epoch on PEMS08.
- `proposed` runs four scale streams with 15 S⁺ passes per block, against 1 for MAGE. A guess is 3–5× MAGE's time, which is 4–6 h per run at the cap.
- 9 arms × 5 seeds = 45 runs, at 12 concurrent runs on 3 GPUs, comes to roughly 1–1.5 days. Staging (seeds 0–1 first) halves this if an arm is dead early.

## Validity review of the draft spec (2026-10-27)

The draft is a sound controlled experiment. The `shared`, `reversed`, `uniform` and `nomix` arms isolate the claim well. Cost is linear in N and T as required. These issues needed fixing or need watching:

1. **"Shared with one scale reproduces linear MAGE" was not true as drafted.**
   - With the lazy operator, k = 1 gives (I + S⁺)/2 − λS⁻, not MAGE's S⁺ − λS⁻. Half of the positive mixing becomes identity, on top of the residual.
   - The draft's plain block stack and single head also dropped MAGE's hyper-residual (h − x) and its second decoder on z.
   - **Fix:** MAGE's wiring is kept per scale, and two arms are added: `mage` (exact operator, code check) and `lazy1` (laziness only). Otherwise, a change between linear MAGE and `shared` could come from any of three edits.
2. **"Hops" on MAGE's graph are not distances.**
   - S⁺ maps each node to 32 latent hubs and back, initialised near uniform. It is a dense graph with no road locality.
   - Powers of such an operator approach the rank-one average P^∞ quickly. So k = 8 may be close to `uniform`, and `proposed` would then mean "coarse scales get the global mean", not "broader neighbourhood".
   - The `uniform` arm (C3) and the ‖P^8 − P^∞‖ diagnostic detect this. The optional `road` arm is the version where depth means physical range.
   - **This is the main risk to the paper's story,** even if accuracy improves.
3. **The learned graph can partly absorb the power.** E1/E2 are trained, so the model can sharpen S⁺ until P^8 stays local. Because all scales share one S⁺, it cannot do that freely for every scale at once. The claim being tested is therefore the coupling ("the same graph at different depths per scale"), not a fixed receptive field per scale. The `learned` arm shows whether depth separates by scale when it is free to.
4. **The eigenvalue shortcut does not shrink the problem here.** The stacked low-rank factors have r = 16 × 32 = 512 > N (170 / 307), and P varies with each sample's routing. So the dense N × N matrix is used for the diagnostics, and the r × r form is kept for LargeST.
5. **Compute is not matched across all arms.** Per block, `proposed` and `reversed` each run 15 S⁺ passes, `shared` runs 4 and MAGE runs 1, and every multi-scale arm carries four streams through the backbone. C2 is the compute-matched comparison. C1 credits structure, not compute, only if `proposed` also beats `reversed`.
6. **The premise has only partial evidence.** `diag_lag` shows slow variation is more correlated across direct road neighbours than fast variation. It does not show that slow variation reaches further. That is what the pre-check measures, and it costs no GPU time.
7. **Novelty is narrow, and a literature check is needed before any claim.** The closest work known so far:
   - MTGNN mix-hop: learned weights over hop depths, but a single temporal scale. The `learned` arm is essentially mix-hop applied per scale.
   - Graph WaveNet: K-step diffusion.
   - MSGNet (AAAI 2024): a separate learned graph per temporal scale.
   - HD-TTS (ICML 2024): joint hierarchical temporal and spatial pooling.

   The specific combination proposed here is untested in our check so far: one shared low-rank graph, with depth tied to temporal scale, at O(N) on large graphs.
8. **The baseline is under-trained.** Linear MAGE at T = 96 hit the 300-epoch cap in 9 of 10 runs (PEMS08 and PEMS04). Gains of a few hundredths could come from training-length differences. Report `epochs_run` per arm, and if needed rerun the baseline with a higher cap.

**Verdict before the pre-check.** Valid to run after the fixes above, starting with the data-only pre-check. The result is publishable only if C1–C3 pass and the diagnostics show that P^8 is still structured, not a global mean.

## Pre-check result (2026-10-27; `MAGE_MS/precheck_hops.py`, `results/precheck_hops*.json`)

Method: mean correlation of de-seasonalised flow between sensor pairs, by road hop distance (median edge length 0.33 / 0.37 km), at each pooling scale, over the training span. The second table first subtracts the network-wide mean at every time step (the common shock).

**Raw (correlation at hop 1 / hop 8, and hop-8 value as a fraction of hop-1):**

| scale | PEMS08 | PEMS04 |
|---|---|---|
| ×1 (5 min) | 0.271 / 0.148 (0.55) | 0.405 / 0.270 (0.67) |
| ×2 | 0.335 / 0.186 (0.56) | 0.482 / 0.317 (0.66) |
| ×4 | 0.375 / 0.218 (0.58) | 0.533 / 0.354 (0.66) |
| ×8 (40 min) | 0.404 / 0.247 (0.61) | 0.567 / 0.385 (0.68) |

**Common shock removed (correlation at hops 1 / 2 / 3 / 8):**

| scale | PEMS08 | PEMS04 |
|---|---|---|
| ×1 | 0.132 / 0.054 / 0.022 / 0.012 | 0.233 / 0.141 / 0.090 / 0.001 |
| ×8 | 0.187 / 0.083 / 0.035 / 0.009 | 0.333 / 0.216 / 0.137 / 0.005 |

**Reading:**
- **Coarser scales are more correlated at every distance, but they do not reach further.** The decay shape (correlation relative to hop 1) is almost the same at all four scales: hop-8 ratio 0.55 → 0.61 on PEMS08, and 0.67 → 0.68 on PEMS04.
- **Most of the long-range correlation is one network-wide common shock.** Once that shock is removed, correlation falls to about a third by hop 3 (about 1 km) at every scale. It reaches roughly zero by hop 8 on PEMS04 and about 0.01 on PEMS08.
- **So the variation splits into two parts, and neither needs depth that grows with scale.** One part is global and the same for all sensors: `uniform`, or MAGE's dense low-rank graph, already covers it. The other is local to 2–3 hops, with the same range at every scale.
- **What does change with scale is strength, not range.** That points to a per-scale mixing weight, or a per-scale split between global and local, rather than per-scale diffusion depth.
- **The premise of k = (1, 2, 4, 8) is not supported by the data.** Expected outcome: `proposed` ≈ `shared`, with `uniform` doing as well at coarse scales.

**Verdict after the pre-check.** The experiment is still valid as a test, but its hypothesis is unlikely to hold. Do not run the full 45-run grid. If the line is kept, a cheap check would be `shared` / `proposed` / `uniform` / `nomix` × 2 seeds on PEMS08, about 8 runs. Reformulate first: "coarse scale → stronger common-mode mixing, local mixing fixed at 1–3 hops" is what the data shows. Note that this is close to gating, which lowers novelty.

## Step 8 runs (2026-10-31)

**Code.**
- `MAGE_MS/streams.py`: the model, with all arms.
- `MAGE_MS/train.py --arm <arm>`: runs one arm. Without the flag, it behaves exactly as in step 7.
- `MAGE_MS/run_step8.py`: the launcher.
- `MAGE_MS/test_streams.py`: the unit tests.

**Unit tests (all pass, on the A6000).**
1. Row sums: max error 2e-7 for S⁺, S⁻ and P⁸. Rows of S_m sum to 1 − λ.
2. Factorised P^k·Z vs the dense matrix power (N = 50, float64): max relative error 1e-15.
3. Arm `mage` vs `MAGEX(encoder='linear')` with the same weights: max relative difference 3e-7. So the new code path reproduces MAGE exactly.
4. Arm `proposed`, forward + backward, N = 500 / 2000 / 8000: 93 / 128 / 461 ms per step, peak memory 0.94 / 3.4 / 12.9 GB. The log-log slope of memory vs N is 0.94, i.e. linear.

**Cheap check (launched 01:28).**
- Runs: PEMS08, T = 96 → 12, `data_w288`, same recipe as step 7, 9 runs in total:
  - arms `proposed`, `shared`, `uniform` and `nomix`, with seeds 0 and 1;
  - arm `mage`, seed 0, as a code check.
- GPUs: 3 (A6000, 5 jobs) and 0 (A4500, 4 jobs). GPUs 1 and 2 are in use by other jobs.
- Time per epoch with 9 runs sharing the GPUs: `mage` 22 s, `nomix` 29–32 s, `uniform` 44 s, `shared` 52 s, `proposed` 74–77 s. If `proposed` runs to the 300-epoch cap, it needs about 6 h.
- Comparisons:
  - `shared` vs step-7 m4 (13.854 ± 0.039): early merge vs separate streams;
  - `shared` vs linear MAGE (13.802 ± 0.044);
  - `proposed` vs `shared`: depth schedule;
  - `uniform` and `nomix`: space controls.


**Cheap-check results (finished 05:12; PEMS08, T = 96, test MAE).** The step-7 baselines for the same seeds are linear MAGE 13.745 / 13.769 and m4 13.827 / 13.899.

| arm | seed 0 | seed 1 | mean | Δ vs linear MAGE (same seed) | Δ vs m4 (same seed) | epochs run | s/epoch* | peak MB |
|---|---|---|---|---|---|---|---|---|
| `mage` (code check) | 13.736 | – | – | −0.010 | – | 300 | 21 | 883 |
| `shared` | 13.775 | 13.918 | 13.847 | +0.029 / +0.149 | −0.053 / +0.019 | 300 / 131 | 36–49 | 2453 |
| `proposed` | 13.790 | 13.671 | 13.731 | +0.045 / −0.098 | −0.037 / −0.227 | 270 / 253 | 40–45 | 3156 |
| `uniform` | 13.968 | 13.956 | 13.962 | +0.222 / +0.187 | +0.141 / +0.057 | 126 / 122 | 44 | 2153 |
| `nomix` | 13.823 | 13.817 | 13.820 | +0.078 / +0.048 | −0.004 / −0.082 | 300 / 300 | 24–27 | 1691 |

\*Time per epoch with 9 runs sharing two GPUs. Parameters: 1.93 M for every multi-stream arm, vs 1.87 M for MAGE.

**Reading (2 seeds; differences under about 0.05 are noise):**
- **The code check passes.** `mage` gives 13.736, against 13.745 from step 7 with the same seed.
- **The network-wide average hurts.** `uniform` is the worst arm, about 0.2 worse than MAGE, and worse even than no mixing. So the coarse streams should not be mixed with a plain global mean. The structured graph matters. This supports keeping the graph, against the reading of the pre-check that "coarse = global mean".
- **Spatial mixing adds little on PEMS08 at T = 96.** `nomix`, with no sensor mixing at all, is only 0.05–0.08 worse than MAGE. This answers step 7b for this setting: MAGE's graph is worth about 0.06 MAE here.
- **`proposed` vs `shared` (2 seeds; see 5-seed result below).** Seed 1 favours `proposed` by 0.25, but `shared` seed 1 stopped early at epoch 131, which suggests a bad run. Seed 0 is a tie (+0.015).
- **`proposed` vs linear MAGE:** a mean of 13.731 vs 13.757, i.e. −0.026, better in 1 of 2 seeds. Not significant yet.
- **Separate streams vs early merge (`shared` vs m4):** −0.053 / +0.019. Undecided.
- **Early stopping looks unstable.** `uniform` (both seeds) and `shared` seed 1 stopped at 122–131 epochs, while the others ran to about 300. This adds seed noise. If results stay close, consider a longer patience.

**Stage 2 launched 05:13.** Seeds 2–4 for `proposed` and `shared` (6 runs; GPU 3 ×3, GPU 0 ×3), following the staging rule: `proposed` was not clearly dead. With 5 seeds, C1 (`proposed` vs `shared`) and C4 (vs linear MAGE, 5 seeds) can be decided.

**Final 5-seed result (stage 2 finished 09:33; PEMS08, T = 96, test MAE).**

| arm | seeds 0–4 | mean ± sd | Δ vs linear MAGE (paired) | Δ vs m4 (paired) | epochs run |
|---|---|---|---|---|---|
| linear MAGE (step 7) | 13.745 / 13.769 / 13.845 / 13.843 / 13.807 | 13.802 ± 0.044 | – | – | 300 ×4, 1 early |
| `shared` | 13.775 / 13.918 / 13.794 / 13.767 / 13.679 | 13.787 ± 0.086 | −0.015 ± 0.108 (3/5) | −0.067 ± 0.087 (**4/5**) | 300 ×4, 131 |
| `proposed` | 13.790 / 13.671 / 13.836 / 13.887 / 13.666 | 13.770 ± 0.099 | −0.032 ± 0.084 (3/5) | −0.084 ± 0.130 (3/5) | 300, 270, 253, 206, 130 |

`proposed` − `shared`, paired: +0.015 / −0.247 / +0.042 / +0.120 / −0.013, mean −0.017 ± 0.138 (2/5 better).

Mean MAE at horizons 3 / 6 / 12, with RMSE and MAPE:

| | h = 3 | h = 6 | h = 12 | RMSE | MAPE |
|---|---|---|---|---|---|
| linear | 13.00 | 13.79 | 15.03 | 22.79 | 9.20 |
| `shared` | 12.97 | 13.80 | 15.00 | 23.02 | 9.25 |
| `proposed` | 12.96 | 13.77 | 14.98 | 22.87 | 9.16 |

**Decisions (rule: lower mean and ≥ 4/5 paired seeds):**
- **C1, depth schedule (`proposed` vs `shared`): fails** (−0.017, 2/5). Mixing coarser scales further does not help. This matches the pre-check: correlation is not longer-range at coarse scales.
- **C4, vs linear MAGE: fails for both arms** (−0.03 and −0.02, 3/5). The means are lower but well inside seed noise. RMSE is slightly worse.
- **Separate streams vs early merge (`shared` vs m4): passes** (−0.067, 4/5). Keeping scales apart through the blocks is better than summing them in the encoder, but that only brings multi-scale back to MAGE's level, not past it.
- **Variance doubled.** The multi-stream arms have seed sd 0.09–0.10 vs 0.044 for MAGE, partly from early stopping at 130–206 epochs in some runs. A small true gain (≤ 0.03) cannot be ruled in or out without more seeds or a fixed training length.

**Conclusion of step 8 (PEMS08).** Scale-dependent diffusion depth does not work. Separate scale streams are the only part with a signal, and only against the early-merge design. Together with `nomix` (graph worth about 0.06) and `uniform` (global mean hurts), the PEMS08 picture is: at T = 96, neither the temporal multi-scale design nor the spatial operator is where the remaining error is. Do not move to PEMS04 / LargeST with this design.

## Step 8 follow-up (2026-11-02, after the reviewer comments; summarised in `experiment_log.md`, "Archived documents")

Results and tables are in `experiment_log.md` (2026-11-02). Decisions:
- **C2 (`proposed` vs `reversed`) fails:** +0.012, 2/5. With C1 already failing, scale-dependent depth is a clean negative on PEMS08.
- **C3 (`proposed` vs pure uniform `uniform0`):** −0.088, 3/5; `shared` vs `uniform0` −0.071, 4/5. A learned graph beats the plain network mean, by less than the old confounded `uniform` arm suggested.
- **Leverage:** the λS⁻ term dominates blocks 1 and 3 (power term 20–35 % of the operator output) and the schedule moves the output by 2–8 % there. A fairer test needs λ bounded (e.g. sigmoid, as the reviewer suggested) or S⁻ removed.
- **Where space matters:** plain MAGE loses 0.4 MAE without spatial mixing on PEMS04 and PEMS07, 0.22 on PEMS08 (T = 96). PEMS04, the proposal's primary dataset, is where any further spatial test should run. Multi-scale streams appear to substitute for spatial mixing on PEMS08 (needs PEMS04 to confirm).
- Code: `streams.py` arms `uniform0` and `nomix1`; `prep.py` horizon argument; `diag_lambda.py`.

# Overview: approaches tried so far (status 2026-11-07)

High-level summary of every direction tested since 2026-10-18: the idea, the final result, and the conclusion. Per-run tables are in `experiment_log.md`. Unless stated otherwise: test MAE, lower is better; differences under ~0.05 MAE are seed noise; "A beats B" means a lower mean and better in ≥ 4/5 paired seeds.

## Settled

| # | Approach | Idea | Final result | Conclusion |
|---|---|---|---|---|
| 1 | Multi-scale temporal encoders (models 2–5, step 7) | Feed MAGE pooled views of the input window (gated, nonlinear) instead of one linear projection | Never beats linear MAGE at T = 12 / 96 / 288 on PEMS04 / 08; nonlinear encoders overfit at long windows | **Negative** |
| 2 | Scale-dependent spatial mixing depth (original proposal, step 8; summary in `experiment_log.md`) | Coarser time scales mix further over one shared graph, depths (1, 2, 4, 8) | PEMS08, 5 seeds: Proposed vs Shared −0.017 (2/5), vs Reversed +0.012 (2/5), vs MAGE −0.032 (3/5). PEMS04 screen (1 seed): fails the go rule | **Negative.** Direction and depth do not matter. Data pre-check: coarse scales are more strongly correlated across sensors, but not over longer road distances |
| 3 | Bounded λ | MAGE's λ is unconstrained (2–10, flips sign across seeds); bound one λ per expert to (0, 1) with a sigmoid | PEMS04, 5 seeds: 18.275 → 18.232 (−0.043, better 5/5), RMSE −0.7 %, lower seed variance | **Small, consistent gain**; too small to stand alone |
| 4 | Separate scale streams (+ bounded λ) | Keep the four time scales apart through MAGE's blocks | PEMS08: beats merging scales at the input (−0.067, 4/5) but only ties MAGE. PEMS04 (1 seed): −0.10 to −0.18 vs MAGE | **Unconfirmed** (1 seed on PEMS04) |
| 5 | Where does spatial mixing matter? | MAGE vs the same model with spatial mixing switched off, 4 datasets | Worth +0.4 MAE on PEMS04 / 07, +0.22 on PEMS08, +0.1–0.4 on PEMS03; ≈ 0 at 4 h ahead | Space matters modestly; mostly the network-wide common signal |
| 6 | "Long history replaces space" | Spatial mixing matters less when the input window is long | PEMS08 goes the other way (0.09 at T = 12, 0.22 at T = 96); PEMS03 noisy | **Not supported** |
| 7 | Graph-collapse fix | Stop MAGE's graph from collapsing (spread-out init, no weight decay on the graph) | +1.3 to +2.2 MAE on PEMS04 / 08: the graph stays structured but random (no road structure) and overfits | **Negative**; the collapse acts as a safeguard |
| 8 | Rank hierarchy (general / time / sensor levels) | Encoder and decoders = full-rank general part + rank-32 time-of-day part + rank-32 per-sensor part | 1 seed: PEMS08 +0.147, PEMS04 +0.044 vs MAGE | **No gain** on full data |
| 9 | Data check: do long windows see rush-hour changes? | Does an 8-hour window predict deviations from the usual pattern better than 1 hour? | Rush hours vary day to day (heavy morning → heavy evening, r = 0.40–0.57), but the last hour already shows it: 8 h vs 1 h adds ≤ 0.02 R² on all four PEMS sets | Long-window / multi-scale designs have little to gain on PEMS; explains rows 1, 2 and 8 |

## Open: what MAGE's graph experts actually do (not concluded)

| # | Question | Evidence so far | Status |
|---|---|---|---|
| 10 | Do MAGE's graph experts collapse? | In every trained MAGE, including the official code at the paper's setting, the graph parameters shrink to ~0 and every expert becomes the plain sensor average (entropy 1.000). Replacing them by the exact average changes test MAE by ≤ 0.001 | Strong for MAGE |
| 11 | How many of MAGE's parameters do useful work? | The collapsed graph experts are 56–69 % of the parameters. A slim MAGE (graph experts replaced by one average term per block) matches MAGE (−0.100, −0.003, +0.006 on 3 runs) with 62–73 % fewer parameters | **Open**: 1–2 seeds at T = 96 only; needs 5 seeds in MAGE's own 12 → 12 setting on PEMS03 / 04 / 07 / 08, and clean speed / memory measurements |
| 12 | Is the collapse general across adaptive-graph models? | All 10 runs (Graph WaveNet, AGCRN, MTGNN, STAEformer, ASTGCN × PEMS04 / 08, 1 seed). None collapses exactly; AGCRN comes close (uniformity 0.997), Graph WaveNet / MTGNN are near-uniform (0.96–0.97), the attention models are not (0.74–0.89). All rely on their learned mixing (swap → +0.4 to +28.9 MAE, vs ≤ 0.001 for MAGE). Road structure: at chance for most; weak for AGCRN (PEMS04) and MTGNN (1.2–2.8× a random pair) | **Open**: exact collapse is MAGE-specific so far; "learned graphs barely reflect roads" holds for most models; whether their learned mixing is *needed* requires retraining with the average (+ a small low-rank graph) |
| 13 | What do learned graphs encode, if not roads? | Over all pairs: nothing measured (roads, daily pattern, commute direction, volume: \|ρ\| ≤ 0.1 mostly). Strongest links: co-fluctuating sensors for the embedding-based graphs (AGCRN, MTGNN, MAGE graph fix; 66–78th percentile); random for Graph WaveNet. All learned graphs are hub-dominated (25–68 % of top-5 links go to 5 % of sensors; 80–100 % for attention). Hubs: sensors that track the network-wide signal (MTGNN, MAGE graph fix, AGCRN on PEMS08), high-volume sensors (Graph WaveNet), "sink" sensors (attention) | **Open, promising**: suggests learned spatial modules mostly compute a global summary (mean or hubs), not road structure; 1 seed, needs a seed check and the retrain test (row 12) |
| 14 | How much of each model's spatial gain does the plain average capture? (retrained from scratch) | AGCRN 100–105 %, MTGNN > 100 % (learned graph no better than the average), MAGE ~100 % (slim). STAEformer: spatial attention gives no gain on PEMS08, average captures 71 % on PEMS04. Graph WaveNet 26–47 %: its learned graph is genuinely needed (+1.2 / +2.7 MAE over the average) despite not following roads. ASTGCN 15–25 % (road-based attention, not comparable) | **Open**: 1 seed; strongly suggests that for most adaptive-graph models the average is all that is needed; Graph WaveNet is the exception to explain |
| 15 | Can a cheaper model beat MAGE, and can the size be chosen automatically? | Slim MAGE (average only) is the best variant. Adding a fixed co-fluctuation graph, the road graph, or a learned rank-2 / 8 / 32 graph gives no consistent gain. **Method A** (grow from the average while validation improves) selects rank 0 on both datasets, training only two small models. **Method B** (detect collapse after epoch 1, switch to slim) works mechanically but ends +0.04 / +0.07 above MAGE | **Open**: 1 seed; slim MAGE = same accuracy at 27–38 % of the parameters, and method A recovers it automatically |
| 16 | RQ: does a learned graph beat the network average only for sensors with local dependence beyond the common signal? | Per-sensor test on saved models (no training): gain from the learned graph is unrelated to leftover local dependence (\|ρ\| ≤ 0.14). Where the graph helps (Graph WaveNet, ASTGCN, the two models without sensor-identity embeddings) it helps most on sensors that strongly follow the network signal (ρ = +0.26 to +0.65) | **Hypothesis not supported**; points instead to "the graph helps only when the model lacks sensor identity" (identity test pending) |
| 17 | What does a sensor-identity embedding represent, and why does a learned graph beat the average without one? | Identity parameters encode each sensor's typical behaviour (volume, daily shape, commute direction, loading on the network signal; R² 0.5–0.93) in MAGE, STAEformer, MTGNN (in its normalisation parameters) and AGCRN (PEMS04). Graph WaveNet has none: its graph vectors and rows encode no sensor properties (R² ≈ 0) | **Open**: working explanation is that the learned graph makes otherwise identical sensors distinguishable; to test with average + sensor embedding and with a fixed random graph |
| 18 | Identity test: does the average + a sensor embedding match the learned graph where the learned graph won? | Graph WaveNet: yes, better than learned on both datasets (14.67 vs 14.78, 18.80 vs 19.14); a fixed random graph closes 55–64 % of the gap. ASTGCN: equal on PEMS08, 47 % of the gap on PEMS04 (its attention sits on road-graph filters) | **Supported for 5 of 6 models** (1 seed): with sensor identity, the network average is all the spatial mixing needed; exception ASTGCN on PEMS04 |

## Current conclusion

- Neither multi-scale temporal designs nor scale-dependent spatial mixing improve MAGE on PEMS. The data explains why: the last hour already carries the information longer windows add, and spatial mixing mostly carries a network-wide common signal.
- MAGE's graph experts collapse to a plain average; this is the strongest finding so far, and the cross-model check shows it is specific to MAGE: the other four models do not collapse exactly and rely on their learned mixing, although most of it barely reflects roads. How much of each model's spatial part is actually needed is still open (rows 11–12).
- The paper framing depends on rows 12–15 (working title: "What do adaptive graphs learn?"): a general "learned graphs do not learn roads / collapse" study if the other models agree, otherwise a MAGE-specific diagnosis with a slim model.

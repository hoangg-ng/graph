# Adaptive graphs learn the mean: sensor identity, not graph structure, drives spatial gains in traffic forecasting

Most recent traffic forecasters learn an adaptive graph between sensors. We ask **when such a learned graph actually beats a simple network-wide average of all sensors**. Across six published models on PEMS04 and PEMS08, the answer is: only when the model has no other way to tell sensors apart. With a sensor-identity representation, the plain average matches the learned graph in five of six models, and in MAGE (NeurIPS 2025) the learned graph collapses to that average on its own. We show that sensor identity is a compact profile of each sensor's typical behaviour, which can be computed directly from its data. This leads to a **learning-free spatial module**: a fixed network-wide average plus a computed sensor profile, with no learned graph and no per-sensor parameters. It matches MAGE with about 70 % fewer parameters and forecasts better for sensors never seen in training.

Setting: PEMS04 (307 sensors, 59 days) and PEMS08 (170 sensors, 62 days), traffic flow every 5 minutes, chronological 6:2:2 split, test MAE (lower is better), one seed per configuration (differences under about 0.05–0.08 MAE are noise). Per-run tables: [`experiment_log.md`](experiment_log.md); full history: [`research_design.md`](research_design.md).

---

## 1. Sensor identity: what it is and why a model needs it

**The data.** Each dataset is one array, *time steps × sensors × measurements* (flow, occupancy, speed), plus a list of road connections. A sensor is only a column index: the data carry no location, road type or direction.

**The problem: indistinguishable sensors.** Two sensors can show almost the same recent history and still have different futures. PEMS08, Tuesday 12 July 2016, 07:00 (flow every 15 minutes):

| | last hour | next hour (actual) |
|---|---|---|
| sensor 31 | 206, 235, 260, 285 | 263, 286, 289, **337** (keeps rising) |
| sensor 148 | 216, 235, 260, 276 | 270, 306, 291, **251** (peaks, then falls) |

A model that sees only recent values and the time of day receives the same input for both and must give both the same forecast.

**Sensor identity** is any information that tells the model which sensor it is looking at and how that sensor usually behaves. Models obtain it in three ways:

| Source of identity | Example models | For a new sensor? |
|---|---|---|
| a learned embedding per sensor | MAGE, STAEformer, STID | no (untrained) |
| other per-sensor parameters, or a learned graph that mixes different sensors for each sensor | MTGNN (normalisation layers), Graph WaveNet (its graph) | no |
| **computed from the sensor's own data (ours)** | – | **yes** |

**What identity encodes.** Predicting sensor properties from the learned per-sensor parameters (cross-validated R², PEMS08 / PEMS04):

| learned per-sensor parameters | traffic volume | daily shape | commute direction (AM vs PM peak) | follows the network signal |
|---|---|---|---|---|
| MAGE sensor embedding | 0.77 / 0.93 | 0.78 / 0.87 | 0.15 / 0.69 | 0.51 / 0.56 |
| STAEformer adaptive embedding | 0.78 / 0.92 | 0.53 / 0.88 | 0.53 / 0.71 | 0.64 / 0.66 |
| MTGNN normalisation parameters | 0.89 / 0.93 | 0.65 / 0.78 | 0.20 / 0.57 | 0.66 / 0.61 |
| Graph WaveNet graph vectors | 0.21 / 0.03 | 0.17 / 0.09 | ≈ 0 / 0.19 | ≈ 0 |

Identity is a **profile of the sensor's typical behaviour**: how busy it is, its daily shape, whether it is a morning or evening commuter road, and how strongly it moves with the whole network. In the example above, the profile separates the two sensors: sensor 148 normally peaks early in the morning (its peak is ending at 07:00), while sensor 31 is an evening-heavy road still ramping up.

**What identity contributes.** Removing it from a model with a fixed spatial average costs +0.82 / +1.68 MAE (PEMS08 / PEMS04), more than any spatial design choice we tested.

## 2. Finding: a learned graph is needed only when identity is missing

Each model was trained from scratch with spatial mixing **off**, fixed to the **network-wide average** (every sensor receives the mean of all sensors), or **learned** (12 → 12 steps; MAGE via a slim variant, 96 → 12):

| Model | identity? | no mixing | average | learned | learned beats average? |
|---|---|---|---|---|---|
| MAGE | yes | – | 13.79 / 18.19 | 13.79 / 18.29 | **no** (its graph collapses to the average) |
| AGCRN | yes | 18.96 / 23.47 | **15.64 / 19.43** | 15.79 / 19.41 | **no** |
| MTGNN | yes | 15.44 / 19.14 | **15.11 / 19.00** | 15.24 / 19.18 | **no** |
| STAEformer | yes | 13.38 / 18.57 | 13.41 / 18.35 | 13.47 / 18.26 | **no** / slightly |
| Graph WaveNet | **no** | 17.04 / 22.77 | 15.98 / 21.83 | 14.78 / 19.14 | **yes** |
| ASTGCN | **no** | 20.97 / 26.33 | 20.50 / 25.70 | 19.05 / 22.04 | **yes** |

**Intervention: give the two exceptions identity, keep the plain average.**

| Test MAE (PEMS08 / PEMS04) | Graph WaveNet | ASTGCN |
|---|---|---|
| learned graph | 14.78 / 19.14 | 19.05 / 22.04 |
| average | 15.98 / 21.83 | 20.50 / 25.70 |
| **average + identity** | **14.67 / 18.80** (beats learned) | **19.04** / 24.00 (equal; half the gap) |
| fixed random graph, never trained | 15.31 / 20.11 | 21.22 / 26.51 |

In Graph WaveNet the learned graph was compensating for missing identity; even a random graph, which only makes sensors look different from each other, recovers 55–64 % of its gain. ASTGCN keeps part of its gain on PEMS04 because its attention operates on the real road graph.

**Supporting evidence.**
- **MAGE's graph collapses to the average.** In every trained MAGE, including the unmodified official code, its 16 graph experts become the plain sensor average within the first epoch; replacing them by the exact average changes MAE by ≤ 0.001, so 56–69 % of its parameters do no work. Preventing the collapse makes the graph random and the model overfits (+1.3 to +2.2 MAE).
- **Learned graphs do not learn roads.** Road neighbours get 0.9–1.2× the weight of a random pair in most models; a few hub sensors dominate the links.
- **Local structure does not explain the gain.** Dependence left after removing the network average does not predict where a learned graph helps (|ρ| ≤ 0.14 per sensor).

**Conclusion.** In these benchmarks, spatial modelling reduces to two parts: a **network-wide common signal** and **each sensor's identity**. Given identity, a learned graph is unnecessary, with one exception: attention built on the real road graph (ASTGCN on PEMS04).

## 3. Proposed: a learning-free spatial module

Both parts can be provided without learning a graph or any per-sensor parameter:

1. **Fixed aggregation:** the network-wide average of all sensors, computed once per time step and shared by every sensor (one learned strength per layer).
2. **Computed identity:** a 52-number profile per sensor from its training data (hourly weekday and weekend shape, volume, morning/evening peak ratio, loading on the network average), passed through one shared linear layer.

Each sensor is then forecast from its own history, the network average and its profile. The forecasting backbone is still trained; the spatial module and the identity involve no learned graph and no per-sensor parameters. Implemented on slim MAGE (`MAGE_MS/streams.py`, arm `lite`, `--identity profile`), 96 → 12 steps:

| PEMS08 / PEMS04 | MAGE | average + learned identity | average, no identity | **average + computed identity** |
|---|---|---|---|---|
| test MAE | 13.79 / 18.29 | 13.79 / 18.19 | 14.61 / 19.87 | **13.85 / 18.25** |
| parameters | 1.87 M / 2.73 M | 0.72 M / 0.74 M | 0.70 M | **0.70 M** + 7 k shared |
| learned per-sensor parameters | ~6,300 per sensor | 128 per sensor | 0 | **0** |

The computed profile recovers 92–96 % of the value of a learned embedding and is within noise of MAGE.

**Sensors never seen in training** (20 % held out of the loss, the network average and model selection; their profile uses their history from the training period):

| MAE, PEMS08 / PEMS04 | learned identity | no identity | **computed identity** |
|---|---|---|---|
| unseen sensors | 22.59 / 32.22 | 16.95 / 21.79 | **16.61 / 20.52** |
| seen sensors | 13.61 / 18.08 | 14.17 / 19.71 | 13.61 / 18.20 |

A learned embedding fails on new sensors, worse than having no identity. The computed profile is the only option that is accurate on seen sensors and best on unseen ones (2 % / 6 % lower error than no identity).

## 4. Related work and novelty

| Work | Claim | Relation to ours |
|---|---|---|
| STID (CIKM 2022), STAEformer (CIKM 2023) | identity embeddings make simple models strong; sensors must be distinguishable | we show *why learned graphs help* in existing models (missing identity), confirm it by intervention, and measure what identity encodes |
| U-Trans (arXiv 2026) | a uniform average matches spatial attention | same aggregation; we explain when it suffices and the exceptions |
| FAF (ICML 2026), SGP (AAAI 2023), SimST | fixed neighbourhood aggregation replaces message passing | in traffic a **global** average suffices; road neighbourhoods added nothing on top |
| DeepGLO (NeurIPS 2019) | global factors + per-series models | same family; reached from what learned graphs actually do |
| SCPT, MoGERNN (2025) | forecasting for unseen roads / locations | SCPT learns a history encoder with pre-training; we use a few computed statistics, no pre-training |
| Common Correlated Effects (Pesaran 2006) | cross-sectional averages proxy common factors | the collapsed adaptive graph computes exactly this average |

The ingredients exist separately. The contribution is the **diagnosis**: in published adaptive-graph forecasters, the learned graph substitutes for sensor identity on top of a network-wide average, shown by intervention across six models, which yields a simple learning-free alternative.

## 5. Limitations

One seed; two datasets (traffic flow only); the five comparison models are reimplemented from the official code; the proposed module is tested inside slim MAGE at 96 → 12 steps; the unseen-sensor profile requires some history for the new sensor. PEMS at a 1-hour horizon is close to its noise floor. Next: 3–5 seeds, PEMS03/07 and LargeST, MAGE's 12 → 12 setting, comparison with SCPT.

---

## Appendix

**How we got here.** The original proposal (summarised in [`experiment_log.md`](experiment_log.md), "Archived documents") gave MAGE several time scales and let coarser scales mix further across the sensor graph. It failed (PEMS08, 5 seeds: ties with uniform and reversed mixing), because coarser scales are more correlated across sensors but not over longer road distances: most shared signal is one network-wide shock. Investigating that shock led to the collapse finding and then to the identity explanation.

**Other directions tried** (negative or inconclusive): multi-scale encoders and longer windows; separate scale streams; bounded λ (−0.04 MAE); co-fluctuation, road or learned low-rank graphs on top of the average; a general / time / sensor rank hierarchy; preventing MAGE's collapse. Growing a learned graph from the average only while validation improves selected the average alone.

**Code.**

| What | Where |
|---|---|
| Slim MAGE, computed identity, unseen-sensor test | `MAGE_MS/streams.py`, `MAGE_MS/train.py`, `run_identity_mage.sh` |
| Other models (reimplemented) and mixing / identity variants | `MAGE_MS/others/reimpl*.py`, `train_other*.py`, `run_identity.sh` |
| What identity encodes; graph analyses; per-sensor test | `MAGE_MS/others/identity_check.py`, `what_graphs_learn.py`, `rq_test.py` |
| Which MAGE parameters do work | `MAGE_MS/diag_useful.py` |

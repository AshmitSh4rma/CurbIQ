# Model Card — CurbIQ Spatiotemporal Violation Forecaster

**Model:** LightGBM gradient-boosted trees, Poisson objective
**Task:** next-day count of recorded parking violations per H3 res-8 cell
**Implementation:** [`curbiq/forecast.py`](../curbiq/forecast.py) · params in [`curbiq/config.py`](../curbiq/config.py) (`LGBM_PARAMS`)
**License:** Apache-2.0 · **Decision brief:** [`RESEARCH.md`](RESEARCH.md)

> **Honest framing (non-negotiable).** The target is *recorded* violations, which reflect **enforcement activity** (where patrols went), not true violation occurrence. The model forecasts where recorded violations will concentrate — useful for routing the next patrol — and must be read alongside the exposure-adjusted ranking and under-enforcement blind spots in [`prioritize.py`](../curbiq/prioritize.py). See Ethical Considerations.

---

## 1. Intended use & users

- **Intended use:** tactical, short-horizon (next-day) prediction of where recorded parking violations will cluster, to help BTP-style enforcement allocate the next shift's patrols and tow vehicles. Drives the `forecast_area` signal that feeds the priority ranking.
- **Intended users:** traffic-enforcement planners / analysts working from an interactive dashboard, and hackathon/audit reviewers assessing technical rigor. The model is a *decision-support* signal, never an automated enforcement trigger.
- **Scale of operation:** one city (Bengaluru), the ever-active H3 res-8 cell grid, daily cadence.

## 2. Out-of-scope uses

- Predicting **true** parking-violation occurrence or compliance (the data is enforcement events, not ground truth).
- Per-vehicle, per-person, or per-plate prediction — the model is cell-aggregate only and plates never reach it.
- Punitive, automated, or individual-level decisions; legal/financial penalty determination.
- Long-horizon forecasting (the model predicts one day ahead), transfer to other cities without retraining, or measuring congestion/delay (that is the *modeled* CIS, a separate component).

---

## 3. Training data

- **Source:** Bengaluru Traffic Police parking-violation export (anonymized), Nov 2023 – Apr 2024. **298,445** records (298,125 counted after dropping duplicates), 54 police stations, 168 named junctions.
- **Panel construction (`build_panel`):** full cartesian product of *ever-active* H3 res-8 cells × every day in the window, left-joined to daily counts. **Absent (cell, day) pairs are materialized as explicit zeros** — without this, the absence signal is lost and PAI is meaningless.
- **Spatial unit:** H3 **res 8** (coarser/denser than the res-9 hotspot grid, which gives the forecaster more signal per cell). The res-8 prediction is mapped down to res-9 cells via H3 parent in prioritization.
- **Time unit:** daily, after **UTC→IST conversion** (done in the ETL; a 5.5h shift would otherwise corrupt day-of-week and peak structure).
- **Filtering:** only `is_counted` rows (duplicates excluded); validation status is carried as a confidence weight upstream.

## 4. Features (`make_features`)

All temporal features are strictly **past-only** (`shift(1)` / `closed='left'`) to prevent leakage.

- **Lags** of the count: `[1, 2, 3, 7, 14, 28]` days.
- **Rolling** mean & std over `[3, 7, 14, 28]`-day windows (computed on the shifted series); **EWM** mean, halflife 7.
- **Lagged spatial neighbour activity:** mean count over the H3 k-ring (k=1 and k=2), as both a 1-day lag and a 7-day rolling mean (computed via an adjacency matmul on the wide grid).
- **Cyclical calendar:** `dow_sin/dow_cos`, day-of-month, month number, `is_weekend`, `is_holiday` (hardcoded India/Karnataka holidays within the window).
- **Peak-hour & heavy-vehicle load (new):** four strictly-past, leakage-safe (`shift(1)`) cell-level features — `peak_lag1` and `peak_roll7` (per-cell peak-hour event counts, IST 08–11 & 17–21) and `heavy_lag1` and `heavy_roll7` (heavy-vehicle load over the LCV/bus/HGV categories in `config.HEAVY_VEHICLE_CATEGORIES`). These realize the long-requested peak_hour_count / heavy_vehicle_count predictors and carry real signal: the peak-hour & heavy-vehicle lags all contribute (e.g. `peak_roll7` importance **1018**, a top-tier feature). Their addition takes the model from **32 to 36 features** and lifts holdout R² **0.64 → 0.65**.
- **Static per-cell:** centroid `lat`/`lon`, mean road-class loss, junction-cell fraction, number of distinct police stations.

## 5. Objective & configuration

- **Objective:** `poisson` (the counts are overdispersed and zero-inflated); `metric='poisson'`. Tweedie (`variance_power≈1.2`) is the documented fallback for the zero-heavier junction grid.
- **Key params:** `n_estimators=4000` (cap; early stopping picks ~516), `learning_rate=0.03`, `num_leaves=63`, `min_child_samples=100`, `feature_fraction=0.8`, `bagging_fraction=0.8 / freq=1`, `lambda_l1=0.5`, `lambda_l2=1.0`, early stopping = 100 rounds on the validation fold. Tuned on walk-forward CV.

---

## 6. Evaluation protocol — walk-forward, no leakage

We **never use random K-fold** (future would leak into the past). Validation is **expanding-window walk-forward by calendar month**:

- For each month from the 4th onward (≥ 3 months of training history first), train on all days strictly before the validation month, validate on that month.
- An **embargo gap** separates train and validation (`walk_forward` default 7 days; `config.FORECAST_EMBARGO = 28` documents the principle that the gap must be ≥ the longest rolling window and ≥ the horizon — the rolling/lag features reach back 28 days).
- The **final model** trains on all months but the last and is evaluated on the **Apr-2024 holdout**; the next-day forecast then uses each cell's last-available-day features.
- Leakage controls: panel zeros materialized; every lag/rolling/EWM/neighbour feature `shift(1)`/`closed='left'`; predictions clipped to ≥ 0.

Metrics (`score_block`) are computed per evaluation day and averaged. PAI/PEI rank cells by **predicted** density; because H3 cells are equal-area, count == density.
- `PAI@k = (fraction of violations captured in the top-k% predicted cells) / k`
- `PEI@k = PAI@k / oracle-PAI@k`, where the oracle ranks by **observed** count (PEI ≤ 1; how close to a perfect ranker).

---

## 7. Metrics (verified)

Holdout = April 2024; CV = mean over expanding walk-forward folds.

| Metric | Walk-forward CV (mean) | Apr-2024 holdout |
|---|---|---|
| MAE | 2.05 | 2.02 |
| RMSE | 7.02 | 6.70 |
| R² | 0.60 | **0.65** |
| Mean Poisson deviance | 3.44 | 3.48 |
| ROC-AUC (top-10% hotspot label) | **0.929** | 0.925 |
| PR-AUC | 0.677 | 0.672 |
| PAI@5% | 12.38 | **12.72** |
| PEI@5% | 0.807 | 0.825 |
| PAI@20% | 4.42 | 4.39 |
| PEI@20% | 0.892 | 0.885 |

> Metrics are reproducible (LightGBM `random_state=42`, `deterministic=True`) and
> are emitted verbatim to `data/artifacts/model_metrics.json` by `build_all.py`.

Interpretation: the top **5%** of predicted cells capture ~12.7× their areal share of next-day violations (83% of an oracle's capture); the top **20%** capture ~4.4×. ROC-AUC ≈0.93 means the model separates top-decile hotspot-days from the rest very well. (Holdout = the most-recent month, the deployment-relevant estimate; the CV mean is reported alongside for honesty.)

## 8. Baseline comparison

All baselines evaluated on the **same Apr-2024 holdout** (`baselines`):

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Poisson dev ↓ | ROC-AUC ↑ | PAI@5% | PEI@5% |
|---|---|---|---|---|---|---|---|
| same-weekday-last-week (`lag7`) | 2.50 | 9.44 | 0.31 | 17.86 | 0.80 | 11.32 | 0.735 |
| rolling-7 mean (`rmean7`) | 2.10 | 7.07 | 0.61 | 5.19 | 0.904 | 12.37 | 0.804 |
| EWM (halflife 7) | 2.05 | 6.88 | 0.63 | 3.61 | 0.920 | 12.54 | 0.814 |
| **LightGBM Poisson (36-feature)** | **2.02** | **6.70** | **0.65** | **3.48** | **0.925** | **12.72** | **0.825** |

**Honest read:** LightGBM clearly beats the naive last-week baseline across every metric (MAE 2.02 vs 2.50, Poisson deviance 3.48 vs 17.86, AUC 0.925 vs 0.80) and edges the strong EWM-persistence baseline on RMSE/deviance/AUC and **narrowly on PAI@5 too (12.72 vs 12.54)** — the margin is modest because dominant hotspots are inherently *persistent* and a good persistence model captures most of the ranking lift. The model's real edge is calibrated counts (deviance/MAE/AUC), responsiveness to neighbour / calendar / peak-hour & heavy-vehicle structure, and a principled probabilistic output.

## 9. External feature enrichment

Two external sources were tested as forecast features: daily **weather** (Open-Meteo) and **metro-station proximity** (OSM / Overpass — 81 Namma Metro stations), evaluated holdout vs CV under the identical protocol.

**Metro.** Nearest-metro distance is a clean enrichment win again under the tuned model: it lifts holdout **PAI@5 from 12.60 (34-feature) to 12.72 (36-feature)** and is a **top-8 feature by importance (1268)** (R² ~flat). Retained on by default; toggle via `enrich`. (81 Namma Metro stations via OSM/Overpass.)

**Weather.** Daily weather nudges holdout MAE/R² but clearly **hurts CV** (noisier in the dry-season folds), so it is **off by default** — pass `enrich="all"` to include it. Enrichment degrades gracefully offline (metro falls back to a built-in station list).

---

## 10. Limitations

- **Persistence-bound PAI.** Because hotspots are stable, the ranking lift over a strong EWM baseline is small; the model earns its keep on calibration and AUC, not PAI.
- **Short horizon.** Forecasts one day ahead from the last available day; multi-day/seasonal forecasting is out of scope.
- **Window length.** Only ~5 months of data → limited seasonal coverage; holidays are hardcoded for the window, not generalized.
- **Spatial granularity trade-off.** Res-8 maximizes forecast signal but is coarser than the res-9 hotspot grid; the parent-cell mapping is approximate.
- **R² ~0.6 with high RMSE** reflects a heavy-tailed count distribution — a few very high-count cells dominate squared error.
- **Single city, single vintage.** No transfer guarantee; must be retrained per city/period.

## 10. Ethical considerations

- **Enforcement bias / feedback loop (the central hazard).** The label is *recorded* violations = patrol activity. Naively acting on the forecast risks a predictive-policing loop: patrol → record → forecast says "hotspot" → patrol again. CurbIQ mitigates this by **not** ranking on raw counts in production: prioritization uses an **exposure-adjusted** rate and publishes the raw↔adjusted Spearman rank-shift (≈0.98 here, evidence the top hotspots are genuine, not patrol artifacts), and it surfaces **under-enforcement blind spots** (408 cells with high modeled propensity but low observed enforcement) as a first-class output — low-count zones are never presented as "compliant."
- **Temporal blind spot made visible.** The fairness layer shows evening-peak (17–20h IST) enforcement share ≈ **0.2%** despite high congestion risk — a direct artifact of patrol scheduling the forecast alone would perpetuate.
- **No individual targeting.** Cell-aggregate only; plates are SHA-256+salted and never reach the model; public layers are H3-only with `count<5` suppression (see [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md)).
- **Decision support, not automation.** Outputs inform human patrol planning; they are not an automated enforcement or penalty mechanism.

---

# Model Card — CurbIQ Hotspot-Emergence Classifier

**Model:** LightGBM gradient-boosted trees, binary objective
**Task:** probability that an H3 cell that is *not currently a hotspot* will *become* one within the next 28 days
**Implementation:** [`curbiq/emergence.py`](../curbiq/emergence.py) · artifact `emergence.json` · API `/api/emergence`
**License:** Apache-2.0

> **Honest framing (non-negotiable).** Same as the forecaster: the target is built from *recorded* violations, which reflect **enforcement activity**, not true occurrence. This model says where recorded violations are likely to *newly concentrate*, and must be read alongside the exposure-adjusted ranking and under-enforcement blind spots. See Ethical considerations.

## 1. Intended use & users

- **Intended use:** forward-looking complement to the existing current-state Gi\* hotspots (where it is hot *now*) and the descriptive Mann-Kendall trend typing (how cells have trended). The classifier flags cells likely to *emerge* as hotspots in the next 28 days, so planners can watch or pre-position resources before a cluster forms.
- **Intended users:** the same traffic-enforcement planners/analysts and audit reviewers as the forecaster. A *decision-support* watch-list signal, never an automated enforcement trigger.
- **Scale of operation:** one city (Bengaluru), the strictly-past forecast panel grid, 28-day forward horizon.

## 2. Out-of-scope uses

- Predicting **true** emergence of illegal parking (the label is built from enforcement events, not ground truth).
- Per-vehicle / per-person / per-plate prediction; punitive, automated, or individual-level decisions.
- Treating the absolute risk score as a calibrated probability of occurrence — the base rate is ~1.5%, so bands and flags are **percentile-based**, not absolute-threshold-based (see Limitations).

## 3. Objective, features & label

- **Objective:** `binary` classification on the **strictly-past forecast panel** (the same leakage-safe, `shift(1)`/`closed='left'` feature construction as the forecaster — lags, rolling stats, neighbour activity, calendar, static per-cell, and the new peak/heavy features).
- **Label definition:** a cell that is **NOT currently hot** — its trailing-28-day count is below the top-decile threshold — **but whose FORWARD-28-day count rises into the top decile**. Currently-hot cells are excluded from the positive class by construction, so "predicted emerging" is **disjoint from currently-hot**.
- **Outputs per cell:** `emergence_risk` (0–1), `risk_band` (high / elevated / low, percentile-based), and `predicted_emerging` (bool).

## 4. Evaluation protocol — temporal holdout, no leakage

- Validation is a **temporal holdout** (train on the past, score the forward window) — **never a random split**, since a random split would leak future panel rows into the past.
- Features are strictly past-only by construction (inherited from the forecast panel); the forward-28-day label window is held out of training.

## 5. Metrics (verified)

| Metric | Value |
|---|---|
| ROC-AUC (temporal holdout) | **0.920** |
| Persistence/trend baseline ROC-AUC | 0.820 |
| Lift over baseline | **+0.10** |
| Cells scored | **2,534** |
| Currently hot | 157 |
| Predicted emerging | **238** |
| Approx. positive base rate | ~1.5% |

**Honest read:** AUC 0.920 vs a 0.820 persistence/trend baseline is a genuine +0.10 lift — the model adds real forward signal over simply assuming hot-stays-hot / trend-continues. But emergence is rare (~1.5% base rate), so the absolute `emergence_risk` values are small; bands and the `predicted_emerging` flag are therefore **percentile-based**, and the AUC is reported as-is without inflation.

## 6. Limitations

- **Low base rate.** Emergence is intrinsically rare (~1.5%); absolute risk scores are small and should be read as a *ranking* (percentile bands), not calibrated probabilities.
- **Enforcement-defined label.** The top-decile label is built from recorded counts, so "emergence" means *recorded* emergence; cells that emerge but go un-patrolled won't be labelled positive.
- **Single city, single vintage; short window.** Same constraints as the forecaster — ~5 months of data, no transfer guarantee, retrain per city/period.

## 7. Ethical considerations

- **Same feedback-loop hazard as the forecaster.** The label is *recorded* violations = patrol activity, so acting naively on emergence risk risks a predictive-policing loop. It must be read alongside the exposure-adjusted ranking (raw↔adjusted Spearman ≈ 0.98) and the under-enforcement blind spots — low recorded counts are never presented as "compliant."
- **No individual targeting.** Cell-aggregate only; plates are SHA-256+salted and never reach the model; public layers are H3-only with `count<5` suppression.
- **Decision support, not automation.** The watch-list informs human patrol planning; it is not an automated enforcement or penalty mechanism.

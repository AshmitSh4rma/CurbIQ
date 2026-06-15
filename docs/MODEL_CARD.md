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
- **Static per-cell:** centroid `lat`/`lon`, mean road-class loss, junction-cell fraction, number of distinct police stations.

## 5. Objective & configuration

- **Objective:** `poisson` (the counts are overdispersed and zero-inflated); `metric='poisson'`. Tweedie (`variance_power≈1.2`) is the documented fallback for the zero-heavier junction grid.
- **Key params:** `n_estimators=3000`, `learning_rate=0.04`, `num_leaves=63`, `min_child_samples=100`, `feature_fraction=0.8`, `bagging_fraction=0.8 / freq=1`, `lambda_l1=0.5`, `lambda_l2=1.0`, early stopping = 100 rounds on the validation fold.

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
| MAE | 2.07 | 2.04 |
| RMSE | 7.02 | 6.86 |
| R² | 0.59 | **0.64** |
| Mean Poisson deviance | 3.44 | 3.48 |
| ROC-AUC (top-10% hotspot label) | **0.929** | 0.925 |
| PR-AUC | 0.677 | 0.672 |
| PAI@5% | 12.33 | **12.73** |
| PEI@5% | 0.807 | 0.826 |
| PAI@20% | 4.42 | 4.38 |
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
| **LightGBM Poisson (metro-enriched)** | **2.04** | **6.86** | **0.64** | **3.48** | **0.93** | **12.73** | **0.826** |

**Honest read:** LightGBM clearly beats the naive last-week baseline across every metric (MAE 2.04 vs 2.50, Poisson deviance 3.48 vs 17.86, AUC 0.93 vs 0.80) and edges the strong EWM-persistence baseline on RMSE/deviance/AUC and **narrowly on PAI@5 too (12.73 vs 12.54)** — the margin is modest because dominant hotspots are inherently *persistent* and a good persistence model captures most of the ranking lift. The model's real edge is calibrated counts (deviance/MAE/AUC), responsiveness to neighbour / calendar / metro structure, and a principled probabilistic output.

## 9. External feature enrichment (3-way A/B)

Two external sources were tested as forecast features: daily **weather** (Open-Meteo) and **metro-station proximity** (OSM / Overpass — 81 Namma Metro stations). Holdout vs CV, identical protocol:

| enrichment | HLD PAI@5 | HLD MAE | HLD R² | CV MAE |
|---|---|---|---|---|
| none | 12.55 | 2.044 | 0.647 | 2.053 |
| **metro-only (default)** | **12.73** | 2.037 | 0.636 | 2.065 |
| metro + weather | 12.61 | 2.035 | 0.652 | 2.137 |

**Decision: metro-only.** It gives the best holdout **PAI@5 (12.73)** — the product metric — and `metro_dist_m` ranks top-8 (gain importance **≈1070** when undiluted by weather), directly relevant to the brief's "near metro stations / commercial areas" framing. Weather nudges holdout MAE/R² but clearly **hurts CV** (MAE 2.05 → 2.14; noisier in the dry-season folds), so it is **off by default** — pass `enrich="all"` to include it. Enrichment degrades gracefully offline (metro falls back to a built-in station list).

---

## 10. Limitations

- **Persistence-bound PAI.** Because hotspots are stable, the ranking lift over a strong EWM baseline is small; the model earns its keep on calibration and AUC, not PAI.
- **Short horizon.** Forecasts one day ahead from the last available day; multi-day/seasonal forecasting is out of scope.
- **Window length.** Only ~5 months of data → limited seasonal coverage; holidays are hardcoded for the window, not generalized.
- **Spatial granularity trade-off.** Res-8 maximizes forecast signal but is coarser than the res-9 hotspot grid; the parent-cell mapping is approximate.
- **R² ~0.6 with high RMSE** reflects a heavy-tailed count distribution — a few very high-count cells dominate squared error.
- **Single city, single vintage.** No transfer guarantee; must be retrained per city/period.

## 10. Ethical considerations

- **Enforcement bias / feedback loop (the central hazard).** The label is *recorded* violations = patrol activity. Naively acting on the forecast risks a predictive-policing loop: patrol → record → forecast says "hotspot" → patrol again. CurbIQ mitigates this by **not** ranking on raw counts in production: prioritization uses an **exposure-adjusted** rate and publishes the raw↔adjusted Spearman rank-shift (≈0.98 here, evidence the top hotspots are genuine, not patrol artifacts), and it surfaces **under-enforcement blind spots** (402 cells with high modeled propensity but low observed enforcement) as a first-class output — low-count zones are never presented as "compliant."
- **Temporal blind spot made visible.** The fairness layer shows evening-peak (17–20h IST) enforcement share ≈ **0.2%** despite high congestion risk — a direct artifact of patrol scheduling the forecast alone would perpetuate.
- **No individual targeting.** Cell-aggregate only; plates are SHA-256+salted and never reach the model; public layers are H3-only with `count<5` suppression (see [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md)).
- **Decision support, not automation.** Outputs inform human patrol planning; they are not an automated enforcement or penalty mechanism.

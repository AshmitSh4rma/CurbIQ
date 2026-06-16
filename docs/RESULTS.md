# CurbIQ — Results & Decision Brief (Judge One-Pager)

**A production AI platform on real Bengaluru Traffic Police parking-violation data that detects illegal-parking hotspots, quantifies their congestion impact, and prioritizes enforcement.**

*Dataset: 298,445 geocoded per-incident records (298,125 counted), Nov 2023 – Apr 2024 · 54 police stations · 168 named junctions · 2,534 H3 res-9 cells · License Apache-2.0 · Privacy by design under India's DPDP Act 2023.*

> *Dataset note:* the public HackerEarth challenge CSV ("jan to may police violation, anonymized") was verified to be the **same** export used here — 298,450 rows, identical Nov 2023 – Apr 2024 IST span and monthly distribution (the "jan to may" filename is mislabeled relative to the actual `created_datetime` values). No data migration was needed; all figures below are computed on this exact dataset.

---

## The Problem (one paragraph)

Bengaluru Traffic Police records hundreds of thousands of parking violations a year, but those records are *enforcement logs*, not a map of where illegal parking actually chokes the city. Tickets cluster where patrols happen to go and when shifts happen to run, so raw counts can't tell an officer *where* a tow vehicle removes the most congestion or *when* a picket is missing. CurbIQ turns the raw enforcement export into three decision layers — statistically rigorous **hotspots**, an explainable **congestion-impact** estimate, and an exposure-corrected **enforcement priority** ranking — so BTP can deploy finite tow vehicles and pickets where and when they buy the most relief, and can see where current enforcement is systematically blind.

---

## 3 Standout Findings

1. **One hotspot zone in Central Bengaluru concentrates ~60k violations.** The top contiguous hotspot zone (Shivajinagar / Commercial St, 33 cells, peak Gi* z = 34.6) holds **59,945 violations** — roughly **20% of the entire city** in one ~3.5 km² cluster. Global Moran's I = 0.343 (z = 42.3, p < 1e-300): this clustering is real, not noise.

2. **The evening peak is an enforcement blind spot — 0.2% of tickets when congestion risk is highest.** During the 17:00–21:00 IST evening peak (hours 17–20 inclusive), enforcement share is **0.20%** (596 of 298,125 records) even though our risk model puts the highest congestion exposure in exactly those hours. The four most under-enforced hours are **19, 20, 18, 17** IST. Enforcement is a *daytime/morning* activity; the evening jam goes essentially un-ticketed.

3. **Just 84 locations cover 50% of all violations.** The coverage curve shows **84 H3 cells capture 50%** of violations and **526 cover 80%** — a small, deployable target set. This is efficiency, not coincidence: exposure-adjusted vs. raw ranking correlate at Spearman **0.98**, so the top hotspots are genuine demand, not patrol artifacts.

---

## Results Table

| Layer | Metric | Value |
|---|---|---|
| **Hotspots** | Global Moran's I (z, p) | 0.343 (z = 42.3, p < 1e-300) — strong clustering |
| | High-confidence hotspots (Gi* z ≥ 3.29) | **112** |
| | Benjamini-Hochberg FDR critical p | 0.00102 |
| | Contiguous hotspot zones | 10 |
| | Top zone (Central Bengaluru) | 33 cells, **59,945** violations, peak Gi* z = 34.6 |
| | Top junction | Safina Plaza — **15,413** (68.9% in peak hours) |
| **Congestion (MODELED)** | City delay-impact index | **7,513** |
| | Mean / p95 / max extra delay per cell | 2.0% / 3.7% / **7.0%** |
| | Method | HCM 2000 capacity loss → BPR 1964 delay multiplier |
| **Forecast (LightGBM Poisson, walk-forward, 36 features)** | Holdout ROC-AUC / R² | **0.925** / **0.65** (was 0.64) |
| | Holdout MAE / RMSE | 2.02 / 6.70 |
| | PAI@5% / PAI@20% | **12.72×** / 4.39× |
| | PEI@5% | 0.825 |
| | New peak/heavy features (importance) | the peak-hour & heavy-vehicle lags all contribute (e.g. `peak_roll7` importance **1018**, a top-tier feature) |
| | vs. baselines (last-week / rolling-7 / EWM) | Wins on MAE/RMSE/AUC; ties strong EWM on PAI |
| **Emergence (forward 28-day, LightGBM binary, temporal holdout)** | ROC-AUC vs persistence/trend baseline | **0.920** vs 0.820 (**+0.10** lift) |
| | Cells scored / currently hot / predicted emerging | 2,534 / 157 / **238** |
| | Positive base rate (bands percentile-based) | ~1.5% |
| **Congestion ROI (MODELED)** | City recoverable-delay index (reconciles with delay-impact index) | **7,513.3** |
| | Cells for 50% / 80% of recoverable modeled delay | **38** / 206 |
| | Top cell share / recoverable | ≈ **4.96%** / 372.8 |
| **Prioritization** | Hotspot cells / under-enforcement blind spots | 109 / **408** |
| | Locations for 50% / 80% coverage | **84** / 526 |
| | Exposure-adjusted vs. raw rank (Spearman) | 0.98 |
| | BTP enforcement alignment points | 154 |
| **Fairness & Privacy** | Evening peak (17:00–21:00 IST) enforcement share | **0.20%** |
| | Spatial disparate-impact ratio (4/5ths rule) | 0.42 (flagged < 0.8) |
| | Public granularity / suppression | H3 cells only; count < 5 suppressed (k = 5) |
| | Plate handling | SHA-256 + salt |

---

## Decision Narrative — outputs → concrete BTP actions

CurbIQ's outputs map directly onto BTP's real enforcement geography (12 corridors / 43 junctions / 99 major roads ≈ **154 high-density points** where tow vehicles + pickets deploy):

- **WHERE to send tow vehicles (standing pickets):** the 84-cell 50%-coverage set, led by the Central Bengaluru zone (Shivajinagar / Commercial St, ~60k) and the priority top-5 cells (ranks 1–5 around 12.97 N, 77.58 E — Upparpet/City-Market core). Top priority cell alone: 12,106 violations, Gi* z = 34.6, CIS = 100. **Action: permanent tow + picket on these 84 points = half the problem with a deployable footprint.**
- **WHEN to deploy — the missing evening shift:** enforcement collapses after 16:00 IST; the 17–20h window (highest modeled congestion risk) gets 0.2% of tickets. **Action: add an evening enforcement shift at the top hotspot junctions (Safina Plaza, KR Market, Elite, Sagar Theatre) targeting hours 17–20.**
- **WHICH junctions for static pickets:** the top junctions are peak-dominated — Safina Plaza 15,413 (69% peak), KR Market 11,528, Elite 10,708, Sagar Theatre 10,538 — and align to BTP's 43 priority junctions / 154-point grid. **Action: picket placement at named junctions, timed to peak share.**
- **WHERE enforcement is blind (don't read counts as truth):** 408 under-enforcement blind spots (83 survive k-anonymity for per-cell display; the rest guide zone-level audits) — cells with high modeled propensity but low ticketing — and stations below the 0.8 coverage ratio (e.g. K.G. Halli 0.58, Peenya 0.70). **Action: scout/audit these before assuming they're clean.**
- **Forward-looking deployment:** the LightGBM forecast (PAI@5% = 12.7×) lets BTP pre-position next week's resources on the cells most likely to flare, not just last week's tickets.

---

## Methodology Rigor

- **Multiple-testing control:** 112 hotspots are Gi* z ≥ 3.29 *after* Benjamini-Hochberg FDR (critical p = 0.00102), not raw p < 0.05 — guards against false hotspots from 2,534 simultaneous tests.
- **No leakage:** forecast is walk-forward / time-blocked CV with a held-out month (2024-04); features are strictly lagged (lag1/7/14/28, rolling stats, neighbor lags, and now strictly-past peak-hour & heavy-vehicle lags). Holdout AUC 0.92 ≈ CV AUC 0.93 → no overfitting. The forward **hotspot-emergence** classifier (predicting which not-currently-hot cells turn hot within 28 days) uses a **temporal holdout, never a random split** — AUC 0.920 vs a 0.820 persistence/trend baseline, reported without inflation (emergence base rate ~1.5%, so risk bands are percentile-based).
- **Congestion ROI reconciles, not invented:** the per-cell recoverable modeled delay `(BPR delay-ratio − 1) × count` sums **exactly** (abs diff 0.0) to congestion's existing `city_delay_impact_index` = **7,513.3** (unitless modeled delay units, *not* minutes). It is a modeled upper bound (assumes clearing a cell removes its parking-induced capacity loss entirely): **38** cells recover 50% and **206** recover 80% of the city's modeled recoverable delay.
- **Honest baselines:** model is benchmarked against last-week, rolling-7, and EWM-persistence; it wins on error/AUC and **narrowly edges** the strong EWM baseline on PAI (12.72 vs 12.54) — a deliberately modest claim, because stable hotspots are inherently persistent.
- **Modeled, not measured congestion:** CIS is an *explainable estimate* (HCM 2000 capacity loss → BPR 1964 delay), labeled as such everywhere. **We have no speed/flow data; we never claim measured delay.**
- **Exposure-adjusted bias correction:** recorded violations reflect *enforcement activity* (patrol-shift bias), not true occurrence. We surface this via exposure-adjusted ranking (Spearman 0.98 vs. raw) and 408 detected under-enforcement blind spots (83 publicly displayable after k-anonymity).
- **Privacy by design (DPDP Act 2023):** public layers are H3-cell-only (never raw lat/lon), cells with count < 5 are suppressed (k-anonymity, ~33% of cells), plates are SHA-256 + salted. Apache-2.0.

---

## Limitations (explicit)

1. **Congestion impact is modeled, not measured.** No traffic speed/flow/volume data exists in the dataset; CIS is a transparent HCM→BPR estimate for *relative* prioritization, not absolute delay in seconds.
2. **Counts are enforcement, not ground truth.** Where patrols don't go, violations look low. We mitigate (exposure adjustment, blind spots) but cannot fully recover unobserved violations.
3. **No anchor for ASTraM/Gridlock identity.** We assert BTP/ASTraM *lineage* and Gridlock-2.0-style framing; this is not a claim to any specific named hackathon's exact dataset.
4. **Six-month window (Nov 2023 – Apr 2024).** No full-year seasonality (monsoon, festivals) is captured; forecast generalizes within-period, not across years.
5. **Geocoding & station mapping inherit source quality.** ~2% road-class "unknown", a "No Police Station" bucket, and 16.7% rejected tickets (down-weighted by confidence, not dropped) reflect raw-export limits.
6. **Persistence ties on PAI.** For ranking *which* cells are hot, a simple persistence baseline is nearly as good; the model's edge is in *change detection* and forward forecasting, which we report honestly.

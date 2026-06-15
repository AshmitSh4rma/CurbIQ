# CurbIQ — Consolidated Technical Brief

**Status:** Architecture-locked decision brief. Supersedes the five specialist findings.
**Scope:** Bengaluru parking-violation analytics over 298k geocoded per-incident records (Nov 2023–Apr 2024): lat/lon, multi-label `violation_type`, `offence_code`, `junction_name`, `police_station`, `validation_status`, UTC `created_datetime`.
**Hard constraints:** numpy / scipy / scikit-learn / lightgbm / h3 / shapely / pandas only. **No GDAL, geopandas, or torch** (disk + GPU constraints). Apache-2.0 / open-source-ready.

---

## 1. Source Hackathon & Evaluation Criteria

**No public event matches our exact schema** (298k geocoded multi-label rows). We assert **lineage, not identity**: CurbIQ is a Bengaluru Traffic Police (BTP) / ASTraM-style enforcement-analytics build. Public BTP open data (OpenCity "Bengaluru Traffic Violations Data", btp.gov.in/Enforcementstats) publishes **aggregate** counts only, confirming our dataset is a custom geocoded export.

**Closest framing match — mirror this:**
- **Flipkart × BTP (ASTraM) "Gridlock Hackathon 2.0"** (HackerEarth): charter = "classify congestion, detect violations, identify movement patterns" using BTP ASTraM road-density records + MapmyIndia layers. Phases: P1 online ML + leaderboard (26 May–7 Jun 2026); P2 working prototype (8–21 Jun); P3 onsite finale (3 Jul 2026). Prize Rs 5,00,000 (2.25L / 1.75L / 1.0L) → 3-finalist **judge-panel** finale, not a pure leaderboard.
- **KSP Datathon 2024** (Hack2skill + Azure): Track 2 = "parking/encroachment within **100 m of junctions**"; Track 4 = anonymization. Closest thematic + privacy match.
- **Bengaluru Mobility Challenge 2024** (IISc/CDPG/BTP): finalists open-source under **Apache-2.0** — adopt this.

**Optimize for the convergent Indian smart-city judge rubric (this IS our scoring surface, since no automated parking-hotspot leaderboard exists anywhere):**

| Criterion | Weight |
|---|---|
| Innovation | ~20% |
| Technical implementation / rigor | ~20–25% |
| Real-world impact | ~20–25% |
| Feasibility / deployability | ~20% |
| Scalability + presentation/clarity | ~15–20% |

**Deliverable = 3 judge-facing layers:** (1) reproducible model/notebook → (2) interactive hotspot/prioritization dashboard → (3) decision narrative tying hotspots to BTP enforcement actions. Align all hotspot output to BTP's real enforcement geography: **12 corridors / 43 junctions / 99 major roads (~154 high-density points)** where tow vehicles + pickets deploy.

---

## 2. Final Method + Parameters per Subsystem

### (a) Statistical Hotspots — Gi* + Local Moran's I on H3

- **Primary grid: H3 resolution 9** (~174–201 m edge, ~0.105 km²) = street/block scale. **Res 10** (~66–76 m, ~0.015 km²) for junction drill-down. Res 8 only for city-overview zoom. *(A fetched source mis-recommended res 8 as primary — the h3geo area table contradicts it; res 9 is correct for 150–300 m block scale.)*
- **Analysis variable** `x` = per-hex violation count (numpy bincount on H3 cell index).
- **Getis-Ord Gi\*** (the "star" — **includes focal cell i**, w_ii=1) is itself a z-score; do not re-standardize:

  `Gi* = (Σⱼ wᵢⱼ·xⱼ − X̄·Σⱼ wᵢⱼ) / (S · sqrt[(n·Σⱼ wᵢⱼ² − (Σⱼ wᵢⱼ)²)/(n−1)])`
  where `X̄ = Σⱼ xⱼ / n`, `S = sqrt(Σⱼ xⱼ²/n − X̄²)`.

- **Local Moran's I (LISA)** for cluster typing (esda-exact, excludes self, row-standardized W):
  `Iᵢ = (n−1)·zᵢ·slagᵢ / Σⱼ zⱼ²`, `z = (x−x̄)/σ` (population std).
  Quadrants: **1=HH** (hotspot core), **2=LH** (outlier), **3=LL** (coldspot), **4=HL** (outlier).
- **Spatial weights = H3 k-ring (`grid_disk`), k=1:** include self for Gi* (binary, 7 cells); exclude self + row-standardize for Moran's (6 neighbors). Expose **k=2** (~18 cells) as smoothing/sensitivity.
- **Inference:** Gi* → analytical normal p (`scipy.stats.norm.sf`). Moran's → **999 conditional permutations**, pseudo-p = `(min(larger, perms−larger)+1)/(perms+1)`, fixed seed.
- **Significance tiers:** |z| ≥ 1.65 (90%), ≥ 1.96 (95%), ≥ 2.58 (99%), ≥ 3.29 (99.9%). **Display only z > 3.29 as "high-confidence hotspot"** to avoid over-flagging.
- **MUST apply Benjamini-Hochberg FDR (α=0.05)** to the full local-p vector (`scipy.stats.false_discovery_control(method='bh')`) before declaring any hotspot — thousands of cells means ~5% false hotspots without it. Store raw + FDR-adjusted.
- **Emerging hotspots:** weekly bins (~26 wks, UTC→IST first), per-(hex,bin) Gi*, Mann-Kendall on each hex's z-series → new/intensifying/persistent/diminishing/sporadic/oscillating. Require a min-events-per-bin threshold or coarsen to res 8 / monthly for sparse cells.
- **Validation harness (build-time only):** assert numpy Gi*/Moran's agree with `esda.G_Local` / `esda.Moran_Local` within tolerance. PySAL is **not a runtime dep**.
- **Diagnostics to always report:** neighbor counts (≥1, ideally ~6–30), # isolates, Global Moran's I as overall clustering sanity check, and a res9 k=1 vs k=2 sensitivity (MAUP defense).

### (b) Congestion Impact Score (CIS) — modeled, never measured

No speed/flow data exists → CIS is an **explainable composite z-score**, labeled "modeled, not measured" everywhere.

**Pipeline:** maneuver intensity → HCM capacity loss → PCU-weight by vehicle mix → BPR delay multiplier → network-wide normalize.

1. **Capacity loss** via HCM parking factor: `fp = (N − 0.1 − 18·Nm/3600)/N` — each maneuver blocks the adjacent lane ~18 s; cap `Nm` at 180/h; floor `fp ≥ 0.05`. *(Worked: N=2, Nm=60 → fp=0.80 = 20% loss; Nm=180 → fp=0.50 = 50% loss.)* Bound total capacity loss to empirical Indian side-friction range, **cap at 0.60**.
2. **PCU weighting** (IRC 106-1990, low-share <5% / high-share ≥10%; pick per local cell composition): 2-wheeler 0.50/0.75, car 1.0, auto-rickshaw 1.20/2.00, LCV 1.40/2.00, bus/truck 2.20/3.70.
3. **BPR delay multiplier:** `t = t0·(1 + α·(v/c')^β)`, `α=0.15`, `β=4`, `c' = c·(1−loss)`, assumed baseline `v/c=0.80`. **Headline explainable output = ratio `M_parked/M_base`.** **Clamp v/c' ≤ 2.0** (β=4 explodes past capacity).

**Final composite (weights are tunable config, exposed in UI):**

```
CIS = 0.35·z(severity_weighted_density)
    + 0.25·z(junction_proximity)
    + 0.25·z(peak_hour_overlap)
    + 0.15·z(road_class_capacity_loss)
→ min-max scale to 0–100
```

- **Junction proximity:** decay `e^(−d/d0)`, `d0 ≈ 150 m` (one res-9 cell radius); honors KSP's **100 m** bottleneck rule.
- **Peak overlap:** UTC→IST first; peaks **08:00–11:00** and **17:00–21:00** IST; overlap = fraction of cell events in peaks.
- **Road-class capacity-loss lookup:** arterial 0.40, sub-arterial 0.30, local 0.15.
- **Multi-label `violation_type`:** one-hot; weight obstruction offences (double-parking, no-parking-zone, footpath/bus-stop) higher.
- **Optional calibration module** (off by default): if probe data supplied (TomTom Traffic Index `congestion% = (actual−freeflow)/freeflow·100`, Google `duration_in_traffic`, archived Uber Movement), fit block-time/α/β via `scipy.optimize` + `IsotonicRegression`, maximize **Spearman ρ** (benchmark Uber-Movement r≈0.87), report ρ/R²/MAE. Core score runs without it.
- **Per-hotspot UI tooltip:** capacity loss %, BPR Δ-delay %, PCU-weighted maneuver rate, dominant violation_type, HCM/IRC/BPR citations.

### (c) Spatiotemporal Forecasting — LightGBM Poisson panel

Frame as a **panel/tabular** problem on (cell × time-bucket); no ConvLSTM/ST-GNN (no GPU, marginal gain).

- **Spatial unit:** H3 **res 8** for station/area map, **res 9** for junction drill-down; restrict to ever-active cells.
- **Time bucket:** **daily** default; optional 6h for intra-day scheduling. UTC→IST first.
- **Panel reshape:** full cartesian (active_cell × time_bucket), left-join counts, **materialize zeros** (absence signal — without it PAI is meaningless).
- **Target:** next-period violation count per (cell, bucket); also a binary top-10% hotspot label for the ROC view.
- **Feature set:** lags `[1,2,3,7,14,28]`; rolling mean/std windows `[3,7,14,28]` + EWM(halflife 7) — all `shift(1)` / `closed='left'`; H3 k-ring(1)&(2) lagged + 7-bucket-rolling neighbor sums (÷ ring size); cyclical hour/dow (sin/cos); weekend + India/Karnataka holiday flags (`holidays` lib); static `police_station` + nearest `junction_name` + road-class proxy + cell centroid lat/lon.
- **LightGBM config:** `objective='poisson'` (primary), Tweedie `variance_power=1.2` fallback for zero-heavy junction grid; `metric='poisson'`, `learning_rate=0.03–0.05`, `num_leaves=31–63`, `min_child_samples=50–200`, `feature_fraction=0.7–0.8`, `bagging_fraction=0.7–0.8`, `bagging_freq=1`, `lambda_l1=0.5`, `lambda_l2=1.0`, `n_estimators=2000–5000`, `early_stopping_rounds=100`.
- **Validation: expanding-window walk-forward by month** (train ≥3 months → validate next), **28-bucket embargo gap** (≥ longest rolling window AND ≥ horizon), final Apr-2024 holdout. **Never random KFold** (future leaks into past).
- **Metrics — product leaderboard = PAI@5% and PAI@20%, with PEI/PEI\* for honesty:**
  - `PAI = (c_a/C)/(a/A)` — rank cells by **predicted density** desc; equal-area H3 means count==density.
  - `PEI = PAI / oracle-PAI` (oracle ranks by **observed** count); **PEI\*** caps oracle at realistically-patrollable cell count.
  - Secondary: mean Poisson deviance, MAE, RMSE, ROC-AUC / PR-AUC (PR preferred, rarity).
- **Beat 3 baselines, surface PAI lift in UI:** same-weekday-last-week, historic cell mean, rolling-7.

### (d) Enforcement Prioritization

- **Default ranking = exposure-adjusted, NOT raw counts** (breaks patrol→record→rank→patrol feedback loop):
  `adjusted_rate = raw_count / (E_cell + α)`, Laplace `α = median(E_cell)`,
  `E_cell = active_hours_recorded + distinct_offence_types + temporal_spread`.
  Raw counts available only as a comparison toggle; **publish the Spearman rank-shift** raw↔adjusted to prove bias correction matters.
- **Under-enforcement gap (first-class output):** `gap = modeled_propensity_percentile − observed_count_percentile`; flag **gap > 0.30** as "enforcement blind spots." Never present low-count zones as "compliant."
- **Patrol allocation:** ranked priority list + **coverage-vs-effort curve** (cumulative violations captured vs # locations enforced). Spatial priority weight = Gi* z-score × junction criticality (the explainable stand-in for absent flow data). Validate top-N against BTP's 154 high-density points and report overlap.

### (e) Fairness / Privacy Safeguards + Metrics

- **Core hazard = predictive-policing feedback loop:** records are enforcement events, not true occurrence. Always show exposure-adjusted alongside raw.
- **Fairness panel (per police-station / zone):**
  - Disparate-impact ratio (EEOC 4/5ths): flag `group_rate / max_group_rate < 0.8`.
  - Statistical-parity difference: flag `|P(enforced|A) − P(enforced|B)| > 0.10`.
  - Enforcement-coverage ratio = observed_percentile / modeled_propensity_percentile.
  - Document that fairness metrics are mutually incompatible at differing base rates (impossibility theorem) — pick DI + parity-difference, explain residual disparity.
- **DPDP Act 2023 (Rules 2025, full compliance 13 May 2027) by design:** drop or SHA-256+salt `vehicle_registration`; **never expose point lat/lon** in API/artifacts — only H3 cell IDs / centroids; **suppress any cell/time-bucket with count < 5** (k-anonymity). Report % cells suppressed (utility-vs-privacy). Ship a short `DATA_GOVERNANCE.md`.
- **`validation_status`:** filter/weight to validated violations before scoring (don't count dismissed tickets); profile distributions first (beware all-TRUE/constant flag false signals); document handling.

### (f) Architecture — precompute-then-serve

- **Offline batch `build_artifacts(df)`** (pure/idempotent) does ALL heavy spatial math (Gi*, Moran's, KDE, CIS, forecast) → versioned **Parquet** (tabular) + compact **JSON** map layers `{h3, gi_z, score, count}` under `/artifacts/v/<YYYY-MM>/` + `manifest.json` (version + checksums).
- **FastAPI = read-only artifact server.** Load Parquet into memory in a `lifespan` startup handler. Endpoints `/api/hotspots`, `/api/score`, `/api/fairness`, `/api/timeseries` return precomputed slices with strong **ETag + Cache-Control**; **version in URL path** → CDN cacheable + cache-bustable. No spatial math in request handlers.
- **Frontend = single static `index.html`, zero build.** CDN `<script>` tags with **pinned exact versions** (unpkg `@x.y.z`): Leaflet + Leaflet.heat (KDE soft layer), deck.gl `@deck.gl/core` + `@deck.gl/geo-layers` (H3HexagonLayer via `getHexagon`/`getFillColor`/`getElevation`), Chart.js (time series + fairness bars).
- **Real-time seam:** keep `build_artifacts(df)` idempotent so a future cron/stream consumer re-emits the same schema + manifest without touching the API/frontend contract.

---

## 3. Hardcoded Defaults (each with one-line citation)

| Parameter | Default | Citation |
|---|---|---|
| Primary H3 resolution | **9** (~200 m edge, 0.1053 km²) | h3geo.org restable — block-scale; res 8 too coarse |
| Drill-down H3 resolution | **10** (~66–76 m, 0.015 km²) | h3geo.org restable |
| Spatial weights | H3 k-ring **k=1** (self for Gi*, 6 nbrs Moran's) | CARTO spatial-hotspot guide; PySAL esda |
| Gi* significance tiers | 1.65 / 1.96 / 2.58 / **3.29** (display ≥3.29) | ArcGIS Hot Spot Analysis z→confidence |
| FDR correction | Benjamini-Hochberg, **α=0.05** | ArcGIS Optimized Hot Spot (FDR default) |
| Moran's permutations | **999**, pseudo-p=(min+1)/(perms+1), fixed seed | PySAL esda Moran_Local default |
| Per-maneuver lane block | **18 s** | HCM 2000 Ch.16 parking factor |
| Maneuver rate cap Nm | **180/h** | HCM 2000 Ch.16 |
| HCM fp floor | **0.05** | HCM 2000 Ch.16 |
| Capacity-loss cap | **0.60** | Indian side-friction studies (17–66%, ETASR/Frontiers 2026) |
| Road-class capacity loss | arterial 0.40 / sub-art 0.30 / local 0.15 | Traffic-eng. rule-of-thumb (curb obstruction 20–50%) |
| PCU (low/high share) | 2W 0.5/0.75, car 1.0, auto 1.2/2.0, LCV 1.4/2.0, bus/truck 2.2/3.7 | IRC 106-1990 PCU table |
| BPR α, β | **0.15, 4** | US BPR 1964 (AequilibraE VDF docs) |
| Baseline v/c | **0.80**; clamp v/c' ≤ **2.0** | BPR sensitivity (β=4 explodes >capacity) |
| CIS composite weights | 0.35 density / 0.25 junction / 0.25 peak / 0.15 road | Convergent explainable-composite design |
| Junction-proximity decay d0 | **150 m** (≈ res-9 radius); KSP 100 m intent | KSP Datathon 2024 Track 2 |
| IST peak windows | **08:00–11:00, 17:00–21:00** IST | Standard Bengaluru AM/PM peaks |
| Timezone shift | UTC → **IST (+5:30)** before all temporal work | dataset `created_datetime` is UTC |
| Forecast time bucket | **daily** (optional 6h) | NIJ/crime-forecast review temporal norms |
| LightGBM objective | **poisson** (tweedie var_power 1.2 fallback) | LightGBM docs; overdispersed zero-inflated counts |
| Lags / rolling windows | lags [1,2,3,7,14,28] / windows [3,7,14,28] + EWM hl=7 | Spatial-crime forecasting best practice |
| CV embargo gap | **28 buckets** (≥ longest window & ≥ horizon) | Purged/embargoed walk-forward CV |
| Product metric | **PAI@5%, PAI@20%** + PEI/PEI\* | Wheeler ptools::pai; White & Hunt 2023 |
| Disparate-impact flag | ratio **< 0.8** (4/5ths rule) | EEOC 80% rule |
| Statistical-parity flag | **> 0.10** absolute | Standard parity-difference convention |
| Under-enforcement gap flag | **> 0.30** percentile gap | Exposure-aware design ("Predictive Enforcement" 2024) |
| Privacy suppression | cell/time-bucket count **< 5** suppressed | DPDP Act 2023 / k-anonymity |
| Plate handling | drop or **SHA-256 + salt** | DPDP Act 2023 (plates = personal data) |
| BTP enforcement geography | **12 corridors / 43 junctions / 99 roads (~154 pts)** | Deccan Herald BTP tow-vehicle deployment |

---

## 4. Top Pitfalls & How We Avoid Them

1. **Patrol-bias laundering** — ranking by raw counts maps where police *go*, not where violations *are*. → Default to exposure-adjusted ranking; surface under-enforcement blind spots; publish raw↔adjusted Spearman shift.
2. **"Congestion" implies measurement** — there is no flow/speed data. → CIS is always labeled "modeled, explainable estimate" with visible component weights + HCM/IRC/BPR citations; never presented as measured delay.
3. **No FDR correction** — thousands of Gi* tests → hundreds of false hotspots at raw p<0.05. → Benjamini-Hochberg always; display only z>3.29 tier.
4. **Random KFold / rolling-feature leakage** — inflates forecast scores. → Expanding walk-forward, 28-bucket embargo, `shift(1)`/`closed='left'` rolling.
5. **Dropping zero-count cells** — destroys the absence signal and makes PAI meaningless. → Materialize zeros over the ever-active grid.
6. **UTC peak-hour skew** — 5.5 h shift corrupts peaks/weekends/holidays. → UTC→IST before any temporal binning, everywhere.
7. **`validation_status` inflation** — dismissed tickets counted as confirmed. → Filter/weight to validated; profile distributions (beware all-TRUE/constant flags).
8. **MAUP / wrong H3 resolution** — res 8 too coarse (a source mis-recommended it). → Res 9 primary (verified against h3geo area table), res 10 drill-down, report k=1 vs k=2 sensitivity.
9. **BPR blow-up at v/c>1** — β=4 explodes past capacity. → Clamp v/c' ≤ 2.0, cap capacity loss at 0.60.
10. **DPDP exposure** — point lat/lon + plates are re-identifying personal data (Rs 250 cr penalty regime, deadline 13 May 2027). → H3-only API, count<5 suppression, plate drop/hash.
11. **Heavy math in API handlers** — slow, non-cacheable. → All spatial compute offline into versioned artifacts; FastAPI read-only.
12. **Unpinned CDN scripts** — silent breakage with no lockfile. → Pin exact unpkg versions.
13. **Ignoring BTP's real enforcement geography** — fancy clustering that doesn't map to the 12/43/99 units scores poorly on "real-world viability." → Align all hotspot output to BTP units; validate top-N against the 154 high-density points.
14. **Claiming a specific named hackathon's dataset** — unverifiable. → Assert BTP/ASTraM lineage and Gridlock-2.0-style framing, not identity.

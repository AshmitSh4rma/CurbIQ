# CurbIQ

**Illegal-parking hotspot detection, modeled congestion-impact scoring, and exposure-aware enforcement prioritization on real Bengaluru Traffic Police violation data.**

`License: Apache-2.0` · `Python 3.13` · `Stack: numpy · scipy · scikit-learn · LightGBM · H3 · shapely · FastAPI` · `Data: 298k geocoded BTP parking violations (Nov 2023 – Apr 2024)` · `GPU: none required`

CurbIQ turns 298,125 counted, geocoded parking-violation records into three operational layers: **(1)** where illegal parking statistically clusters, **(2)** how much each cluster is *modeled* to choke the carriageway, and **(3)** which locations and time-windows enforcement should target next — ranked to correct for its own patrol bias.

---

## Why this matters

Illegal and spillover parking is one of the most direct, fixable contributors to urban gridlock: a single double-parked lorry on an arterial or a row of vehicles within 100 m of a junction removes a live through-lane exactly when the network is most loaded. In Bengaluru this plays out as:

- **Carriageways and junctions choked by curb obstruction** — parking on main roads, double-parking, parking near zebra crossings and bus stops, all squeezing capacity at peak.
- **Reactive, patrol-shift-driven enforcement** — tickets get written where and when officers happen to be on shift, which is *not* where and when violations actually peak. Recorded counts measure **enforcement activity**, not true occurrence.
- **No unified hotspot-vs-congestion picture** — published BTP open data is aggregate counts only; there is no map fusing *statistical* hotspots with their *congestion consequence*.
- **Hard to prioritize** — with 54 stations, 168 named junctions and thousands of street-blocks, "where do we deploy tow vehicles and pickets first?" has no defensible, data-backed answer.

CurbIQ answers those four questions with a reproducible pipeline and an interactive dashboard, while being explicit about what is *measured* versus *modeled*, and about the enforcement bias baked into the raw data.

---

## Highlights / key results

All figures are produced by `build_all.py` from the real dataset and stored in `data/artifacts/`.

| Layer | Result |
|---|---|
| **Dataset** | 298,445 raw records → **298,125 counted** violations, Nov 2023 – Apr 2024; 54 police stations, 168 named junctions, **2,534** populated H3 res-9 cells |
| **Spatial clustering** | Global Moran's I = **0.343** (z = **42.3**, p < 1e-300) → strong, non-random clustering |
| **Hotspots** | **112** high-confidence hotspots (Getis-Ord Gi\* z ≥ 3.29) after Benjamini-Hochberg FDR (critical p = **0.00102**); **10** contiguous hotspot zones; top zone ≈ 60k violations in Central Bengaluru (Shivajinagar / Commercial St) |
| **Top junction** | Safina Plaza — **15,413** violations, **69%** in peak hours |
| **Congestion (MODELED)** | HCM capacity-loss → BPR delay multiplier → composite CIS (0–100); city delay-impact index ≈ **7,513**; max cell ≈ **7%** extra modeled delay |
| **Congestion ROI (MODELED)** | Recoverable modeled delay per cell `(BPR ratio − 1) × count` reconciles exactly with the city delay-impact index **7,513**; **38** cells recover 50% (206 → 80%) of the city's modeled recoverable delay; top cell ≈ **5.0%** |
| **Forecast** | LightGBM Poisson, walk-forward (no leakage), now **36 features** (was 32) incl. peak-hour & heavy-vehicle lags: holdout **ROC-AUC 0.925**, **R² 0.65** (was 0.64), **MAE 2.02**, **PAI@5% 12.7×**, PAI@20% 4.4×, PEI@5% 0.82 — beats last-week / rolling-7 / EWM baselines |
| **Emergence (forward risk)** | LightGBM binary classifier predicting which cells *become* hotspots in the next **28 days** (temporal holdout): **AUC 0.92 vs 0.82** persistence/trend baseline (+0.10 lift); scores **2,534** cells → **157** currently hot, **238** predicted emerging |
| **Prioritization** | **109** rankable hotspot cells (of 112 Gi\*-significant; 3 are zero-violation neighbour-spillover), **408** detected under-enforcement blind spots (83 survive k-anonymity for display); **84** locations cover 50% of violations (526 → 80%); exposure-adjusted vs raw Spearman **0.98** (top hotspots are genuine, not patrol artifacts) |
| **Fairness** | Evening peak (17:00–21:00 IST, hours 17–20) enforcement share = **0.2%** (the headline blind spot); spatial disparate-impact ratio **0.42** (flagged < 0.8) |
| **Privacy** | Public layers H3-only; cells with count < 5 suppressed (**33%** suppressed); plates SHA-256 + salted |

> **Honest note on the forecast:** CurbIQ ties a strong EWM-persistence baseline on PAI. That is expected and stated openly — stable hotspots *are* persistent. The model's edge is in error metrics (MAE/RMSE), the hotspot AUC, and producing a calibrated next-day count surface rather than a naive carry-forward.

> **Note on the dataset.** The public HackerEarth challenge CSV ("jan to may police violation, anonymized") was verified to be the *same* export CurbIQ already uses — 298,450 rows, identical Nov 2023 – Apr 2024 IST span and monthly distribution; the "jan to may" filename is mislabeled relative to the actual `created_datetime` values. No data migration was needed, and every result here is computed on this exact dataset.

---

## Architecture — precompute, then serve

CurbIQ does **all** heavy spatial/ML math once, offline, into versioned, privacy-safe artifacts. The API and dashboard never run spatial computation in a request — they only read immutable JSON slices. This makes the serving layer trivially fast, CDN-cacheable (strong ETag + `Cache-Control`, 304-aware), and reproducible.

```
                       ┌──────────────────────────────────────────────────────┐
 data/raw/             │                  OFFLINE BATCH                         │
 police_violations.csv.gz                 build_all.py                          │
        │              │                                                        │
        ▼              │   curbiq/etl/pipeline.py                               │
 ┌──────────────┐      │   • textual-null cleanup, dedupe, BBOX clip           │
 │  ETL          │─────▶│   • UTC → IST (+5:30)  ◀── critical for peaks/weekday  │
 │  + features   │      │   • offence/vehicle/road-class/severity/confidence    │
 └──────────────┘      │   • H3 res-9 / res-8 indexing                          │
        │              │                                                        │
        ▼              │   curbiq/features.py                                   │
 data/processed/       │                                                        │
 violations.parquet ───┼──▶  ANALYTICS MODULES (pure, idempotent)               │
                       │     spatial.py   Gi*, Moran's I (global+local), MK     │
                       │     hotspots.py  hotspots / zones / junctions / emerging│
                       │     congestion.py  HCM → BPR → CIS   (MODELED)          │
                       │     forecast.py  LightGBM Poisson panel, walk-forward   │
                       │     emergence.py  forward 28-day hotspot-emergence risk │
                       │     timing.py    per-hotspot 4h enforcement windows     │
                       │     scenario.py  recoverable modeled delay / ROI        │
                       │     prioritize.py  exposure-adjusted rank + blind spots │
                       │     fairness.py  equity (4/5ths) + DPDP privacy         │
                       │                          │                              │
                       │            curbiq/artifacts.py  + k-anonymity           │
                       └──────────────────────────┼──────────────────────────────┘
                                                   ▼
                          data/artifacts/*.json (kpis, cells, forecast, zones,
                          junctions, emerging, emergence, timing, scenario,
                          priority, fairness, timeseries, model_metrics)
                          + manifest.json   |   models/forecast_lgbm.txt
                                                   │
                                                   ▼
                    curbiq/api/main.py  —  FastAPI read-only artifact server
                    /health · /api/{manifest,kpis,cells,forecast,zones,junctions,
                    emerging,emergence,timing,scenario,priority,fairness,
                    timeseries,model-metrics}
                    (loads JSON into memory at startup; ETag + Cache-Control; no math)
                                                   │
                                                   ▼
                    web/  —  build-free Leaflet + h3-js + Chart.js dashboard
                    (map metric switcher · overlays · 4 analytics tabs)
```

**The contract:** `build_artifacts(df)` is pure and idempotent, the manifest carries a version + per-file byte sizes, and the API/frontend depend only on the artifact schema. A future stream/cron consumer can re-emit the same artifacts without touching the server or UI (see *Real-time extension seam*).

---

## Methodology

Full citations and parameter defaults live in [`docs/RESEARCH.md`](docs/RESEARCH.md); every tunable constant is centralized in [`curbiq/config.py`](curbiq/config.py).

### 1. Statistical hotspots — Gi\* / Moran's I / FDR  (`spatial.py`, `hotspots.py`)

- **Grid:** H3 **resolution 9** (~174 m edge, ~0.105 km²) as the primary street-block unit; res 8 for the forecast panel and zonal views.
- **Getis-Ord Gi\*** (the "star" includes the focal cell, so it is itself a z-score) over an H3 `grid_disk` k=1 spatial weight (self + 6 neighbours). Significance is read off analytical normal p-values; only **|z| ≥ 3.29** (99.9%) is displayed as a high-confidence hotspot to avoid over-flagging.
- **Local Moran's I (LISA)** for cluster *typing* (HH core / LL coldspot / HL,LH outliers) using **999 conditional permutations**, pseudo-p = `(min(larger,perms−larger)+1)/(perms+1)`, fixed seed.
- **Benjamini-Hochberg FDR (α = 0.05)** applied to the **Gi\* p-value vector** — this is what gates hotspot declaration; the LISA pseudo-p is used only for cluster *typing* (raw p ≤ 0.05, no FDR). With thousands of simultaneous tests, raw p < 0.05 alone would yield hundreds of false hotspots; FDR pulled the critical p down to **0.00102**.
- **Global Moran's I** is reported as an overall clustering sanity check; **emerging hotspots** use weekly per-cell Gi\* series with a **Mann-Kendall** trend test (new / intensifying / persistent / diminishing / sporadic / oscillating).
- *Citations:* ArcGIS Hot Spot / Optimized Hot Spot Analysis (z→confidence, FDR default), PySAL `esda` (Gi\*, Moran_Local), CARTO spatial-hotspot guide.

### 2. Congestion Impact Score — HCM → BPR  (`congestion.py`)  — **MODELED, not measured**

There is **no speed or flow data** in the dataset, so CIS is an **explainable, modeled composite** — labeled as such everywhere it appears, never presented as measured delay.

1. **Capacity loss** via the HCM 2000 Ch.16 on-street-parking factor `fp = (N − 0.1 − 18·Nm/3600)/N`, where each maneuver blocks the adjacent lane ~18 s, the maneuver rate `Nm` is capped at 180/h, `fp` floored at 0.05, and total loss capped at **0.60** (Indian side-friction range 17–66%).
2. **PCU weighting** of the vehicle mix (IRC 106-1990) — a parked bus/lorry obstructs far more carriageway than a two-wheeler.
3. **BPR delay multiplier** `t = t0·(1 + α·(v/c′)^β)` with α = 0.15, β = 4, baseline v/c = 0.80, `v/c′` clamped ≤ 2.0 (β = 4 explodes past capacity). The headline explainable output is the ratio `M_parked / M_base`.
4. **Composite CIS** = `0.35·z(severity-weighted density) + 0.25·z(junction proximity, e^(−d/150m) decay) + 0.25·z(peak-hour overlap, 08–11 & 17–21 IST) + 0.15·z(road-class capacity loss)`, then min-max scaled to **0–100**. Every per-cell tooltip exposes capacity-loss %, BPR Δ-delay %, dominant offence, and the underlying citations.
- *Citations:* HCM 2000 Ch.16, IRC 106-1990 PCU table, US BPR 1964 (AequilibraE VDF docs), KSP Datathon 2024 100 m junction-bottleneck rule.

### 3. Spatiotemporal forecasting — LightGBM Poisson panel  (`forecast.py`)

- **Panel:** full cartesian (ever-active H3 res-8 cell × day) with the **absence signal materialized as explicit zeros** — without it PAI is meaningless. Target = next-day violation count; a binary top-10% hotspot label drives the ROC view.
- **Features (all strictly past, 36 total):** lags `[1,2,3,7,14,28]`, rolling mean/std windows `[3,7,14,28]` + EWM (halflife 7); **lagged peak-hour & heavy-vehicle (LCV/bus/HGV) event counts** per cell; all `shift(1)` / `closed='left'`; H3 k-ring(1)&(2) lagged neighbour sums; cyclical hour/dow; weekend + India/Karnataka holiday flags; static station / nearest junction / road-class / centroid lat-lon; **nearest-metro distance** (OSM/Overpass, 81 Namma Metro stations; daily weather optional via `enrich='all'`).
- **Model:** `objective='poisson'` (overdispersed, zero-inflated counts), 4000 trees, lr 0.03 (tuned on walk-forward CV), 63 leaves, L1/L2 regularization, early stopping.
- **Validation:** expanding-window **walk-forward by month** (train ≥ 90 days → validate the next month). Leakage is prevented by strictly-lagged features (`shift(1)` / `closed='left'`), a 7-day train/validation split, and a **28-day early-stopping carve from the train tail** (`FORECAST_EMBARGO`) — so early stopping never sees the validation/holdout and gradient training ends ≥ 28 days before the scored period. Final April-2024 holdout is reported alongside the CV mean. **Never random K-fold** (future would leak into past).
- **Metrics judges check:** **PAI@5% / PAI@20%** (rank cells by predicted density, measure hotspot capture vs area share) with **PEI** = PAI / oracle-PAI for honesty, plus ROC-AUC / PR-AUC, MAE, RMSE, R², mean Poisson deviance. Benchmarked against last-week, rolling-7 and EWM baselines.
- *Citations:* LightGBM docs, Wheeler `ptools::pai` / White & Hunt 2023 (PAI/PEI), purged-and-embargoed walk-forward CV.

### 4. Enforcement prioritization — exposure-adjusted + blind spots  (`prioritize.py`)

- **Default ranking is exposure-adjusted, not raw counts**, to break the patrol → record → rank → patrol feedback loop: `adjusted_rate = raw_count / (E_cell + α)` with Laplace `α = median(E_cell)` and exposure proxied by active recorded hours, distinct offence types, and temporal spread. Raw counts remain available only as a comparison toggle. The raw↔adjusted **Spearman 0.98** is published to prove the top hotspots are genuine, not patrol artifacts.
- **Under-enforcement blind spots are a first-class output:** `gap = modeled_propensity_percentile − observed_count_percentile`; cells with **gap > 0.30** are flagged. Low recorded counts are *never* presented as "compliant." By construction blind spots have low observed counts, so most fall under the k-anonymity floor (count < 5): **408** are detected for analysis but **83** survive suppression for public per-cell display — the rest inform aggregate/zone-level guidance.
- **Patrol allocation:** the ranked priority list plus a **coverage-vs-effort curve** (cumulative violations captured as more locations are enforced) — the knee is the efficient deployment size. Top-N output is sanity-checked against BTP's ~154 high-density enforcement points.

### 5. Fairness & privacy  (`fairness.py`)

- **Temporal & spatial equity:** EEOC **4/5ths disparate-impact** rule (flag `group_rate/max < 0.8` → ratio came out **0.42**, flagged) and statistical-parity difference (flag > 0.10). The headline temporal finding: evening-peak enforcement share of **0.2%** against high modeled congestion risk. Fairness metrics are acknowledged as mutually incompatible at differing base rates (impossibility theorem); CurbIQ reports DI + parity and explains residual disparity.
- **Privacy by design under India's DPDP Act 2023:** public artifacts carry **H3 cell IDs / centroids only — never raw point lat/lon**; any cell or time-bucket with count < 5 is **suppressed (k-anonymity, ~33% of cells)**; vehicle plates are dropped or **SHA-256 + salted** and never reach the artifact layer. `validation_status` is profiled, with rejected tickets down-weighted and duplicates excluded from counts.

---

## Dashboard tour

A build-free single-page app (`web/`) over Leaflet + h3-js + Chart.js. Open `http://localhost:8000`.

- **Map — metric switcher (left panel):** recolour the H3 hex layer by **Enforcement priority**, **Hotspot intensity (Gi\* z)**, **Congestion impact (CIS)**, **Violation count**, **Forecast (next day)**, **Under-enforcement gap**, **Emergence risk** (forward 28-day probability a cell turns into a hotspot), or **Recoverable delay** (modeled congestion-ROI per cell). A live legend tracks the active scale; clicking a hex opens a tooltip with violations, priority/rank, Gi\* z, CIS, modeled extra-delay %, next-day forecast, top offence, and a blind-spot warning.
- **Overlays:** H3 hex layer · density heatmap · top-60 junctions (sized by count) · hotspot zones · under-enforcement blind spots · **emergence watch-list** (markers for the predicted-emerging cells = areas to watch, disjoint from currently-hot) · "significant hotspots only" filter. Plus offence-type and minimum-priority filters.
- **Overview tab:** the priority table ("where the carriageway chokes"), the coverage-vs-effort curve, the top-junctions table, and the congestion-ROI coverage curve ("**38 cells recover 50% of modeled delay**") that ranks cells by recoverable modeled delay.
- **Timing tab:** enforcement vs. modeled congestion-risk by hour (IST) — the evening-peak blind spot in one chart — plus the recommended per-hotspot 4-hour enforcement windows and citywide shift windows, carrying the **recording-bias note** (recorded times describe *when enforcement was logged*, heavily skewed to morning/early hours — busiest recorded hour 10:00 IST — not when violations occur; the genuine evening signal is the risk-vs-enforcement gap, not a literal "enforce at night" instruction). Also daily-violation trend, vehicle mix and top-offence breakdowns.
- **Model tab:** the forecast scorecard (PAI@5/20%, ROC-AUC, R², MAE, PEI@5%) — now noting the forecast uses **36 features** incl. peak-hour & heavy-vehicle lags — a model-vs-baselines PAI bar, feature importances, and an **emergence scorecard** (AUC **0.92 vs 0.82** baseline, **238** predicted emerging).
- **Equity tab:** the hourly under-enforcement gap, spatial-equity stats (disparate-impact ratio, parity diff, most under-enforced stations), and an emerging-hotspots breakdown.

The footer always shows the artifact version, record count, % of cells k-anon suppressed, and the license.

---

## Extended capabilities

Beyond the core hotspot → congestion → priority loop, CurbIQ ships five advanced modules plus production hardening:

1. **Congestion calibration** (`curbiq/calibration.py`) — validates the *modeled* CIS against probe-speed data (Spearman ρ) and re-fits its component weights by NNLS. A physically-grounded synthetic probe is bundled (ρ **0.70 → 0.76** after calibration, isotonic R² 0.62); drop in a real feed with `build_all.py --probe speeds.csv` (TomTom / Google / Uber Movement). → `/api/calibration`, Model tab.
2. **Live camera ingestion** (`curbiq/cv/`) — **real ONNX vehicle detection** (SSD-MobileNet COCO *or* YOLOv8) via onnxruntime + IOU tracking + dwell-time + shapely no-parking-zone geofencing + pixel→GPS homography, emitting violations **in the dataset schema** so live detections feed the very same analytics. A 28 MB SSD-MobileNet model is fetched by `./run.sh --with-cv` (or `scripts/get_cv_model.sh`); point `CURBIQ_YOLO_ONNX` at any YOLOv8 `.onnx` instead. Falls back to a `SimulationDetector` only if no model/runtime is present. Demo (real inference on a street image): `python scripts/cv_demo.py`.
3. **Geo-validation** (`curbiq/validate_geo.py`) — precision@N / recall of statistical hotspots vs BTP's recognized junctions (**72%** of top-50 within 300 m) and **39 novel off-junction hotspots** = candidate new deployment points. Plug in the official list with `--enforcement-points pts.csv`. → `/api/geo-validation`, map overlay.
4. **Patrol routing** (`curbiq/patrol.py`) — a prize-collecting VRP (OR-Tools, with a greedy+2-opt fallback) turning the priority ranking into balanced evening-shift routes + ETAs (**6 units, 100% of top-60 covered, ~145 km**). → `/api/patrol`, map polylines.
5. **Feature enrichment** (`curbiq/enrich.py`) — nearest-metro distance (OSM/Overpass, **81 Namma Metro stations**). Nearest-metro distance is a clean enrichment win again under the tuned model: it lifts holdout **PAI@5 from 12.60 (34-feature) to 12.72 (36-feature)** and is a **top-8 feature by importance (1268)** (R² ~flat). Retained on by default; toggle via `enrich`. Daily weather (Open-Meteo) is available (`enrich='all'`) but off by default since it hurt CV. Full enrichment A/B in the model card.

**Dashboard extras:** a **week-by-week time-slider** (animate hotspot evolution across 23 weeks) and an optional **deck.gl 3D extruded-hex** layer (feature-detected, degrades to 2D).

**Production hardening:** env-gated API-key auth (`CURBIQ_API_KEY` → `X-API-Key` on `/api/*`), an installable **PWA** (offline app-shell + artifact cache, `web/sw.js`), `/health` + `/version` endpoints, and a nightly refresh script (`scripts/refresh.sh` for cron). `build_artifacts()` stays idempotent, so the API/frontend contract never changes.

---

## Quickstart

**Prerequisites:** Python 3.11+ (developed on 3.13), ~1 GB free disk for the venv. No GDAL, geopandas, or torch — intentionally lean. (OR-Tools + Pillow are included; onnxruntime is optional, only for live CCTV.)

```bash
# 1. Clone / enter the repo
cd CurbIQ

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset (gzipped CSV) at:
#    data/raw/police_violations.csv.gz

# 5. Build everything (ETL -> analytics -> versioned artifacts + model).
#    Takes ~90s. Artifacts + model are git-ignored, so this step is required on
#    a fresh clone (and to reproduce from scratch on new data).
python build_all.py
#    Force a full ETL re-run from the raw CSV:
#    python build_all.py --rebuild-etl

# 6. Serve the read-only API + dashboard
uvicorn curbiq.api.main:app --port 8000

# 7. Open http://localhost:8000
```

> **PYTHONPATH note.** `curbiq` is imported as a top-level package, so the repo root must be on `PYTHONPATH`. If you run without activating the venv (or from a script), prefix commands with the repo root:
>
> ```bash
> PYTHONPATH=. python build_all.py
> PYTHONPATH=. uvicorn curbiq.api.main:app --port 8000
> ```

The dashboard pulls its CDN libraries (Leaflet, h3-js, Chart.js) from pinned `unpkg` versions, so an internet connection is needed the first time the page loads.

---

## Repository structure

```
CurbIQ/
├── run.sh                    # one-shot launcher (venv -> deps -> dataset -> build -> serve)
├── build_all.py              # end-to-end: ETL -> analytics -> artifacts + model
├── requirements.txt          # lean runtime (no GDAL/geopandas/torch)
├── README.md
├── curbiq/
│   ├── config.py             # single source of truth: all domain constants & params
│   ├── etl/
│   │   └── pipeline.py        # raw csv.gz -> cleaned/feature-engineered parquet (UTC->IST)
│   ├── features.py           # temporal / road-class / footprint / severity / confidence + H3
│   ├── spatial.py            # Gi*, global+local Moran's I (perm pseudo-p), BH-FDR, Mann-Kendall
│   ├── hotspots.py           # hotspots / zones / junctions / emerging
│   ├── congestion.py         # HCM -> BPR -> CIS (MODELED congestion impact)
│   ├── forecast.py           # LightGBM Poisson panel, walk-forward CV, PAI/PEI, baselines
│   ├── emergence.py          # forward 28-day hotspot-emergence risk (LightGBM binary, temporal holdout)
│   ├── timing.py             # per-hotspot 4h enforcement windows + citywide shift profiles
│   ├── scenario.py           # congestion ROI: recoverable modeled delay + coverage curve
│   ├── prioritize.py         # exposure-adjusted ranking, blind spots, coverage curve
│   ├── fairness.py           # temporal/spatial equity (4/5ths) + DPDP privacy helpers
│   ├── calibration.py        # probe-speed validation + NNLS CIS-weight re-fit
│   ├── enrich.py             # weather (Open-Meteo) + metro (Overpass) features
│   ├── validate_geo.py       # hotspots vs BTP junctions + novel-hotspot finder
│   ├── patrol.py             # OR-Tools / greedy VRP dispatch routing
│   ├── cv/                   # live camera ingestion (ONNX detect -> geofence -> records)
│   │   ├── detector.py        #   YOLOv8 / SSD-MobileNet backends + SimulationDetector
│   │   ├── geofence.py        #   no-parking zones + pixel->GPS homography
│   │   ├── tracker.py         #   IOU tracker + dwell-time
│   │   └── pipeline.py        #   frame -> violations -> dataset-schema records
│   ├── artifacts.py          # build_artifacts(): runs everything -> JSON + manifest
│   └── api/
│       └── main.py            # FastAPI read-only server (ETag/304, API-key gate, serves web/)
├── web/                      # build-free dashboard (Leaflet + h3-js + Chart.js + deck.gl) + PWA
│   ├── index.html / app.js / styles.css
│   └── sw.js / manifest.webmanifest / icon.svg   # installable PWA + offline shell
├── docs/                     # RESEARCH, ARCHITECTURE, MODEL_CARD, DATA_GOVERNANCE, RESULTS
├── data/
│   ├── raw/                   # police_violations.csv.gz  (you supply this)
│   ├── processed/             # violations.parquet  (ETL output)
│   └── artifacts/             # versioned JSON map/analytics layers + manifest.json
├── models/                   # forecast_lgbm.txt + feature_cols.json
└── scripts/                  # ad-hoc profiling utilities
```

---

## Limitations & caveats

CurbIQ is deliberate about what it does and does **not** claim:

- **Congestion impact is MODELED, not measured.** There is no speed or flow data. The CIS is an explainable HCM→BPR composite with visible component weights and citations — it estimates relative congestion *consequence*, not observed delay. It must always be labeled as such.
- **Recorded violations reflect ENFORCEMENT ACTIVITY, not true occurrence.** Counts encode patrol-shift bias: where and when officers are deployed. CurbIQ surfaces this directly via exposure-adjusted ranking, under-enforcement blind spots, and the temporal/spatial equity panels — and never presents low recorded counts as compliance.
- **No live feed yet.** The current build is a batch over a fixed Nov 2023 – Apr 2024 export. Forecasts are next-day given that window.
- **Inferred road context.** Road class and assumed lane counts are inferred from free-text addresses and engineering rules-of-thumb, not an authoritative road network; PCU and capacity factors are literature defaults, tunable in `config.py` and (optionally) calibratable against probe data if it becomes available.

### Real-time extension seam

`build_artifacts(df)` is pure and idempotent, the manifest is versioned, and the API/frontend depend only on the artifact schema. A future cron job or stream consumer can call `build_artifacts(df)` on fresh data (or incrementally) and atomically swap in a new versioned artifact directory — the FastAPI server and dashboard re-serve it with **zero code changes**, and the version-in-manifest makes it cache-bustable. That same seam is where live ASTraM/sensor feeds or an operator's known enforcement-capture rate (`ENFORCEMENT_CAPTURE_MULT` in `config.py`) would plug in.

---

## License & data source

**License: Apache-2.0** — matching the open-source norm for Bengaluru smart-mobility finalists (e.g. Bengaluru Mobility Challenge 2024).

**Data source.** CurbIQ is built on a geocoded, anonymized Bengaluru Traffic Police parking-violation export. We assert **lineage, not identity**: this is a BTP / ASTraM-style enforcement-analytics build aligned with Gridlock-2.0-style framing (classify congestion, detect violations, identify movement patterns) and BTP's real enforcement geography (~12 corridors / 43 junctions / 99 major roads / 154 high-density points). It is **not** a claim to be any specific named hackathon's exact dataset, and published BTP open data is aggregate-only — confirming this is a custom geocoded export. Privacy handling follows India's **DPDP Act 2023** by design (H3-only public layers, k-anonymity suppression, hashed plates).

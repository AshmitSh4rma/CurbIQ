# CurbIQ — System Architecture

**License:** Apache-2.0 · **Lineage:** Bengaluru Traffic Police (BTP) / ASTraM-style enforcement analytics · **Design brief:** see [`RESEARCH.md`](RESEARCH.md).

CurbIQ turns ~298k geocoded parking-violation records into three decision layers — statistical hotspots, a *modeled* congestion-impact score, and a bias-aware enforcement priority list — and serves them through a read-only API and a build-free dashboard.

The single architectural commitment is **precompute-then-serve**: every expensive spatial/ML computation runs once, offline, in an idempotent batch job that emits compact, versioned, privacy-safe JSON. The API and frontend never run spatial math; they only read those artifacts. This keeps request latency flat, makes every response CDN-cacheable, and gives us one reproducible audit trail per data vintage.

---

## 1. Dataflow & module diagram

```
                    raw/police_violations.csv.gz   (UTC timestamps, plates, point lat/lon)
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  curbiq/etl/pipeline.py   build_processed()                            │
  │   • read USECOLS, coerce geo, drop out-of-bbox / unparseable rows      │
  │   • nullify sentinels, coalesce updated_* vehicle fields               │
  │   • created_datetime: parse UTC ─► convert to IST wall-clock (CRITICAL)│
  │   • curbiq/features.py engineer():  offences→severity, vehicle→PCU,    │
  │     road-class regex, validation→confidence, peak_overlap, H3 r8/r9/r10│
  └──────────────────────────────────────────────────────────────────────┘
                                 │  (idempotent transform)
                                 ▼
                 data/processed/violations.parquet   (zstd, ~one row / ticket)
                                 │  load_processed()
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  curbiq/artifacts.py   build_artifacts(df)   ── the precompute core ── │
  │                                                                        │
  │   hotspots.py  ─► Gi*/Moran/LISA + FDR (res 9)   [via spatial.py]      │
  │   congestion.py─► HCM capacity loss ─► BPR delay ─► composite CIS      │
  │   forecast.py  ─► LightGBM Poisson panel (res 8 × day), walk-forward   │
  │   prioritize.py─► exposure-adjusted rank + blind spots + coverage curve│
  │   hotspots.py  ─► zones / junctions / emerging (Mann-Kendall)          │
  │   fairness.py  ─► temporal/spatial equity + DPDP privacy (k-anon)      │
  │                                                                        │
  │   merge → k_anon_suppress(count<5) → round → orjson                    │
  └──────────────────────────────────────────────────────────────────────┘
                    │                                         │
                    ▼                                         ▼
        data/artifacts/*.json  + manifest.json      models/forecast_lgbm.txt
                    │                                  + feature_cols.json
                    ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  curbiq/api/main.py   FastAPI (read-only)                             │
  │   • lifespan: load every artifact into memory, precompute MD5 ETag    │
  │   • GET /api/{name}: return bytes + ETag + Cache-Control; 304 aware    │
  │   • NO spatial math in any handler                                     │
  │   • mount web/ static at /                                             │
  └──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                 web/  (index.html · app.js · styles.css)
                 Leaflet + h3-js + Chart.js, zero build, CDN-pinned
```

`build_all.py` is the one-command entry point: it runs the ETL only if the parquet is missing (or `--rebuild-etl`), then calls `build_artifacts(df)`.

---

## 2. Per-module responsibilities & key functions

| Module | Responsibility | Key functions |
|---|---|---|
| `config.py` | Single source of truth for **all** domain constants — offence taxonomy & carriageway severity, vehicle PCU footprints, road-class regex+weights, HCM/BPR params, CIS weights, FDR target, forecast lags/windows/LGBM params, fairness thresholds, privacy (`K_ANON_MIN=5`, plate salt), BTP geography. | constants only |
| `etl/pipeline.py` | Raw `csv.gz` → cleaned, feature-engineered parquet. Idempotent. **UTC→IST conversion** before any temporal work. | `build_processed()`, `load_processed()` |
| `features.py` | Pure, testable row enrichment used by every downstream module. | `engineer()`, `classify_road()`, `primary_offence()`, `max_carriageway_severity()`, `peak_overlap()`, `time_of_day_bucket()`, `parse_int_list()`/`parse_str_list()` |
| `spatial.py` | Pure numpy/scipy/h3 spatial-stats engine. | `build_active_lattice()`, `build_weights()`, `getis_ord_gi_star()`, `z_to_p_two_sided()`, `benjamini_hochberg()`, `global_morans_i()`, `local_morans()`, `mann_kendall()`, `gi_confidence_band()` |
| `hotspots.py` | Cell/zone/junction/emerging hotspots. Analysis variable = **confidence-weighted** count. | `compute_hotspots()` (Gi*/Moran/LISA, res 9), `hotspot_zones()`, `junction_hotspots()`, `emerging_hotspots()`, `aggregate_cells()` |
| `congestion.py` | Congestion Impact Score — **modeled, never measured**. maneuver intensity → HCM capacity loss → BPR delay multiplier → 4-factor z-scored composite (0–100). | `compute_congestion()`, `hcm_capacity_loss()`, `bpr_delay_ratio()`, `nearest_junction_distance()`, `haversine_m()` |
| `forecast.py` | LightGBM Poisson spatiotemporal panel (res 8 × day, materialized zeros). Anti-leakage by construction. | `run_forecast()`, `make_features()`/`build_panel()`, `walk_forward()`, `baselines()`, `score_block()`, `pai_pei()` |
| `prioritize.py` | Fuses Gi* z, CIS, forecast into an **exposure-adjusted** rank; surfaces under-enforcement blind spots and a coverage-vs-effort curve. | `run_prioritization()`, `exposure()` |
| `fairness.py` | Makes patrol bias visible (temporal/spatial equity, 4/5ths disparate impact, statistical parity) + DPDP privacy helpers. | `run_fairness()`, `temporal_gap()`, `spatial_equity()`, `data_quality()`, `k_anon_suppress()`, `hash_plate()` |
| `artifacts.py` | Runs everything once, merges per-cell layers, applies k-anonymity, writes versioned JSON + model + manifest. | `build_artifacts()`, `compute_timeseries()` |
| `api/main.py` | Read-only artifact server with ETag/304 + Cache-Control. | `lifespan`, `_load_artifacts()`, `_serve()`, `api()`, `health()` |

---

## 3. The artifact JSON contract

`build_artifacts()` writes 10 data files + `manifest.json` to `data/artifacts/`, plus the trained model to `models/`. All numbers are rounded (3 dp); numpy types are serialized via orjson `OPT_SERIALIZE_NUMPY`. **Public layers carry H3 cell ids + centroids only — never raw point lat/lon — and any cell with `count < 5` is suppressed.**

| File | API name | Shape |
|---|---|---|
| `manifest.json` | `manifest` | `{name, version ("YYYYMM_YYYYMM"), generated_at (UTC ISO), license, dataset:{records, date_range, source}, config:{h3_primary, forecast_res, gi_k, fdr_q, cis_weights, k_anon}, files:[{file, bytes}], headline_metrics:{global_moran_I, n_hotspots, forecast_pai_at_5, forecast_roc_auc, evening_peak_enforcement_share}}` |
| `kpis.json` | `kpis` | flat dict: `total_violations, date_range, n_police_stations, n_junctions, n_h3_cells, n_hotspots, global_moran_I, global_moran_z, n_hotspot_zones, n_blind_spots, locations_for_50pct, city_delay_impact_index, evening_peak_enforcement_share, forecast_pai_at_5, forecast_roc_auc, top_station, top_offence` |
| `cells.json` | `cells` | `{resolution: 9, k_anon:{k,n_total,n_suppressed,frac_suppressed}, cells:[ {h3, lat, lon, count, gi_z, gi_band, is_hotspot, lisa_quadrant, cis_score, extra_delay_pct, capacity_loss, priority_score, priority_rank, forecast_area, under_enforcement_gap, is_blind_spot, zone_id, top_offence, top_vehicle} ]}` — the primary res-9 map layer |
| `forecast_cells.json` | `forecast` | `{resolution: 8, cells:[ {h3, lat, lon, predicted_next_day} ]}` — res-8 next-day prediction, kept where `predicted_next_day ≥ 1.0` |
| `zones.json` | `zones` | `list[ {zone_id ("Z000"…), n_cells, count, peak_gi_z, lat, lon, top_offence, top_vehicle, mean_peak_share} ]` — contiguous hotspot zones (top by count) |
| `junctions.json` | `junctions` | `list[ {junction_id, count, lat, lon, mean_severity, peak_share, top_offence, rank, count_pctile} ]` — top 60 named junctions |
| `emerging.json` | `emerging` | `{by_category:{category→count}, cells:[ {h3, lat, lon, category, trend, mk_z, mk_p, tau, total, active_week_frac} ]}` — Mann-Kendall trend typing (top 150) |
| `priority.json` | `priority` | `{summary:{…}, coverage_curve:{frac_locations[], frac_violations_captured[]}, top:[ {h3, lat, lon, priority_score, priority_rank, count, gi_z, cis_score, forecast_area, top_offence} ], blind_spots:[ {h3, lat, lon, under_enforcement_gap, propensity, count, cis_score} ]}` — blind spots filtered to `count ≥ 5` |
| `fairness.json` | `fairness` | `{temporal:{hour[], enforcement_share[], risk_share[], under_enforcement_gap[], most_under_enforced_hours[], evening_peak_enforcement_share}, spatial_equity:{n_stations, disparate_impact_ratio, disparate_impact_flag, statistical_parity_diff, statistical_parity_flag, most_under_enforced[], most_over_enforced[]}, data_quality:{…}, privacy_policy:{…}}` |
| `timeseries.json` | `timeseries` | descriptive breakdowns: `{daily, weekly, hourly_ist, day_of_week, vehicle_category, top_vehicle_types, top_offences, road_class, top_stations, time_of_day}` |
| `model_metrics.json` | `model-metrics` | `{forecast:{cv_mean, holdout:{month, metrics}, baselines:{last_week, rolling_7, ewm}, feature_importances (top 20), best_iteration, n_panel_rows}, hotspots:{…stats…}, congestion:{…summary…}, prioritization:{…summary…}}` |
| `models/forecast_lgbm.txt` | — | saved LightGBM booster (text format) |
| `models/feature_cols.json` | — | ordered feature-column list used at train time |

The manifest's `files[]` records each artifact's byte size; the `version` string (e.g. `202311_202404`) is derived from the data's IST date span and is the cache-bust key (see §4).

---

## 4. API endpoints & caching strategy

The API (`curbiq/api/main.py`) is intentionally trivial: at startup a `lifespan` handler reads every artifact into an in-memory `_CACHE` dict and precomputes a strong **MD5 ETag** per file. There is no spatial computation, no DataFrame, no model inference in any request path.

Endpoints:

| Method · Path | Returns |
|---|---|
| `GET /health` | `{status, artifacts:[…names…]}` — liveness + which artifacts loaded |
| `GET /api/{name}` | the named artifact bytes; `name` ∈ {`manifest, kpis, cells, forecast, zones, junctions, emerging, priority, fairness, timeseries, model-metrics`}. Unknown name → 404 with `available` list; not-yet-built → 503 |
| `GET /` (and static) | the `web/` dashboard (mounted last so `/api/*` and `/health` win) |

**Caching:**
- Every artifact response carries `ETag: "<md5>"` and `Cache-Control: public, max-age=300`.
- If the client sends `If-None-Match` equal to the current ETag, the server returns **304 Not Modified** with the headers and no body.
- The ETag changes only when the artifact bytes change — i.e. only after a rebuild — so clients revalidate cheaply and re-download only what actually moved.
- **Version lives in the manifest** (and the data vintage in `manifest.version`). The frontend reads `/api/manifest` first; a CDN/reverse-proxy can additionally key on that version to serve immutable, far-future-cacheable copies and bust the whole layer atomically on a new vintage.
- CORS is open for `GET` only — the surface is read-only by construction.

---

## 5. Build-free frontend rationale

`web/` is a single `index.html` + `app.js` + `styles.css`, served as static files. The map is **Leaflet** with **h3-js** (render H3 cells as polygons from the `h3` ids in `cells.json`) and **Chart.js** (time series + fairness bars), all loaded from CDN with pinned versions.

Why no build step:
- **Zero toolchain / zero lockfile drift.** No npm, bundler, or node_modules — nothing to break silently, nothing to install on a disk-constrained host. Pinning exact CDN versions is the lockfile.
- **The data contract is the only coupling.** The frontend consumes the same stable artifact JSON the API serves; changing analytics never requires a frontend rebuild, only a re-fetch.
- **Trivially deployable & demonstrable.** A judge or operator can open the dashboard from any static host (or `file://`-style) against the artifact server; the precompute-then-serve split means the UI is always fast regardless of dataset size.
- **H3-only rendering reinforces privacy** — the client literally never receives point coordinates, only cell ids/centroids (see [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md)).

---

## 6. Real-time ingestion seam (future cron / stream)

`build_artifacts(df)` is a **pure, idempotent function of its input frame** — it takes a DataFrame (defaulting to `load_processed()`), writes a fixed set of files, and returns the manifest. Nothing in it mutates global state or depends on wall-clock beyond stamping `generated_at`. This is the seam for going from batch to near-real-time without touching the API or frontend:

- **Same contract, new vintage.** A future cron job or stream consumer that appends fresh tickets, re-runs the ETL transform, and calls `build_artifacts(df)` re-emits the *identical schema* and a new `manifest.version`. The API picks it up on the next `_load_artifacts()` (restart or a reload hook); ETags change, clients revalidate, the dashboard refreshes — no code change anywhere downstream.
- **Idempotency guarantees.** The ETL (`build_processed`) and the artifact builder both overwrite deterministically: given the same input rows they produce byte-stable outputs (rounding + fixed seeds in `spatial.py`/`forecast.py`), so re-runs are safe and diffable.
- **Versioned, atomic swap.** Because the version is encoded in the manifest (and intended to live in the artifact path/CDN key), a new build can be staged and swapped atomically; readers either see the old complete vintage or the new one, never a half-written mix.
- **Where the heavy work stays.** All Gi*/Moran/CIS/forecast cost is paid in the batch, off the request path, exactly once per vintage — so scaling to higher ingestion frequency is a scheduling problem (run the batch more often), not an API-latency or frontend problem.

# Data Governance & Privacy — CurbIQ

**Regulation:** India's Digital Personal Data Protection (DPDP) Act, 2023 (Rules 2025; full compliance deadline 13 May 2027)
**Posture:** privacy by design — raw point coordinates and vehicle plates never leave the processing layer
**License:** Apache-2.0
**Where enforced in code:** [`curbiq/config.py`](../curbiq/config.py) (privacy constants), [`curbiq/fairness.py`](../curbiq/fairness.py) (`k_anon_suppress`, `hash_plate`), [`curbiq/artifacts.py`](../curbiq/artifacts.py) (suppression applied at write time)

> CurbIQ processes data that is personal under the DPDP Act — vehicle registration numbers identify a person, and a precise lat/lon + timestamp is re-identifying. The governance model below is built into the pipeline, not bolted on: the public artifact layer is structurally incapable of exposing point locations or plates.

---

## 1. Data minimization

- The ETL reads only the columns it needs (`USECOLS` in [`etl/pipeline.py`](../curbiq/etl/pipeline.py)); columns that are 100% null or redundant are never loaded.
- Records outside the Bengaluru bounding box or with unparseable timestamps/geo are dropped at ingest.
- The processed parquet retains personal fields (point lat/lon, `vehicle_number`) **only in the internal processing layer** for analytics; these fields are deliberately excluded from every public artifact (§2–§4).
- Aggregation is the default unit of analysis: hotspots, congestion, forecasting and prioritization all operate on H3-cell or junction aggregates, not individuals.

## 2. Public granularity — H3 only, never raw point lat/lon

Every public/API artifact carries **H3 cell ids and cell centroids only** — never the original incident lat/lon. This is the single most important re-identification control: an aggregated cell centroid (~174 m res-9 / ~530 m res-8 edge) cannot be traced to a parked car at an address, whereas a raw point + timestamp can.

- The map layer (`cells.json`) emits `{h3, lat, lon, …}` where `lat`/`lon` are the **cell centroid**, computed from the H3 id, not the source point.
- The frontend renders cells from `h3` ids via h3-js and literally never receives source coordinates (see [`ARCHITECTURE.md`](ARCHITECTURE.md) §5).
- Junction artifacts use the *mean* centroid of a named junction's events, not individual points.

## 3. k-anonymity suppression (count < 5)

`config.K_ANON_MIN = 5`. Before any cell layer is written, [`artifacts.py`](../curbiq/artifacts.py) calls `k_anon_suppress(cells, "count", 5)`, which **drops every cell whose counted-violation total is below 5**. Sparse cells are exactly where a small count + a centroid could re-identify a specific incident, so they are removed from the public layer entirely.

- The suppression report `{k, n_total, n_suppressed, frac_suppressed}` is embedded in `cells.json` under `k_anon` and logged at build time — making the **utility-vs-privacy trade-off auditable** (~33% of cells suppressed on this vintage).
- The same `count ≥ 5` floor is applied to the `priority.json` blind-spot list.
- This protects against the small-cell disclosure risk that pure H3 aggregation alone does not.

## 4. Plate handling — SHA-256 + salt

`config.HASH_PLATES = True`, with a versioned salt (`PLATE_SALT`, rotated in production via env). `fairness.hash_plate(plate)` returns `sha256(f"{salt}:{plate}")[:16]`.

- Vehicle registration numbers are **never carried into the artifact layer** at all — no analytic surface needs them, so they stop at processing.
- Where a plate must appear in any internal export, it is salted-hashed (one-way, salt-rotated) so plates are not directly stored or exposed and cross-dataset linkage is blocked.

## 5. `validation_status` handling

Recorded tickets vary in trustworthiness; counting dismissed/duplicate tickets as confirmed violations would both bias analytics and over-represent individuals. CurbIQ handles this explicitly (`features.engineer`, weights in `config.VALIDATION_CONFIDENCE`):

- A per-record **confidence weight** is assigned: `approved`=1.00, `processing`/`created1`/`NULL`=0.60, `rejected`=0.15, `duplicate`=0.00.
- `is_counted = (status != "duplicate")` — duplicates are excluded from all counts; rejected tickets are heavily down-weighted (the hotspot analysis variable is the **confidence-weighted** count, not the raw count).
- `fairness.data_quality()` profiles the `validation_status` distribution and **guards against constant-flag false signals** (the "all-TRUE trap"): it flags `data_sent_to_scita` as near-constant if its TRUE-share is >0.95 or <0.05, so a degenerate flag is never mistaken for signal.

## 6. Retention, versioning & reproducibility

- Artifacts are **versioned by data vintage** — `manifest.version` is `"YYYYMM_YYYYMM"` derived from the IST date span (e.g. `202311_202404`), with `generated_at` (UTC) and per-file byte sizes recorded.
- The pipeline is **idempotent and deterministic** (fixed seeds in spatial permutations and the forecaster; deterministic rounding), so any vintage can be rebuilt and diffed for audit.
- This supports DPDP storage-limitation/accountability principles: a new vintage atomically supersedes the prior one, and the manifest is the retention/version record. Operators set the underlying raw-data retention window per their policy; the artifact layer keeps only what the current vintage needs.

---

## 7. Bias & fairness safeguards + metrics

Recorded violations are **enforcement events, not ground-truth occurrence**. Treating them as occurrence launders patrol bias into "where the problems are" and risks a predictive-policing feedback loop. CurbIQ makes the bias measurable and corrects for it (`fairness.py`, `prioritize.py`):

- **Exposure-adjusted ranking** (`prioritize.run_prioritization`): production priority divides counts by an exposure proxy `E_cell = active-hours + offence-variety + temporal-spread` (Laplace-smoothed by `α = median(E)`), breaking the patrol→record→rank→patrol loop. The **raw↔adjusted Spearman rank-shift** is published (≈0.98 on this vintage) as evidence the top hotspots are genuine, not patrol artifacts.
- **Under-enforcement blind spots** (first-class output): cells where `modeled_propensity_percentile − observed_count_percentile > 0.30` are flagged (402 here). Low-count zones are **never** presented as "compliant."

**Fairness metrics** (`fairness.spatial_equity` / `temporal_gap`), with the thresholds in `config.py`:

| Metric | Definition | Flag threshold |
|---|---|---|
| **Disparate-impact ratio** (EEOC 4/5ths rule) | `min(coverage_ratio) / max(coverage_ratio)` across police stations, where `coverage_ratio = observed_share / modeled-need_share` | `< 0.80` (`DISPARATE_IMPACT_FLAG`) — **0.42 on this vintage → flagged** |
| **Statistical-parity difference** | `max | observed_share − need_share |` across stations | `> 0.10` (`STAT_PARITY_FLAG`) |
| **Under-enforcement gap** | per-cell `propensity_pctile − observed_pctile` | `> 0.30` (`UNDER_ENFORCEMENT_GAP_FLAG`) |
| **Temporal under-enforcement** | `risk_share − enforcement_share` by IST hour | headline: evening-peak (17–20h) enforcement share ≈ **0.2%** despite high congestion risk |

- **Impossibility caveat (documented).** Group-fairness metrics are mutually incompatible at differing base rates; CurbIQ reports disparate-impact + parity-difference together and treats residual disparity as a finding to explain, not a number to optimize away.

---

## 8. Compliance summary

| DPDP / fairness principle | CurbIQ control | Code |
|---|---|---|
| Data minimization | `USECOLS`, bbox/time filtering, aggregate-only analysis | `etl/pipeline.py` |
| Purpose limitation / no individual targeting | cell-aggregate outputs; no per-plate prediction | `forecast.py`, `prioritize.py` |
| Personal-data protection (location) | H3 cell ids + centroids only in all public artifacts | `artifacts.py`, `api/main.py`, `web/` |
| Personal-data protection (plates) | dropped from artifacts; SHA-256 + rotated salt elsewhere | `fairness.hash_plate`, `config.PLATE_SALT` |
| Small-cell disclosure | k-anonymity suppression, `count < 5`, with audit report | `fairness.k_anon_suppress`, `config.K_ANON_MIN` |
| Data quality / accuracy | validation-status confidence weighting; duplicates excluded; constant-flag guard | `features.engineer`, `fairness.data_quality` |
| Storage limitation / accountability | versioned, deterministic, reproducible artifacts + manifest | `artifacts.build_artifacts` |
| Non-discrimination / fairness | exposure-adjusted ranking, blind spots, 4/5ths + parity + temporal-gap metrics | `prioritize.py`, `fairness.py` |

# Gridlock Hackathon 2.0 — Round 2 submission (copy-paste pack)

Problem statement: **Poor Visibility on Parking-Induced Congestion** — *"How can AI-driven parking
intelligence detect illegal-parking hotspots and quantify their impact on traffic flow to enable
targeted enforcement?"*

---

## Title
**CurbIQ — AI parking intelligence: illegal-parking hotspots, modeled congestion impact & targeted enforcement for Bengaluru**

## Theme
Pick the theme closest to **congestion management / traffic-flow optimization / AI for urban mobility**
(the problem statement is parking-induced congestion + enforcement). If a "data analytics" or
"smart enforcement" theme exists, that also fits.

## Description (paste into the Description box)
CurbIQ turns **298,125 geocoded Bengaluru Traffic Police parking-violation records (Nov 2023–Apr 2024)**
into three operational layers — and is scrupulous about what is *measured* vs *modeled*.

**1. Where illegal parking clusters (detection).** Getis-Ord Gi\* / Moran's I on an H3 res-9 street-block
grid with Benjamini-Hochberg FDR → **112 statistically-significant hotspots** (global Moran's I 0.343,
z 42.3), merged into enforcement zones and ranked named junctions.

**2. How much it chokes traffic (modeled impact).** An explainable HCM-2000 capacity-loss → BPR
volume-delay pipeline yields a Congestion Impact Score per cell (no speed data is faked — it is labeled
*modeled* everywhere). A **what-if ROI layer** shows that clearing just **38 cells recovers 50%** of the
city's modeled parking-induced delay.

**3. Who to enforce, where and when (targeted enforcement).** An **exposure-adjusted** priority ranking
breaks the patrol→record→rank→patrol bias loop (raw↔adjusted Spearman 0.98), surfaces **408
under-enforcement blind spots**, and shows **84 locations cover 50%** of violations. A LightGBM Poisson
**forecast** (36 features, walk-forward validated, no leakage) predicts next-day hotspot pressure —
holdout **PAI@5% 12.7×, ROC-AUC 0.92, R² 0.65**, beating last-week/rolling-7/EWM baselines — and a
forward **emergence classifier** flags **238 cells likely to *become* hotspots in 28 days (AUC 0.92 vs
0.82 baseline)**. Plus OR-Tools patrol routing and per-cell enforcement time-windows.

**Built for the real world & for trust:** privacy by design under India's DPDP Act 2023 (H3-only public
layers, k-anonymity suppression of cells with <5 records, hashed plates); every metric reported honestly
(it openly states where it ties baselines and labels all modeled/synthetic components). Reproducible
offline pipeline (`build_all.py`, ~90s, no GPU) → versioned JSON artifacts → fast read-only FastAPI →
build-free Leaflet dashboard.

**Stack:** Python · NumPy/pandas · scikit-learn · LightGBM · H3 · shapely · FastAPI · Leaflet/Chart.js · OR-Tools.

## Instructions to Run (paste into the Instructions box)
```bash
# Python 3.11+ (developed on 3.13). No GPU required. ~1 GB disk for the venv.
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place the dataset at data/raw/police_violations.csv.gz — the anonymized BTP / HackerEarth
# challenge CSV (~16 MB; git-ignored as raw input). Then build artifacts + model (~90s):
python build_all.py

# Serve the read-only API + dashboard, then open http://localhost:8000
uvicorn curbiq.api.main:app --port 8000

# Run the test suite (optional): pytest -q     # 206 tests
```

## Field checklist
- **Repository URL** — **https://github.com/AshmitSh4rma/CurbIQ**
- **Source Code** — `git archive --format=zip -o curbiq.zip HEAD` (or zip the repo excluding `.venv/` and `data/raw/`).
- **Demo Link** — **https://ashmitsh4rma.github.io/CurbIQ/** — live GitHub Pages build of the dashboard on the privacy-safe artifacts (no backend needed; full map + all four analytics tabs).
- **Video URL** — 2–3 min screen-record: map metric switcher (priority/hotspot/CIS/forecast/emergence/recoverable-delay) → click a hotspot → the 4 analytics tabs → the evening-peak blind-spot chart.
- **Presentation** — slide-by-slide outline in **`PITCH.md`** (12 slides); build it into a deck (Keynote/PPT/Slides) and upload.
- **Snapshots** — screenshot the map (hotspot + emergence overlays) and the Model + Equity tabs.

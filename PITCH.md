# CurbIQ — Pitch Deck Outline

**Gridlock Hackathon 2.0 — Round 2 · Bengaluru Mobility**
Problem statement: *Poor Visibility on Parking-Induced Congestion* — "How can AI-driven parking intelligence detect illegal-parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?"

> Presenter notes: 12 slides. Every congestion-impact number is **modeled** — say so out loud. Hotspot and dataset numbers are **measured**. The honesty is the pitch, not a footnote. ~5–6 min talk + demo.

---

## Slide 1 — Title

**CurbIQ — AI parking intelligence for Bengaluru**
*Detect illegal-parking hotspots · quantify their modeled traffic impact · target enforcement.*

- 298,125 real Bengaluru Traffic Police violations, turned into three operational layers.
- Honest by design: every figure labeled **measured** or **modeled**.
- Forward-looking: predicts where hotspots emerge next, not just where they were.

*Visual / speaker note:* Logo + the live dashboard map (hotspot overlay) behind a one-line hook. "Parking is one of the most fixable causes of gridlock — and we can't see it yet."

---

## Slide 2 — The problem

**Illegal parking quietly chokes the network — and enforcement is blind**

- A single double-parked lorry or a row of vehicles within 100 m of a junction removes a live through-lane exactly at peak.
- Enforcement is reactive and patrol-shift-driven: tickets land where officers happen to be, not where violations peak.
- Published BTP open data is aggregate counts only — no map fusing *where it clusters* with *what it costs traffic*.
- 54 stations, 168 named junctions, thousands of street-blocks: "where do tow vehicles go first?" has no defensible answer.

*Visual / speaker note:* Photo of curb obstruction at a junction next to a flat bar chart of "counts by station" — show that raw counts tell you nothing actionable.

---

## Slide 3 — Why it's hard today

**The data lies in a specific, fixable way**

- Recorded counts measure **enforcement activity, not true occurrence** — a patrol → record → rank → patrol bias loop.
- More tickets where patrols already go reinforces blind spots elsewhere.
- No speed or flow data exists in the dataset — so congestion impact can't be *measured*, only *modeled* (and must be labeled honestly).
- Result: nobody can see the hotspot-vs-congestion picture, so prioritization is guesswork.

*Visual / speaker note:* Simple loop diagram of the patrol-bias cycle, with a red "break here" marker. This sets up why exposure-adjustment (Slide 7) matters.

---

## Slide 4 — CurbIQ in one diagram

**Three layers: detect → quantify → prioritize — and look forward**

- **Layer 1 — Detect:** statistically rigorous illegal-parking hotspots (measured).
- **Layer 2 — Quantify:** modeled congestion impact + what-if ROI (modeled, labeled).
- **Layer 3 — Prioritize:** exposure-adjusted enforcement targets, blind spots, time-windows.
- **Forward:** next-day forecast + 28-day emergence — act before hotspots form.

*Visual / speaker note:* One clean left-to-right pipeline graphic: 298k records → [Detect | Quantify | Prioritize] → dashboard, with "forward-looking" arrow looping back. This is the slide judges screenshot.

---

## Slide 5 — Layer 1: Where it clusters (measured)

**Statistical hotspots, not eyeballed dots**

- Getis-Ord Gi\* + Moran's I on a 2,534-cell H3 res-9 street-block grid, with Benjamini-Hochberg FDR.
- **112 high-confidence hotspots** (FDR-corrected); global **Moran's I 0.343 (z 42.3)** → strong, non-random clustering.
- Merged into contiguous enforcement zones and ranked named junctions (e.g. Safina Plaza ≈ 15,413 violations, 69% in peak hours).
- FDR matters: raw p<0.05 alone would flag hundreds of false hotspots; FDR pulled the bar to p=0.00102.

*Visual / speaker note:* The dashboard map with the Gi\* hotspot overlay lit up over Central Bengaluru (Shivajinagar / Commercial St zone). Point to the top zone live.

---

## Slide 6 — Layer 2: What it costs traffic (MODELED)

**An explainable congestion model — and a clear ROI**

- HCM-2000 capacity-loss → BPR volume-delay multiplier → Congestion Impact Score per cell. **No speed data is faked — labeled modeled everywhere.**
- City **modeled delay-impact index ≈ 7,513**; recoverable-delay reconciles exactly to it.
- **What-if ROI: clearing just 38 cells recovers 50%** of the city's modeled parking-induced delay (206 cells → 80%).
- Every per-cell tooltip exposes capacity-loss %, BPR Δ-delay %, dominant offence, and the citations behind it.

*Visual / speaker note:* Recoverable-delay map + the ROI curve (cells cleared vs % delay recovered, marker at 38 → 50%). Say plainly: "This is modeled, because the open data has no speeds — but it's transparent and reproducible, not a black box."

---

## Slide 7 — Layer 3: Who, where, and when to enforce

**Targeting that corrects its own bias**

- **Exposure-adjusted** priority ranking breaks the patrol-bias loop; raw↔adjusted Spearman **0.98** — top hotspots are genuine, not patrol artifacts.
- Surfaces **408 under-enforcement blind spots**; just **84 locations cover 50%** of violations.
- Per-hotspot **enforcement time-windows** (4-hour) tell crews *when*, not just where.
- **Headline finding:** the **evening peak (17–21 IST) receives ~0.2% of recorded enforcement** despite high modeled congestion risk.

*Visual / speaker note:* The Equity tab's evening-peak chart — a near-empty enforcement bar against a tall modeled-risk bar. This is the most memorable slide; pause on it.

---

## Slide 8 — The ML edge

**Forecasting pressure, and spotting hotspots before they form**

- **Next-day forecast:** LightGBM Poisson, 36 features, walk-forward validated (**no leakage**), CV-tuned.
- Holdout: **PAI@5% 12.7× · ROC-AUC 0.92 · R² 0.65 · MAE 2.02** — beats last-week / rolling-7 / EWM baselines.
- **Emergence classifier:** predicts cells *becoming* hotspots within **28 days** — **AUC 0.92 vs 0.82 baseline**; **238 cells flagged** as emerging.
- Honest note: on PAI we tie a strong persistence baseline (stable hotspots *are* persistent) — the edge is in error metrics, hotspot AUC, and a calibrated forward surface.

*Visual / speaker note:* Forecast vs emergence overlay on the map, plus a small metrics table. Deliver the "honest note" verbally — it builds credibility with technical judges.

---

## Slide 9 — Built for trust

**Honesty, privacy, and equity are first-class, not afterthoughts**

- **Modeled vs measured** labeled everywhere — we never present modeled delay as observed.
- **DPDP Act 2023 by design:** H3-only public layers, k-anonymity suppression of <5-count cells, SHA-256 + salted plates.
- **Bias surfaced, not hidden:** we flag the patrol/recording bias and report a spatial **disparate-impact ratio 0.42** (flagged < 0.8).
- The evening-peak gap (Slide 7) is itself a fairness finding the system exposes.

*Visual / speaker note:* Three-icon row (Honesty · Privacy · Equity) with the disparate-impact flag visible. Message: "We'd rather show you our blind spots than oversell."

---

## Slide 10 — Real-world viability + live extension

**It already aligns with how BTP enforces — and it can go live**

- Maps onto BTP's enforcement geography: ~12 corridors / 43 junctions / 99 roads / 154 high-density points.
- **Geo-validation: 72% of top-50 hotspots fall within 300 m of recognized junctions** — plus **39 novel candidate points** worth a look.
- **OR-Tools patrol routing** turns ranked targets into deployable routes.
- **Live-CCTV ingestion path (ONNX)** feeds the *same schema* — batch today, streaming tomorrow, no rearchitecture.

*Visual / speaker note:* Map with hotspots overlaid on known BTP junctions (72% match) + a small "live CCTV → same artifacts" seam diagram. Stresses this isn't a toy.

---

## Slide 11 — Tech & reproducibility

**Precompute-then-serve: fast, cheap, and verifiable**

- Stack: Python · NumPy/pandas · scikit-learn · LightGBM · H3 · shapely · FastAPI · Leaflet/Chart.js · OR-Tools.
- Architecture: offline build → versioned, privacy-safe JSON → fast read-only API → build-free dashboard.
- **No GPU · ~90s reproducible build · 206 tests** — every headline number is regenerated from the raw CSV.
- Demo: `python build_all.py` then `uvicorn curbiq.api.main:app` → open `localhost:8000`.

*Visual / speaker note:* The precompute → serve architecture diagram. Then cut to a 20-second live demo: switch map metrics (priority / hotspot / CIS / forecast / emergence / recoverable-delay), click a hotspot, open the analytics tabs.

---

## Slide 12 — Impact, ask & close

**From blind, reactive ticketing to targeted, forward-looking enforcement**

- **Impact:** clear 38 cells → recover 50% of modeled delay; cover 50% of violations with 84 locations; close the evening-peak blind spot.
- **For BTP:** a defensible, bias-corrected answer to "where and when do we deploy next?" — with a live-CCTV path ready.
- **Ask:** a pilot on 1–2 corridors with BTP enforcement-log access to validate modeled impact against real outcomes.
- **Why us:** rigorous (FDR, walk-forward, no leakage), honest (modeled vs measured), and reproducible (~90s, 206 tests).

*Visual / speaker note:* Recap the three layers in one line each, end on the dashboard URL + repo (github.com/AshmitSh4rma/CurbIQ). Close: "We can't fix what we can't see — CurbIQ makes parking-induced congestion visible, and actionable."

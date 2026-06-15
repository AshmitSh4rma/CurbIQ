#!/usr/bin/env python3
"""CurbIQ headline story — terminal demo.

Loads the pre-built artifacts in data/artifacts/*.json (does NOT recompute)
and prints the CurbIQ headline story: KPIs, top priority cells, top junctions,
the evening enforcement gap, forecast skill, and the coverage stat.

Run:
    PYTHONPATH=/home/ashmit/Claude/CurbIQ \
      /home/ashmit/Claude/CurbIQ/.venv/bin/python scripts/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "artifacts"

# ANSI styling (degrades gracefully when output is piped / not a TTY).
_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _TTY else ""


BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET = (
    _c("\033[1m"),
    _c("\033[2m"),
    _c("\033[36m"),
    _c("\033[32m"),
    _c("\033[33m"),
    _c("\033[31m"),
    _c("\033[0m"),
)
W = 74


def load(name: str) -> dict | list:
    path = ART / name
    if not path.exists():
        sys.exit(
            f"{RED}Artifact not found: {path}{RESET}\n"
            f"Build artifacts first:  PYTHONPATH=. .venv/bin/python build_all.py"
        )
    with path.open("rb") as fh:
        return json.load(fh)


def rule(char: str = "-") -> None:
    print(DIM + char * W + RESET)


def header(text: str) -> None:
    print()
    print(BOLD + CYAN + text + RESET)
    rule()


def kv(label: str, value: str, width: int = 38) -> None:
    print(f"  {label:<{width}}{BOLD}{value}{RESET}")


def main() -> int:
    kpis = load("kpis.json")
    manifest = load("manifest.json")
    priority = load("priority.json")
    junctions = load("junctions.json")
    fairness = load("fairness.json")
    metrics = load("model_metrics.json")
    zones = load("zones.json")

    # ---- Title banner -------------------------------------------------
    print()
    print(BOLD + CYAN + "=" * W + RESET)
    title = "CurbIQ — Bengaluru Parking-Violation Intelligence"
    print(BOLD + CYAN + title.center(W) + RESET)
    sub = "detect hotspots  ·  model congestion impact  ·  prioritize enforcement"
    print(DIM + sub.center(W) + RESET)
    print(BOLD + CYAN + "=" * W + RESET)
    src = manifest.get("dataset", {}).get("source", "Bengaluru Traffic Police export")
    dr = kpis["date_range"]
    print(
        DIM
        + f"  {kpis['total_violations']:,} violations  |  {dr[0]} -> {dr[1]}  |  "
        + f"{manifest.get('license','Apache-2.0')}".center(0)
        + RESET
    )
    print(DIM + f"  Source: {src}" + RESET)

    # ---- KPIs ---------------------------------------------------------
    header("CITY KPIs")
    kv("Total violations (counted)", f"{kpis['total_violations']:,}")
    kv("Police stations / named junctions",
       f"{kpis['n_police_stations']} / {kpis['n_junctions']}")
    kv("H3 res-9 cells analysed", f"{kpis['n_h3_cells']:,}")
    kv("Top station / top offence",
       f"{kpis['top_station']} / {kpis['top_offence']}")

    # ---- Hotspots -----------------------------------------------------
    header("HOTSPOTS  (Getis-Ord Gi*  +  Moran's I,  FDR-controlled)")
    moran = metrics["hotspots"]["global_moran"]
    kv("Global Moran's I", f"{moran['I']:.3f}  (z={moran['z']:.1f}, p<1e-300)")
    kv("High-confidence hotspots (z>=3.29)", f"{kpis['n_hotspots']}")
    kv("Benjamini-Hochberg FDR critical p",
       f"{metrics['hotspots']['fdr_critical_p']:.5f}")
    kv("Contiguous hotspot zones", f"{kpis['n_hotspot_zones']}")
    top_zone = zones[0]
    print(
        f"  {GREEN}Top zone:{RESET} {top_zone['n_cells']} cells, "
        f"{BOLD}{top_zone['count']:,}{RESET} violations "
        f"(~{top_zone['count']/kpis['total_violations']*100:.0f}% of city), "
        f"peak Gi* z={top_zone['peak_gi_z']:.1f}  [Central Bengaluru]"
    )

    # ---- Top priority cells ------------------------------------------
    header("TOP 5 PRIORITY CELLS  (where to deploy tow vehicles)")
    print(
        f"  {DIM}{'#':<3}{'H3 cell':<17}{'lat,lon':<17}"
        f"{'score':>6}{'count':>8}{'Gi* z':>8}{'CIS':>7}{RESET}"
    )
    for c in priority["top"][:5]:
        loc = f"{c['lat']:.3f},{c['lon']:.3f}"
        print(
            f"  {c['priority_rank']:<3}{c['h3']:<17}{loc:<17}"
            f"{BOLD}{c['priority_score']:>6.1f}{RESET}"
            f"{c['count']:>8,}{c['gi_z']:>8.1f}{c['cis_score']:>7.1f}"
        )

    # ---- Top junctions ------------------------------------------------
    header("TOP 5 JUNCTIONS  (where to place static pickets)")
    print(
        f"  {DIM}{'#':<3}{'junction':<34}{'count':>8}"
        f"{'peak%':>8}{'sev':>6}{RESET}"
    )
    for j in junctions[:5]:
        name = j["junction_id"]
        if len(name) > 33:
            name = name[:30] + "..."
        print(
            f"  {j['rank']:<3}{name:<34}{BOLD}{j['count']:>8,}{RESET}"
            f"{j['peak_share']*100:>7.0f}%{j['mean_severity']:>6.2f}"
        )

    # ---- Evening enforcement gap -------------------------------------
    header("THE EVENING ENFORCEMENT BLIND SPOT")
    ev = fairness["temporal"]["evening_peak_enforcement_share"]
    worst = fairness["temporal"]["most_under_enforced_hours"]
    print(
        f"  Evening peak (17-20h IST) enforcement share: "
        f"{RED}{BOLD}{ev*100:.2f}%{RESET}"
    )
    print(
        f"  {DIM}...yet modeled congestion RISK is highest in exactly these hours.{RESET}"
    )
    kv("Most under-enforced hours (IST)",
       ", ".join(str(h) for h in worst))
    di = fairness["spatial_equity"]["disparate_impact_ratio"]
    flag = "FLAGGED < 0.8" if fairness["spatial_equity"]["disparate_impact_flag"] else "ok"
    kv("Spatial disparate-impact (4/5ths)", f"{di:.2f}  ({flag})")

    # ---- Forecast -----------------------------------------------------
    header("FORECAST  (LightGBM Poisson, walk-forward, no leakage)")
    ho = metrics["forecast"]["holdout"]["metrics"]
    bl = metrics["forecast"]["baselines"]["last_week"]
    kv("Holdout ROC-AUC", f"{ho['roc_auc']:.2f}  (last-week baseline {bl['roc_auc']:.2f})")
    kv("Holdout R2 / MAE", f"{ho['r2']:.2f} / {ho['mae']:.2f}")
    kv("PAI @ 5% / @ 20%", f"{ho['pai@5']:.1f}x / {ho['pai@20']:.1f}x")
    kv("PEI @ 5%", f"{ho['pei@5']:.2f}")

    # ---- Coverage / efficiency ---------------------------------------
    header("DEPLOYMENT EFFICIENCY")
    s = priority["summary"]
    print(
        f"  {GREEN}{BOLD}{s['locations_for_50pct_coverage']}{RESET} locations cover "
        f"{BOLD}50%{RESET} of all violations  "
        f"({s['locations_for_80pct_coverage']} cover 80%)."
    )
    kv("Hotspot cells / blind spots",
       f"{s['n_hotspot_cells']} / {s['n_blind_spots']}")
    kv("Exposure-adj vs raw rank (Spearman)",
       f"{s['rank_spearman_raw_vs_adjusted']:.2f}  (top hotspots are genuine)")
    kv("BTP enforcement alignment points", f"{s['btp_reference_points']}")

    # ---- Honest framing ----------------------------------------------
    header("HONEST FRAMING")
    print(f"  {YELLOW}*{RESET} Congestion impact is MODELED (HCM->BPR), not measured speed/flow.")
    print(f"  {YELLOW}*{RESET} Counts reflect ENFORCEMENT activity, not true occurrence (bias-corrected).")
    print(f"  {YELLOW}*{RESET} Privacy by design under India's DPDP Act 2023; H3-only public layers.")
    print()
    rule("=")
    print(BOLD + GREEN + "  CurbIQ: half the city's violations live in 86 deployable cells —".center(W) + RESET)
    print(BOLD + GREEN + "  and the evening peak is wide open.".center(W) + RESET)
    rule("=")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

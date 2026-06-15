"""Enforcement prioritization.

Fuses the three signals — statistical hotspot intensity (Gi* z), modeled
congestion impact (CIS), and forecasted next-period demand — into an
actionable, **bias-aware** enforcement plan.

Two deliberate design choices from the research brief:

1. **Exposure-adjusted ranking** breaks the patrol -> record -> rank -> patrol
   feedback loop. Raw counts map where police already *go*, not where violations
   *are*; we divide by an exposure proxy and publish the rank-shift so reviewers
   can see the bias correction bite.
2. **Under-enforcement blind spots** are a first-class output: cells with high
   modeled propensity but low observed enforcement (percentile gap > 0.30).
   We never present a low-count zone as "compliant".
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from curbiq import config as C


def exposure(df: pd.DataFrame, res: int = C.H3_RES_PRIMARY) -> pd.Series:
    """Enforcement-exposure proxy E_cell = active-hours + offence-variety + temporal-spread."""
    col = {8: "h3_r8", 9: "h3_r9", 10: "h3_r10"}[res]
    d = df[df["is_counted"]]
    active_hours = d.drop_duplicates([col, "date", "hour"]).groupby(col, observed=True).size()
    offence_variety = d.groupby(col, observed=True)["primary_offence"].nunique()
    temporal_spread = d.groupby(col, observed=True)["date"].nunique()
    E = (active_hours.add(offence_variety, fill_value=0)
         .add(temporal_spread, fill_value=0))
    E.index.name = "h3"
    return E.rename("exposure")


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)


def run_prioritization(df: pd.DataFrame, hot_df: pd.DataFrame, cis_df: pd.DataFrame,
                       forecast_df: pd.DataFrame, res: int = C.H3_RES_PRIMARY) -> dict:
    # --- assemble per-cell signal table ---------------------------------
    t = hot_df[["count", "gi_z", "lat", "lon", "top_offence", "top_vehicle",
                "is_hotspot"]].copy()
    cis = cis_df.set_index("h3")[["cis_score", "junction_proximity",
                                  "extra_delay_pct", "nearest_junction_m"]]
    t = t.join(cis, how="left")

    # map res-8 area forecast down to res-9 cells via H3 parent
    fc = forecast_df.set_index("h3")["predicted_next_day"]
    parent = {c: h3.cell_to_parent(c, C.FORECAST_RES) for c in t.index}
    t["forecast_area"] = [fc.get(parent[c], np.nan) for c in t.index]

    # only rank cells that actually have observed activity
    t = t[t["count"] > 0].copy()
    t["cis_score"] = t["cis_score"].fillna(t["cis_score"].median())
    t["forecast_area"] = t["forecast_area"].fillna(0.0)
    t["junction_proximity"] = t["junction_proximity"].fillna(0.0)

    # --- exposure-adjusted enforcement rate (bias correction) -----------
    E = exposure(df, res=res).reindex(t.index).fillna(0.0)
    alpha = float(np.median(E[E > 0])) if (E > 0).any() else 1.0
    t["exposure"] = E
    t["adjusted_rate"] = t["count"] / (E + alpha)
    rank_raw = t["count"].rank(ascending=False)
    rank_adj = t["adjusted_rate"].rank(ascending=False)
    rho, _ = spearmanr(rank_raw, rank_adj)

    # --- modeled propensity vs observed -> under-enforcement gap --------
    gi_pos = t["gi_z"].clip(lower=0)
    propensity = (0.40 * _z(t["cis_score"]) + 0.30 * _z(gi_pos)
                  + 0.30 * _z(t["forecast_area"]))
    t["propensity"] = propensity
    t["propensity_pctile"] = propensity.rank(pct=True)
    t["observed_pctile"] = t["count"].rank(pct=True)
    t["under_enforcement_gap"] = t["propensity_pctile"] - t["observed_pctile"]
    t["is_blind_spot"] = t["under_enforcement_gap"] > C.UNDER_ENFORCEMENT_GAP_FLAG

    # --- final priority score (explainable, flow-free stand-in) ---------
    # spatial weight = Gi* z * junction criticality (the brief's stand-in for
    # absent flow data), blended with congestion impact and forecast demand.
    spatial_weight = _z(gi_pos * (1.0 + t["junction_proximity"]))
    blend = (0.35 * _z(t["cis_score"]) + 0.30 * spatial_weight
             + 0.20 * _z(t["forecast_area"]) + 0.15 * _z(t["count"]))
    t["priority_score"] = (_minmax(blend) * 100).round(2)
    t = t.sort_values("priority_score", ascending=False)
    t["priority_rank"] = np.arange(1, len(t) + 1)

    # --- coverage-vs-effort curve (cumulative capture vs #locations) ----
    by_pri = t.sort_values("priority_score", ascending=False)
    cum = by_pri["count"].cumsum() / by_pri["count"].sum()
    frac_cells = np.arange(1, len(by_pri) + 1) / len(by_pri)
    step = max(1, len(by_pri) // 100)
    coverage_curve = {
        "frac_locations": frac_cells[::step].round(4).tolist(),
        "frac_violations_captured": cum.to_numpy()[::step].round(4).tolist(),
    }
    # locations needed to cover 50% / 80% of violations
    cov50 = int((cum < 0.50).sum() + 1)
    cov80 = int((cum < 0.80).sum() + 1)

    blind = t[t["is_blind_spot"]].sort_values("propensity", ascending=False)

    summary = {
        "resolution": res,
        "n_ranked_cells": int(len(t)),
        "n_hotspot_cells": int(t["is_hotspot"].sum()),
        "n_blind_spots": int(t["is_blind_spot"].sum()),
        "exposure_alpha": alpha,
        "rank_spearman_raw_vs_adjusted": float(rho),
        "locations_for_50pct_coverage": cov50,
        "locations_for_80pct_coverage": cov80,
        "btp_reference_points": C.BTP_ENFORCEMENT_GEOGRAPHY["high_density_points"],
    }
    return {
        "table": t.reset_index(),
        "blind_spots": blind.reset_index(),
        "coverage_curve": coverage_curve,
        "summary": summary,
    }


if __name__ == "__main__":
    from curbiq.etl import load_processed
    from curbiq.hotspots import compute_hotspots
    from curbiq.congestion import compute_congestion
    from curbiq.forecast import run_forecast

    df = load_processed()
    hot, _ = compute_hotspots(df)
    cis, _ = compute_congestion(df)
    fc = run_forecast(df)["forecast"]
    r = run_prioritization(df, hot, cis, fc)
    print("== prioritization summary ==")
    for k, v in r["summary"].items():
        print(f"  {k}: {v}")
    print("\n== top-8 priority cells ==")
    print(r["table"].head(8)[["h3", "priority_score", "count", "gi_z", "cis_score",
                              "forecast_area", "top_offence"]].round(2).to_string(index=False))
    print(f"\n== {len(r['blind_spots'])} under-enforcement blind spots; top 5 ==")
    print(r["blind_spots"].head(5)[["h3", "under_enforcement_gap", "propensity",
                                    "count", "cis_score"]].round(3).to_string(index=False))

"""Congestion Impact Score (CIS) — a modeled, explainable estimate.

There is **no speed or flow data** in the dataset, so congestion impact is
*modeled, not measured*. The pipeline is fully transparent and citable:

    maneuver intensity  --HCM Ch.16-->  capacity loss
    capacity loss       --BPR 1964-->   delay multiplier (M_parked / M_base)
    4 z-scored factors  --weighted-->   composite CIS (0-100)

Every number traces to HCM 2000, IRC 106-1990 (PCU), or the US BPR (1964)
volume-delay function. The composite weights live in ``config.CIS_WEIGHTS`` and
are surfaced in the UI so reviewers can see exactly how the score is built.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from curbiq import config as C

EARTH_R_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dphi = lat2 - lat1
    dl = lon2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_junction_distance(cell_lat, cell_lon, jlat, jlon) -> np.ndarray:
    """Min distance (m) from each cell centroid to any named junction."""
    if len(jlat) == 0:
        return np.full(len(cell_lat), np.inf)
    # (M, J) broadcast — M cells x J junctions
    d = haversine_m(cell_lat[:, None], cell_lon[:, None], jlat[None, :], jlon[None, :])
    return d.min(axis=1)


# --------------------------------------------------------------------------- #
# Traffic-engineering primitives
# --------------------------------------------------------------------------- #
def hcm_capacity_loss(maneuver_rate_hr: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """HCM 2000 Ch.16 on-street-parking capacity loss = 1 - fp.

    fp = (N - 0.1 - block_s * Nm/3600) / N,  Nm capped, fp floored, loss capped.
    """
    nm = np.clip(maneuver_rate_hr, 0, C.HCM_MANEUVER_RATE_CAP)
    n = np.maximum(lanes.astype(float), 1.0)
    fp = (n - 0.1 - C.HCM_MANEUVER_BLOCK_S * nm / 3600.0) / n
    fp = np.clip(fp, C.HCM_FP_FLOOR, 1.0)
    return np.clip(1.0 - fp, 0.0, C.MAX_CAPACITY_LOSS)


def bpr_delay_ratio(capacity_loss: np.ndarray, vc0: float = C.BASELINE_VC_RATIO) -> np.ndarray:
    """Delay multiplier M_parked / M_base from the BPR volume-delay function."""
    vc_park = np.clip(vc0 / np.maximum(1.0 - capacity_loss, 1e-6), 0.0, C.VC_CLAMP)
    base = 1.0 + C.BPR_ALPHA * vc0 ** C.BPR_BETA
    park = 1.0 + C.BPR_ALPHA * vc_park ** C.BPR_BETA
    return park / base


def _zscore(v: pd.Series) -> pd.Series:
    sd = v.std(ddof=0)
    return (v - v.mean()) / sd if sd > 0 else pd.Series(0.0, index=v.index)


def _minmax_0_100(v: pd.Series) -> pd.Series:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) * 100.0 if hi > lo else pd.Series(50.0, index=v.index)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def compute_congestion(df: pd.DataFrame, res: int = C.H3_RES_PRIMARY) -> tuple[pd.DataFrame, dict]:
    col = {8: "h3_r8", 9: "h3_r9", 10: "h3_r10"}[res]
    d = df[df["is_counted"]].copy()

    # road-class capacity loss + assumed lanes per event, then cell means/mode
    d["rc_loss"] = d["road_class"].map(C.ROAD_CLASS_CAPACITY_LOSS).fillna(C.DEFAULT_ROAD_CAPACITY_LOSS)
    d["lanes"] = d["road_class"].map(C.ASSUMED_LANES).fillna(C.DEFAULT_LANES)

    agg = d.groupby(col, observed=True).agg(
        count=("id", "size"),
        weighted_count=("confidence", "sum"),
        blockage_sum=("blockage", "sum"),
        peak_share=("peak_overlap", "mean"),
        rc_loss=("rc_loss", "mean"),
        lanes=("lanes", "median"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
    )
    agg.index.name = "h3"

    # Maneuver rate = detected events per *active* peak-hour (conditional
    # intensity), not averaged over the whole window. Optionally scaled by an
    # enforcement-capture multiplier (default 1.0 = use detections as-is).
    m0, m1 = C.MORNING_PEAK
    e0, e1 = C.EVENING_PEAK
    peak_mask = d["hour"].between(m0, m1 - 1) | d["hour"].between(e0, e1 - 1)
    pk = d.loc[peak_mask, [col, "date", "hour"]]
    peak_ev = pk.groupby(col, observed=True).size()
    active_ph = pk.drop_duplicates([col, "date", "hour"]).groupby(col, observed=True).size()
    nm = (peak_ev / active_ph).reindex(agg.index).fillna(0.0) * C.ENFORCEMENT_CAPTURE_MULT
    agg["active_peak_hours"] = active_ph.reindex(agg.index).fillna(0).astype(int)
    agg["maneuver_rate_hr"] = nm

    # nearest named junction -> proximity decay
    jc = (d[d["has_junction"]].groupby("junction_id", observed=True)[["latitude", "longitude"]]
          .mean())
    dist = nearest_junction_distance(
        agg["lat"].to_numpy(), agg["lon"].to_numpy(),
        jc["latitude"].to_numpy(), jc["longitude"].to_numpy())
    agg["nearest_junction_m"] = dist
    agg["junction_proximity"] = np.exp(-dist / C.JUNCTION_DECAY_D0_M)

    # HCM capacity loss + BPR delay multiplier (the explainable headline)
    agg["capacity_loss"] = hcm_capacity_loss(
        agg["maneuver_rate_hr"].to_numpy(), agg["lanes"].to_numpy())
    agg["bpr_delay_ratio"] = bpr_delay_ratio(agg["capacity_loss"].to_numpy())
    agg["extra_delay_pct"] = (agg["bpr_delay_ratio"] - 1.0) * 100.0

    # composite CIS: z-standardize the 4 factors, weight, min-max to 0-100
    z_density = _zscore(agg["blockage_sum"])
    z_junction = _zscore(agg["junction_proximity"])
    z_peak = _zscore(agg["peak_share"])
    z_road = _zscore(agg["rc_loss"])
    w = C.CIS_WEIGHTS
    composite = (w["severity_density"] * z_density
                 + w["junction_proximity"] * z_junction
                 + w["peak_overlap"] * z_peak
                 + w["road_capacity_loss"] * z_road)
    agg["cis_z"] = composite
    agg["cis_score"] = _minmax_0_100(composite)
    agg["z_density"] = z_density
    agg["z_junction"] = z_junction
    agg["z_peak"] = z_peak
    agg["z_road"] = z_road

    # city-level modeled delay-impact index (unitless, "modeled" — never "measured")
    delay_index = float(((agg["bpr_delay_ratio"] - 1.0) * agg["count"]).sum())
    summary = {
        "resolution": res,
        "n_cells": int(len(agg)),
        "modeled": True,
        "city_delay_impact_index": delay_index,
        "mean_extra_delay_pct": float(agg["extra_delay_pct"].mean()),
        "p95_extra_delay_pct": float(agg["extra_delay_pct"].quantile(0.95)),
        "max_extra_delay_pct": float(agg["extra_delay_pct"].max()),
        "weights": dict(w),
        "params": {
            "bpr_alpha": C.BPR_ALPHA, "bpr_beta": C.BPR_BETA,
            "baseline_vc": C.BASELINE_VC_RATIO, "junction_decay_m": C.JUNCTION_DECAY_D0_M,
        },
    }
    return agg.reset_index(), summary


if __name__ == "__main__":
    from curbiq.etl import load_processed

    df = load_processed()
    cis, summary = compute_congestion(df)
    print("== congestion summary ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    top = cis.sort_values("cis_score", ascending=False).head(8)
    print("\n== top-8 cells by CIS ==")
    print(top[["h3", "count", "cis_score", "extra_delay_pct", "capacity_loss",
               "nearest_junction_m", "maneuver_rate_hr", "peak_share"]].round(2).to_string(index=False))

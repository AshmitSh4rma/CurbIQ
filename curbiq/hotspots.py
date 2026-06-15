"""Hotspot engine.

Turns geocoded violations into statistically significant hotspots:

* ``compute_hotspots`` — H3-cell Gi*/Moran/LISA with Benjamini-Hochberg FDR.
* ``hotspot_zones``    — merges contiguous hot cells into enforcement zones.
* ``junction_hotspots``— ranks the dataset's named junctions.
* ``emerging_hotspots``— Mann-Kendall trend typing on weekly per-cell counts.

The analysis variable is the **confidence-weighted** violation count per cell
(rejected/duplicate tickets are down-weighted) — addressing validation-status
inflation while keeping the Gi* count semantics.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from curbiq import config as C
from curbiq import spatial as S

_RES_COL = {8: "h3_r8", 9: "h3_r9", 10: "h3_r10"}


def _res_col(res: int) -> str:
    if res not in _RES_COL:
        raise ValueError(f"unsupported resolution {res}; use one of {list(_RES_COL)}")
    return _RES_COL[res]


def _mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iat[0] if not m.empty else None


def aggregate_cells(df: pd.DataFrame, res: int = C.H3_RES_PRIMARY) -> pd.DataFrame:
    """Per-cell aggregates over counted records."""
    col = _res_col(res)
    d = df[df["is_counted"]]
    agg = d.groupby(col, observed=True).agg(
        count=("id", "size"),
        weighted_count=("confidence", "sum"),
        blockage_sum=("blockage", "sum"),
        mean_severity=("carriageway_severity", "mean"),
        mean_footprint=("footprint_pcu", "mean"),
        peak_share=("peak_overlap", "mean"),
        junction_share=("has_junction", "mean"),
        road_weight=("road_weight", "mean"),
        top_offence=("primary_offence", _mode),
        top_vehicle=("vehicle_type", _mode),
        n_stations=("police_station", "nunique"),
    )
    agg.index.name = "h3"
    return agg


# --------------------------------------------------------------------------- #
# Core: Gi* / Moran / LISA on the H3 grid
# --------------------------------------------------------------------------- #
def compute_hotspots(df: pd.DataFrame, res: int = C.H3_RES_PRIMARY,
                     k: int = C.GI_NEIGHBOR_K) -> tuple[pd.DataFrame, dict]:
    agg = aggregate_cells(df, res=res)
    cell_values = agg["weighted_count"].to_dict()

    cells, x, idx = S.build_active_lattice(cell_values, k=k)
    W_star = S.build_weights(cells, idx, k=k, include_self=True)
    W_rs = S.build_weights(cells, idx, k=k, include_self=False, row_standardize=True)

    gi_z = S.getis_ord_gi_star(x, W_star)
    gi_p = S.z_to_p_two_sided(gi_z)
    sig, pcrit = S.benjamini_hochberg(gi_p, C.FDR_Q)
    moran = S.global_morans_i(x, W_rs)
    lisa = S.local_morans(x, W_rs)
    n_nbrs = np.asarray(W_star.sum(axis=1)).ravel() - 1.0   # exclude self

    latlng = [h3.cell_to_latlng(c) for c in cells]
    out = pd.DataFrame({
        "h3": cells,
        "lat": [p[0] for p in latlng],
        "lon": [p[1] for p in latlng],
        "x": x,
        "gi_z": gi_z,
        "gi_p": gi_p,
        "gi_sig": sig,
        "lisa_I": lisa["I"],
        "lisa_quadrant": lisa["quadrant"],
        "lisa_p": lisa["p"],
        "n_neighbors": n_nbrs.astype(int),
    }).set_index("h3")

    out = out.join(agg, how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    out["weighted_count"] = out["weighted_count"].fillna(0.0)

    out["is_hotspot"] = out["gi_sig"] & (out["gi_z"] >= C.GI_DISPLAY_Z)
    out["is_coldspot"] = out["gi_sig"] & (out["gi_z"] <= -C.GI_DISPLAY_Z)
    out["gi_band"] = [S.gi_confidence_band(z, bool(s))
                      for z, s in zip(out["gi_z"], out["gi_sig"])]
    out["lisa_sig"] = out["lisa_p"] <= 0.05

    stats = {
        "resolution": res,
        "k": k,
        "n_active_cells": int(len(cells)),
        "n_populated_cells": int((out["count"] > 0).sum()),
        "n_hotspots": int(out["is_hotspot"].sum()),
        "n_coldspots": int(out["is_coldspot"].sum()),
        "n_isolates": int((n_nbrs == 0).sum()),
        "mean_neighbors": float(n_nbrs[n_nbrs > 0].mean()) if (n_nbrs > 0).any() else 0.0,
        "fdr_critical_p": float(pcrit),
        "global_moran": {
            "I": moran.I, "expected": moran.expected, "z": moran.z, "p": moran.p,
        },
        "total_violations": int(out["count"].sum()),
    }
    return out, stats


# --------------------------------------------------------------------------- #
# Merge contiguous hot cells into enforcement zones
# --------------------------------------------------------------------------- #
def hotspot_zones(hot_df: pd.DataFrame, max_zones: int | None = None) -> pd.DataFrame:
    """Connected components of significant hot cells via H3 adjacency."""
    hot = set(hot_df.index[hot_df["is_hotspot"]])
    seen: set[str] = set()
    rows = []
    for start in hot:
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            for nb in h3.grid_disk(cur, 1):
                if nb in hot and nb not in seen:
                    stack.append(nb)
        sub = hot_df.loc[comp]
        rows.append({
            "n_cells": len(comp),
            "count": int(sub["count"].sum()),
            "weighted_count": float(sub["weighted_count"].sum()),
            "peak_gi_z": float(sub["gi_z"].max()),
            "lat": float(sub["lat"].mean()),
            "lon": float(sub["lon"].mean()),
            "top_offence": _mode(sub["top_offence"]),
            "top_vehicle": _mode(sub["top_vehicle"]),
            "mean_peak_share": float(sub["peak_share"].mean()),
            "cells": comp,
        })
    zones = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    zones.insert(0, "zone_id", [f"Z{i:03d}" for i in range(len(zones))])
    if max_zones:
        zones = zones.head(max_zones)
    return zones


# --------------------------------------------------------------------------- #
# Junction-level hotspots (named junctions are first-class for BTP)
# --------------------------------------------------------------------------- #
def junction_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["is_counted"] & df["has_junction"]]
    g = d.groupby("junction_id", observed=True).agg(
        count=("id", "size"),
        weighted_count=("confidence", "sum"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        mean_severity=("carriageway_severity", "mean"),
        peak_share=("peak_overlap", "mean"),
        top_offence=("primary_offence", _mode),
        top_vehicle=("vehicle_type", _mode),
        n_stations=("police_station", "nunique"),
    ).sort_values("count", ascending=False)
    g["rank"] = np.arange(1, len(g) + 1)
    g["count_pctile"] = g["count"].rank(pct=True)
    return g.reset_index()


# --------------------------------------------------------------------------- #
# Emerging hotspots: Mann-Kendall trend typing on weekly per-cell counts
# --------------------------------------------------------------------------- #
def emerging_hotspots(df: pd.DataFrame, res: int = C.H3_RES_PRIMARY,
                      min_total: int = 20) -> pd.DataFrame:
    col = _res_col(res)
    d = df[df["is_counted"]]
    wk = d.groupby([col, "week"], observed=True).size().unstack(fill_value=0).sort_index(axis=1)
    totals = wk.sum(axis=1)
    wk = wk[totals >= min_total]
    if wk.empty:
        return pd.DataFrame(columns=["h3", "category", "trend", "mk_z", "mk_p", "tau",
                                     "total", "recent4", "early4", "active_week_frac"])
    n_weeks = wk.shape[1]
    half = max(1, n_weeks // 4)
    rows = []
    for cell, series in wk.iterrows():
        s = series.to_numpy(dtype=float)
        mk = S.mann_kendall(s)
        recent = float(s[-half:].sum())
        early = float(s[:half].sum())
        active_frac = float((s > 0).mean())
        cv = float(s.std() / s.mean()) if s.mean() > 0 else 0.0
        if early == 0 and recent > 0:
            category = "new"
        elif mk["trend"] == "intensifying":
            category = "intensifying"
        elif mk["trend"] == "diminishing":
            category = "diminishing"
        elif active_frac >= 0.8:
            category = "persistent"
        elif active_frac <= 0.35:
            category = "sporadic"
        elif cv > 1.0:
            category = "oscillating"
        else:
            category = "stable"
        latlng = h3.cell_to_latlng(cell)
        rows.append({
            "h3": cell, "lat": latlng[0], "lon": latlng[1],
            "category": category, "trend": mk["trend"],
            "mk_z": mk["z"], "mk_p": mk["p"], "tau": mk["tau"],
            "total": int(s.sum()), "recent4": recent, "early4": early,
            "active_week_frac": active_frac,
        })
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from curbiq.etl import load_processed

    df = load_processed()
    hot, stats = compute_hotspots(df)
    print("== hotspot stats ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    zones = hotspot_zones(hot)
    print(f"\n== {len(zones)} hotspot zones; top 5 ==")
    print(zones.head(5)[["zone_id", "n_cells", "count", "peak_gi_z", "top_offence", "lat", "lon"]])
    jh = junction_hotspots(df)
    print(f"\n== top 5 junctions ({len(jh)} total) ==")
    print(jh.head(5)[["junction_id", "count", "top_offence", "peak_share"]])
    em = emerging_hotspots(df)
    print(f"\n== emerging categories ==\n{em['category'].value_counts().to_dict()}")

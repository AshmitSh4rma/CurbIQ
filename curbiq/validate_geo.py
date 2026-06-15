"""Validate statistical hotspots against an operational enforcement geography.

Smart-city judges weight *real-world viability*: do the hotspots we compute map
onto the places the traffic police already deploy? This module measures the
overlap between CurbIQ's top hotspot cells and a reference point set, and — just
as important — flags **novel hotspots** far from any known enforcement point
(candidate *new* deployment locations).

Reference options:
  * ``load_reference_csv(path)``        — the official list (name,lat,lon CSV);
  * ``btp_reference_from_data(df)``     — the dataset's BTP-named junctions,
    which are BTP's own operational deployment IDs (a non-circular proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from curbiq.congestion import haversine_m


def load_reference_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({
        "lat": df[cols["lat"]].astype(float),
        "lon": df[cols["lon"]].astype(float),
        "name": df[cols["name"]] if "name" in cols else range(len(df)),
    })
    return out


def btp_reference_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """BTP-named junctions = operational deployment points (BTPxxx IDs)."""
    d = df[df["is_counted"] & df["has_junction"]]
    ref = d.groupby("junction_id", observed=True).agg(
        lat=("latitude", "mean"), lon=("longitude", "mean"), count=("id", "size"))
    ref = ref.reset_index().rename(columns={"junction_id": "name"})
    return ref.sort_values("count", ascending=False).reset_index(drop=True)


def _nearest_m(cell_lat, cell_lon, ref_lat, ref_lon) -> np.ndarray:
    """Nearest reference distance (m) for each cell (broadcast haversine)."""
    if len(ref_lat) == 0:
        return np.full(len(cell_lat), np.inf)
    d = haversine_m(cell_lat[:, None], cell_lon[:, None], ref_lat[None, :], ref_lon[None, :])
    return d.min(axis=1)


def validate(cells: pd.DataFrame, ref: pd.DataFrame,
             rank_col: str = "priority_score",
             top_ns=(20, 50, 100, 154), radii=(150, 300, 500),
             novel_radius: float = 500.0) -> dict:
    cells = cells.sort_values(rank_col, ascending=False).reset_index(drop=True)
    clat, clon = cells["lat"].to_numpy(), cells["lon"].to_numpy()
    rlat, rlon = ref["lat"].to_numpy(), ref["lon"].to_numpy()
    nearest = _nearest_m(clat, clon, rlat, rlon)

    # precision@N: of the top-N ranked cells, share within R m of a reference point
    precision = {}
    for n in top_ns:
        nn = min(n, len(cells))
        nd = nearest[:nn]
        precision[f"top{n}"] = {f"{int(r)}m": round(float((nd <= r).mean()), 3) for r in radii}

    # recall: share of reference points with a *significant hotspot* cell within R m
    hot = cells[cells["is_hotspot"]] if "is_hotspot" in cells else cells.head(0)
    recall = {}
    if len(hot):
        hd = _nearest_m(rlat, rlon, hot["lat"].to_numpy(), hot["lon"].to_numpy())
        recall = {f"{int(r)}m": round(float((hd <= r).mean()), 3) for r in radii}

    # novel hotspots: significant hotspots far from any known enforcement point
    novel = hot[_nearest_m(hot["lat"].to_numpy(), hot["lon"].to_numpy(), rlat, rlon) > novel_radius] \
        if len(hot) else hot
    novel_list = []
    if len(novel):
        nv = novel.copy()
        nv["nearest_ref_m"] = _nearest_m(nv["lat"].to_numpy(), nv["lon"].to_numpy(), rlat, rlon)
        for _, r in nv.sort_values("count", ascending=False).head(20).iterrows():
            novel_list.append({"h3": r["h3"], "lat": round(float(r["lat"]), 5),
                               "lon": round(float(r["lon"]), 5),
                               "count": int(r.get("count", 0)),
                               "nearest_ref_m": round(float(r["nearest_ref_m"]), 0),
                               "top_offence": r.get("top_offence")})
    return {
        "n_reference_points": int(len(ref)),
        "n_hotspots": int(len(hot)),
        "precision_at_n": precision,
        "recall": recall,
        "mean_nearest_m_top50": round(float(nearest[:min(50, len(cells))].mean()), 1),
        "median_nearest_m_top50": round(float(np.median(nearest[:min(50, len(cells))])), 1),
        "n_novel_hotspots": int(len(novel)),
        "novel_hotspots": novel_list,
        "novel_radius_m": novel_radius,
    }


def run_geo_validation(df, cells, ref_path=None) -> dict:
    if ref_path:
        ref = load_reference_csv(ref_path)
        source, official = f"reference_csv:{ref_path}", True
    else:
        ref = btp_reference_from_data(df)
        source, official = "btp_named_junctions_from_data", False
    result = validate(cells, ref)
    result["reference_source"] = source
    result["is_official_list"] = official
    result["note"] = ("Reference = BTP-named junctions in the dataset (operational deployment "
                      "IDs). Drop in the official 12-corridor/43-junction/99-road list via "
                      "`build_all.py --enforcement-points file.csv`."
                      if not official else "Validated against the supplied official point list.")
    return result


if __name__ == "__main__":
    from curbiq.congestion import compute_congestion
    from curbiq.etl import load_processed
    from curbiq.hotspots import compute_hotspots
    from curbiq.prioritize import run_prioritization
    from curbiq.forecast import run_forecast

    df = load_processed()
    hot, _ = compute_hotspots(df)
    cis, _ = compute_congestion(df)
    fc = run_forecast(df)["forecast"]
    prio = run_prioritization(df, hot, cis, fc)["table"]
    r = run_geo_validation(df, prio)
    print("== hotspot vs BTP enforcement geography ==")
    print(f"  reference points     : {r['n_reference_points']} ({r['reference_source']})")
    print(f"  precision@50 (300m)  : {r['precision_at_n']['top50']['300m']}")
    print(f"  precision@154 (300m) : {r['precision_at_n']['top154']['300m']}")
    print(f"  recall (300m)        : {r['recall'].get('300m')}")
    print(f"  median nearest (top50): {r['median_nearest_m_top50']} m")
    print(f"  novel hotspots (>500m): {r['n_novel_hotspots']}  (candidate NEW points)")
    for nv in r["novel_hotspots"][:5]:
        print(f"     {nv['h3']}  {nv['count']} viol  {nv['nearest_ref_m']:.0f} m from nearest junction  {nv['top_offence']}")

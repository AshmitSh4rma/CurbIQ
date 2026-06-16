"""Offline artifact builder — the precompute-then-serve contract.

Runs every analytical module once and emits compact, versioned, **privacy-safe**
JSON (+ the trained model) under ``data/artifacts/``. The FastAPI layer and the
static dashboard only ever read these files — no spatial math at request time.

Privacy by design (India DPDP Act 2023):
  * public layers carry **H3 cell ids + centroids only**, never raw point lat/lon;
  * any cell with count < ``K_ANON_MIN`` is suppressed (k-anonymity);
  * vehicle plates never reach the artifact layer.
"""
from __future__ import annotations

import datetime as _dt

import h3
import orjson
import pandas as pd

from curbiq import config as C
from curbiq.calibration import run_calibration
from curbiq.congestion import compute_congestion
from curbiq.etl import load_processed
from curbiq.fairness import k_anon_suppress, run_fairness
from curbiq.forecast import run_forecast
from curbiq.hotspots import (compute_hotspots, emerging_hotspots,
                             hotspot_zones, junction_hotspots)
from curbiq.patrol import optimize_patrols
from curbiq.emergence import run_emergence
from curbiq.prioritize import run_prioritization
from curbiq.scenario import run_scenario
from curbiq.timing import run_timing
from curbiq.validate_geo import run_geo_validation


def _write(name: str, obj, outdir) -> dict:
    path = outdir / name
    data = orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)
    path.write_bytes(data)
    return {"file": name, "bytes": len(data)}


def _round_records(df: pd.DataFrame, cols: list[str], ndigits=3) -> list[dict]:
    sub = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].round(ndigits)
    return sub.to_dict("records")


# --------------------------------------------------------------------------- #
# Descriptive time-series / breakdowns straight from the processed frame
# --------------------------------------------------------------------------- #
def compute_timeseries(df: pd.DataFrame) -> dict:
    d = df[df["is_counted"]]
    daily = d.groupby("date", observed=True).size()
    weekly = d.groupby("week", observed=True).size()
    hourly = d.groupby("hour", observed=True).size().reindex(range(24), fill_value=0)
    dow = d.groupby("dow", observed=True).size().reindex(range(7), fill_value=0)
    vehicle = d["vehicle_category"].value_counts()
    vehicle_type = d["vehicle_type"].value_counts().head(12)
    offence = d["primary_offence"].value_counts().head(12)
    road = d["road_class"].value_counts()
    stations = d["police_station"].value_counts().head(15)
    tod = d["tod_bucket"].value_counts()
    return {
        "daily": {"date": daily.index.tolist(), "count": daily.tolist()},
        "weekly": {"week": weekly.index.tolist(), "count": weekly.tolist()},
        "hourly_ist": {"hour": list(range(24)), "count": hourly.tolist()},
        "day_of_week": {"dow": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        "count": dow.tolist()},
        "vehicle_category": vehicle.to_dict(),
        "top_vehicle_types": vehicle_type.to_dict(),
        "top_offences": offence.to_dict(),
        "road_class": road.to_dict(),
        "top_stations": stations.to_dict(),
        "time_of_day": tod.to_dict(),
    }


def compute_weekly(df: pd.DataFrame, top_n: int = 300) -> dict:
    """Per-week violation counts for the top cells (drives the time-slider / 3D layer)."""
    d = df[df["is_counted"]]
    weeks = sorted(w for w in d["week"].dropna().unique())
    top_cells = d["h3_r9"].value_counts().head(top_n).index
    sub = d[d["h3_r9"].isin(top_cells)]
    pivot = (sub.groupby(["h3_r9", "week"], observed=True).size()
             .unstack(fill_value=0).reindex(columns=weeks, fill_value=0))
    cells = []
    for cell, row in pivot.iterrows():
        lat, lon = h3.cell_to_latlng(cell)
        cells.append({"h3": cell, "lat": round(lat, 5), "lon": round(lon, 5),
                      "counts": [int(v) for v in row.to_numpy()]})
    return {"weeks": list(weeks),
            "max_count": int(pivot.to_numpy().max()) if len(pivot) else 0,
            "cells": cells}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_artifacts(df: pd.DataFrame | None = None, outdir=C.ARTIFACTS_DIR,
                    verbose=True, probe_path: str | None = None,
                    enforcement_points: str | None = None) -> dict:
    def log(m):
        if verbose:
            print(f"[artifacts] {m}")

    if df is None:
        df = load_processed()
    outdir.mkdir(parents=True, exist_ok=True)

    log("hotspots (Gi*/Moran/LISA, res 9) ...")
    hot, hot_stats = compute_hotspots(df)
    log("congestion impact (HCM/BPR/CIS) ...")
    cis, cis_summary = compute_congestion(df)
    log("forecasting (LightGBM Poisson, res 8) ...")
    fc = run_forecast(df)
    log("prioritization (exposure-adjusted + blind spots) ...")
    prio = run_prioritization(df, hot, cis, fc["forecast"])
    log("geo-validation vs BTP enforcement geography ...")
    geo = run_geo_validation(df, prio["table"], ref_path=enforcement_points)
    log(f"  precision@154 (300m)={geo['precision_at_n']['top154']['300m']}, "
        f"{geo['n_novel_hotspots']} novel off-junction hotspots")
    log("patrol routing optimization (VRP) ...")
    patrol = optimize_patrols(prio["table"])
    log(f"  {patrol['solver']}: {patrol['n_units']} units cover "
        f"{patrol['stops_covered']}/{patrol['candidate_stops']} stops, {patrol['total_distance_km']} km")
    log("zones / junctions / emerging ...")
    zones = hotspot_zones(hot)
    junctions = junction_hotspots(df)
    emerging = emerging_hotspots(df)
    log("fairness / privacy ...")
    fair = run_fairness(df)
    log("congestion calibration (probe-speed validation) ...")
    calib = run_calibration(df, probe_path=probe_path, synthetic=probe_path is None)
    log(f"  CIS vs probe: Spearman {calib['spearman_default']} -> {calib['spearman_calibrated']} "
        f"(calibrated){' [synthetic probe]' if calib['synthetic'] else ''}")
    log("time series + weekly evolution ...")
    ts = compute_timeseries(df)
    weekly = compute_weekly(df)
    log("emergence risk (forward hotspot-formation model) ...")
    emg = run_emergence(df)
    log(f"  {emg['summary']['n_predicted_emerging']} cells predicted emerging; "
        f"AUC {emg['summary']['model_auc']:.3f} vs baseline {emg['summary']['baseline_auc']:.3f}")
    log("enforcement timing windows ...")
    timing = run_timing(df, hot)
    log("congestion ROI / what-if delay recovery ...")
    scen = run_scenario(df, cis)
    log(f"  recoverable-delay index {scen['summary']['city_recoverable_delay_index']:.1f}; "
        f"{scen['summary']['cells_for_50pct']} cells recover 50%")

    # --- rich per-cell map layer (res 9), merged + k-anonymized ----------
    cells = prio["table"].set_index("h3")
    cells = cells.join(hot[["gi_band", "lisa_quadrant", "lisa_p", "weighted_count"]],
                       how="left")
    # extra_delay_pct already arrives via the prioritization table; add the rest
    cells = cells.join(cis.set_index("h3")[["capacity_loss", "z_density",
                                            "z_junction", "z_peak", "z_road"]],
                       how="left")
    # forward-looking + ROI + timing layers (joined for map coloring of visible cells)
    cells = cells.join(emg["cells"].set_index("h3")[
        ["emergence_risk", "risk_band", "predicted_emerging"]], how="left")
    cells = cells.join(scen["cells"].set_index("h3")[
        ["recoverable_delay", "recoverable_pct"]], how="left")
    cells = cells.join(timing["cells"].set_index("h3")[
        ["peak_hour", "window_start", "window_end"]], how="left")
    # per-cell offences present (primary_offence with >= K_ANON_MIN records; privacy-safe) so the
    # dashboard can filter cells CONTAINING an offence, not just the single dominant one.
    _oc = (df[df["is_counted"]].dropna(subset=["primary_offence"])
           .groupby(["h3_r9", "primary_offence"], observed=True).size())
    _oc = _oc[_oc >= C.K_ANON_MIN].reset_index()
    _cell_off = _oc.groupby("h3_r9", observed=True)["primary_offence"].apply(list).to_dict()
    cells["offences"] = [_cell_off.get(h, []) for h in cells.index]
    # don't double-label an established Gi* hotspot as "emerging" (distinct definitions)
    cells["predicted_emerging"] = (cells["predicted_emerging"].fillna(False)
                                   & ~cells["is_hotspot"].fillna(False))
    # cell -> zone id
    cell_zone = {}
    for _, row in zones.iterrows():
        for c in row["cells"]:
            cell_zone[c] = row["zone_id"]
    cells["zone_id"] = [cell_zone.get(c) for c in cells.index]
    cells = cells.reset_index()
    cells_pub, kanon = k_anon_suppress(cells, "count", C.K_ANON_MIN)
    log(f"k-anon: suppressed {kanon['n_suppressed']}/{kanon['n_total']} "
        f"({kanon['frac_suppressed']:.1%}) cells with count < {C.K_ANON_MIN}")

    cell_cols = ["h3", "lat", "lon", "count", "gi_z", "gi_band", "is_hotspot",
                 "lisa_quadrant", "cis_score", "extra_delay_pct", "capacity_loss",
                 "priority_score", "priority_rank", "forecast_area",
                 "under_enforcement_gap", "is_blind_spot", "zone_id",
                 "top_offence", "top_vehicle",
                 "emergence_risk", "risk_band", "predicted_emerging",
                 "recoverable_delay", "recoverable_pct",
                 "peak_hour", "window_start", "window_end", "offences"]
    cells_records = _round_records(cells_pub, [c for c in cell_cols if c in cells_pub], 3)

    # --- forecast layer (res 8), k-anon on predicted -> keep top area cells
    fcast = fc["forecast"].copy()
    fcast = fcast[fcast["predicted_next_day"] >= 1.0]
    forecast_records = _round_records(fcast, ["h3", "lat", "lon", "predicted_next_day"], 3)

    # --- KPIs (headline) -------------------------------------------------
    span = (df["created_ist"].min(), df["created_ist"].max())
    kpis = {
        "total_violations": int(df["is_counted"].sum()),
        "date_range": [str(span[0].date()), str(span[1].date())],
        "n_police_stations": int(df["police_station"].nunique()),
        "n_junctions": int(df.loc[df["has_junction"], "junction_id"].nunique()),
        "n_h3_cells": hot_stats["n_populated_cells"],
        "n_hotspots": hot_stats["n_hotspots"],
        "global_moran_I": round(hot_stats["global_moran"]["I"], 4),
        "global_moran_z": round(hot_stats["global_moran"]["z"], 1),
        "n_hotspot_zones": int(len(zones)),
        "n_blind_spots": prio["summary"]["n_blind_spots"],
        "locations_for_50pct": prio["summary"]["locations_for_50pct_coverage"],
        "city_delay_impact_index": round(cis_summary["city_delay_impact_index"], 1),
        "evening_peak_enforcement_share": fair["temporal"]["evening_peak_enforcement_share"],
        "forecast_pai_at_5": round(fc["holdout"]["metrics"].get("pai@5", 0), 2),
        "forecast_roc_auc": round(fc["holdout"]["metrics"].get("roc_auc", 0), 3),
        "top_station": max(ts["top_stations"], key=ts["top_stations"].get),
        "top_offence": max(ts["top_offences"], key=ts["top_offences"].get),
        "cis_validation_rho": calib["spearman_calibrated"],
        "cis_validation_synthetic": calib["synthetic"],
        "hotspot_btp_precision_300m": geo["precision_at_n"]["top154"]["300m"],
        "n_novel_hotspots": geo["n_novel_hotspots"],
        "patrol_coverage_pct": patrol["coverage_pct"],
        "patrol_solver": patrol["solver"],
        "n_predicted_emerging": emg["summary"]["n_predicted_emerging"],
        "emergence_model_auc": (round(emg["summary"]["model_auc"], 3)
                                if emg["summary"]["model_auc"] == emg["summary"]["model_auc"]
                                else None),
        "cells_for_50pct_delay": scen["summary"]["cells_for_50pct"],
        "enforcement_peak_hour_ist": timing["summary"]["peak_hour"],
    }

    # --- model metrics (rigor view) -------------------------------------
    model_metrics = {
        "forecast": {
            "cv_mean": fc["cv"]["mean"],
            "holdout": fc["holdout"],
            "baselines": fc["baselines"],
            "feature_importances": dict(list(fc["feature_importances"].items())[:20]),
            "best_iteration": fc["best_iteration"],
            "n_panel_rows": fc["n_panel_rows"],
        },
        "hotspots": hot_stats,
        "congestion": cis_summary,
        "prioritization": prio["summary"],
        "emergence": emg["summary"],
        "scenario": scen["summary"],
        "timing": {k: v for k, v in timing["summary"].items()
                   if k not in ("global_hourly_ist", "weekday_hourly", "weekend_hourly")},
    }

    # --- assemble + write ------------------------------------------------
    version = f"{span[0].strftime('%Y%m')}_{span[1].strftime('%Y%m')}"
    files = []
    files.append(_write("kpis.json", kpis, outdir))
    files.append(_write("cells.json", {"resolution": 9, "k_anon": kanon, "cells": cells_records}, outdir))
    files.append(_write("forecast_cells.json", {"resolution": C.FORECAST_RES, "cells": forecast_records}, outdir))
    files.append(_write("zones.json", _round_records(
        zones, ["zone_id", "n_cells", "count", "peak_gi_z", "lat", "lon",
                "top_offence", "top_vehicle", "mean_peak_share"], 3), outdir))
    files.append(_write("junctions.json", _round_records(
        junctions.head(60), ["junction_id", "count", "lat", "lon", "mean_severity",
                             "peak_share", "top_offence", "rank", "count_pctile"], 3), outdir))
    files.append(_write("emerging.json", {
        "by_category": emerging["category"].value_counts().to_dict(),
        "cells": _round_records(emerging.head(150),
                                ["h3", "lat", "lon", "category", "trend", "mk_z",
                                 "mk_p", "tau", "total", "active_week_frac"], 3),
    }, outdir))
    files.append(_write("priority.json", {
        "summary": prio["summary"],
        "coverage_curve": prio["coverage_curve"],
        "top": _round_records(
            prio["table"][prio["table"]["count"] >= C.K_ANON_MIN].head(60), [
            "h3", "lat", "lon", "priority_score", "priority_rank", "count", "gi_z",
            "cis_score", "forecast_area", "top_offence"], 3),
        "blind_spots": _round_records(
            prio["blind_spots"][prio["blind_spots"]["count"] >= C.K_ANON_MIN].head(60),
            ["h3", "lat", "lon", "under_enforcement_gap", "propensity", "count",
             "cis_score"], 3),
    }, outdir))
    files.append(_write("fairness.json", fair, outdir))
    files.append(_write("calibration.json", calib, outdir))
    files.append(_write("geo_validation.json", geo, outdir))
    files.append(_write("patrol.json", patrol, outdir))
    files.append(_write("timeseries.json", ts, outdir))
    files.append(_write("weekly.json", weekly, outdir))
    files.append(_write("model_metrics.json", model_metrics, outdir))
    # k-anon: the standalone layers must honor the same count<K floor as cells.json
    gi_hot = set(hot.index[hot["is_hotspot"]])
    files.append(_write("emergence.json", {
        "summary": emg["summary"],
        "cells": _round_records(
            emg["cells"][emg["cells"]["predicted_emerging"]
                         & ~emg["cells"]["h3"].isin(gi_hot)].head(150),
            ["h3", "lat", "lon", "emergence_risk", "risk_band"], 4),
    }, outdir))
    files.append(_write("timing.json", {
        "summary": timing["summary"],
        "cells": _round_records(
            timing["cells"][timing["cells"]["n"] >= C.K_ANON_MIN].head(120), [
            "h3", "lat", "lon", "n", "peak_hour", "window_start", "window_end",
            "window_share", "morning_share", "evening_share"], 3),
    }, outdir))
    files.append(_write("scenario.json", {
        "summary": scen["summary"],
        "curve": scen["curve"],
        "top": _round_records(
            scen["cells"][scen["cells"]["count"] >= C.K_ANON_MIN].head(60), [
            "h3", "lat", "lon", "recoverable_delay", "recoverable_pct", "cum_pct",
            "rank", "extra_delay_pct", "count"], 4),
    }, outdir))

    # --- trained model + manifest ---------------------------------------
    model_path = C.MODELS_DIR / "forecast_lgbm.txt"
    fc["model"].booster_.save_model(str(model_path))
    (C.MODELS_DIR / "feature_cols.json").write_bytes(orjson.dumps(fc["feature_cols"]))

    manifest = {
        "name": "CurbIQ",
        "version": version,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "license": C.LICENSE,
        "dataset": {
            "records": int(len(df)),
            "date_range": kpis["date_range"],
            "source": "Bengaluru Traffic Police parking-violation export (anonymized)",
        },
        "config": {
            "h3_primary": C.H3_RES_PRIMARY, "forecast_res": C.FORECAST_RES,
            "gi_k": C.GI_NEIGHBOR_K, "fdr_q": C.FDR_Q,
            "cis_weights": C.CIS_WEIGHTS, "k_anon": C.K_ANON_MIN,
        },
        "files": files,
        "headline_metrics": {
            "global_moran_I": kpis["global_moran_I"],
            "n_hotspots": kpis["n_hotspots"],
            "forecast_pai_at_5": kpis["forecast_pai_at_5"],
            "forecast_roc_auc": kpis["forecast_roc_auc"],
            "evening_peak_enforcement_share": kpis["evening_peak_enforcement_share"],
            "cis_validation_spearman": calib["spearman_calibrated"],
            "hotspot_btp_precision_300m": geo["precision_at_n"]["top154"]["300m"],
            "emergence_model_auc": kpis["emergence_model_auc"],
            "city_recoverable_delay_index": round(scen["summary"]["city_recoverable_delay_index"], 1),
        },
    }
    _write("manifest.json", manifest, outdir)
    log(f"wrote {len(files) + 1} artifacts ({sum(f['bytes'] for f in files) / 1e6:.2f} MB) "
        f"to {outdir}")
    log(f"saved model -> {model_path}")
    return manifest


if __name__ == "__main__":
    m = build_artifacts()
    print("\n== manifest headline ==")
    for k, v in m["headline_metrics"].items():
        print(f"  {k}: {v}")
    print("\n== files ==")
    for f in m["files"]:
        print(f"  {f['file']}: {f['bytes'] / 1000:.1f} KB")

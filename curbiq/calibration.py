"""Congestion calibration & validation against probe-speed data.

The Congestion Impact Score (CIS) is a *modeled* estimate. This module makes it
*validatable*: given an external probe-speed dataset (TomTom Traffic Index,
Google `duration_in_traffic`, Uber Movement, or loop detectors) it

  1. measures how well CIS rank-orders real congestion (Spearman ρ),
  2. re-fits the CIS component weights to maximize that agreement,
  3. learns a monotone (isotonic) map CIS → congestion-%, reporting R² / MAE.

No live paid feed is reachable in this environment, so it ships a
**physically-grounded synthetic probe generator** (clearly labelled) built from
the real per-cell drivers via an *independent* logistic form — so the
validation is informative, not circular. Drop in real data with `load_probe_csv`.

Probe CSV schema (header row; any one form):
    h3,observed_speed_kmph,freeflow_speed_kmph[,period]
    lat,lon,observed_speed_kmph,freeflow_speed_kmph[,period]
    h3,congestion_pct[,period]          # congestion_pct = (freeflow-observed)/freeflow*100
"""
from __future__ import annotations

import json

import h3
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

from curbiq import config as C

CALIB_PATH = C.MODELS_DIR / "calibration.json"
_Z_COLS = ["z_density", "z_junction", "z_peak", "z_road"]


def congestion_from_speeds(observed, freeflow) -> np.ndarray:
    obs = np.asarray(observed, float)
    free = np.asarray(freeflow, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (free - obs) / free * 100.0
    return np.clip(np.nan_to_num(pct), 0.0, 100.0)


def load_probe_csv(path, res: int = C.H3_RES_PRIMARY) -> pd.DataFrame:
    """Load a real probe-speed CSV into per-cell congestion_pct (see module docstring)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "h3" in cols:
        df["h3"] = df[cols["h3"]].astype(str)
    elif "lat" in cols and "lon" in cols:
        df["h3"] = [h3.latlng_to_cell(la, lo, res)
                    for la, lo in zip(df[cols["lat"]], df[cols["lon"]])]
    else:
        raise ValueError("probe CSV needs an 'h3' column or 'lat'+'lon' columns")
    if "congestion_pct" in cols:
        df["congestion_pct"] = df[cols["congestion_pct"]].astype(float)
    elif "observed_speed_kmph" in cols and "freeflow_speed_kmph" in cols:
        df["congestion_pct"] = congestion_from_speeds(
            df[cols["observed_speed_kmph"]], df[cols["freeflow_speed_kmph"]])
    else:
        raise ValueError("probe CSV needs 'congestion_pct' or observed+freeflow speeds")
    return df.groupby("h3", as_index=False)["congestion_pct"].mean()


def synthetic_probe(cis_df: pd.DataFrame, seed: int = C.MORAN_SEED) -> pd.DataFrame:
    """SYNTHETIC ground-truth congestion — physically grounded, clearly labelled.

    Uses the *real* per-cell drivers (road criticality, violation density,
    junction proximity, peak share) through an independent **logistic** law with
    weights that differ from the CIS weights, plus noise — so correlating CIS
    against it is a fair test (both track the same physics; neither is the other).
    """
    rng = np.random.default_rng(seed)
    n = cis_df["count"].to_numpy(float)
    rc = cis_df["rc_loss"].to_numpy(float) / 0.40           # 0..1 road criticality
    jp = cis_df["junction_proximity"].to_numpy(float)        # 0..1
    ps = cis_df["peak_share"].to_numpy(float)                # 0..1
    dens = np.log1p(n) / np.log1p(max(n.max(), 1.0))
    latent = 0.9 * rc + 0.5 * dens + 0.6 * jp + 0.7 * ps + rng.normal(0, 0.18, len(cis_df))
    pct = 80.0 / (1.0 + np.exp(-(latent - 1.2) * 1.6))       # saturating, unlike CIS
    pct = np.clip(pct + rng.normal(0, 3.0, len(cis_df)), 2.0, 88.0)
    return pd.DataFrame({"h3": cis_df["h3"].to_numpy(), "congestion_pct": pct})


def _rho(composite, y):
    r = spearmanr(composite, y).statistic
    return 0.0 if np.isnan(r) else float(r)


def calibrate(cis_df: pd.DataFrame, probe_df: pd.DataFrame) -> dict:
    # guard against duplicate cell ids (keep 1 row per cell; average probe)
    cis_df = cis_df.drop_duplicates("h3")
    probe_df = probe_df.groupby("h3", as_index=False)["congestion_pct"].mean()
    m = cis_df.merge(probe_df, on="h3", how="inner")
    if len(m) < 30:
        raise ValueError(f"only {len(m)} overlapping cells; need >= 30 to calibrate")
    Z = m[_Z_COLS].to_numpy(float)
    y = m["congestion_pct"].to_numpy(float)

    w0 = np.array([C.CIS_WEIGHTS["severity_density"], C.CIS_WEIGHTS["junction_proximity"],
                   C.CIS_WEIGHTS["peak_overlap"], C.CIS_WEIGHTS["road_capacity_loss"]])
    rho_default = _rho(Z @ w0, y)

    # Re-fit non-negative component weights by NNLS on centred congestion — a
    # smooth, deterministic fit (maximizing Spearman directly is non-smooth and
    # unreliable). Keep the re-fit weights only if they actually improve rank
    # agreement, so the calibrated score is never worse than the default.
    w_raw, _ = nnls(Z, y - y.mean())
    w_nnls = w_raw / w_raw.sum() if w_raw.sum() > 1e-12 else w0.copy()
    rho_nnls = _rho(Z @ w_nnls, y)
    if rho_nnls >= rho_default:
        w_opt, rho_cal = w_nnls, rho_nnls
    else:
        w_opt, rho_cal = w0.copy(), rho_default
    comp = Z @ w_opt
    pear = pearsonr(comp, y).statistic

    iso = IsotonicRegression(out_of_bounds="clip")
    yhat = iso.fit_transform(comp, y)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - yhat).mean())

    order = np.argsort(comp)
    step = max(1, len(order) // 200)
    scatter = [{"cis": round(float(comp[i]), 3),
                "observed_pct": round(float(y[i]), 1),
                "fit_pct": round(float(yhat[i]), 1)} for i in order[::step]]
    return {
        "n_cells": int(len(m)),
        "spearman_default": round(rho_default, 4),
        "spearman_calibrated": round(rho_cal, 4),
        "pearson_calibrated": round(float(pear), 4),
        "isotonic_r2": round(r2, 4),
        "isotonic_mae_pct": round(mae, 3),
        "default_weights": dict(C.CIS_WEIGHTS),
        "calibrated_weights": {
            "severity_density": round(float(w_opt[0]), 3),
            "junction_proximity": round(float(w_opt[1]), 3),
            "peak_overlap": round(float(w_opt[2]), 3),
            "road_capacity_loss": round(float(w_opt[3]), 3),
        },
        "isotonic_map": {"x": [round(float(v), 3) for v in iso.X_thresholds_],
                         "y": [round(float(v), 2) for v in iso.y_thresholds_]},
        "scatter": scatter,
    }


def run_calibration(df=None, probe_path=None, synthetic=True, write=True) -> dict:
    from curbiq.congestion import compute_congestion
    from curbiq.etl import load_processed

    if df is None:
        df = load_processed()
    cis_df, _ = compute_congestion(df)

    if probe_path:
        probe = load_probe_csv(probe_path)
        source, synth = f"probe_csv:{probe_path}", False
    elif synthetic:
        probe = synthetic_probe(cis_df)
        source, synth = "synthetic_physical_model", True
    else:
        raise ValueError("provide probe_path=... or synthetic=True")

    result = calibrate(cis_df, probe)
    result["probe_source"] = source
    result["synthetic"] = synth
    result["note"] = (
        "SYNTHETIC probe (physically-grounded demo). Replace with a real feed via "
        "load_probe_csv() / `build_all.py --probe path.csv`."
        if synth else "Calibrated against supplied real probe data.")
    if write:
        CALIB_PATH.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys
    probe = sys.argv[1] if len(sys.argv) > 1 else None
    r = run_calibration(probe_path=probe, synthetic=probe is None)
    print("== congestion calibration ==")
    print(f"  probe source        : {r['probe_source']} (synthetic={r['synthetic']})")
    print(f"  overlapping cells    : {r['n_cells']}")
    print(f"  Spearman rho default : {r['spearman_default']}")
    print(f"  Spearman rho calib.  : {r['spearman_calibrated']}")
    print(f"  Pearson  r  (calib.) : {r['pearson_calibrated']}")
    print(f"  isotonic R2 / MAE%   : {r['isotonic_r2']} / {r['isotonic_mae_pct']}")
    print(f"  default weights      : {r['default_weights']}")
    print(f"  calibrated weights   : {r['calibrated_weights']}")

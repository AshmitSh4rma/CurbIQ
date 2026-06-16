"""Forward-looking hotspot-emergence risk model.

The rest of CurbIQ describes the *present* (Gi*/Moran hotspots in ``hotspots``),
forecasts the *next day* count (``forecast``), and types *historical* trends with
Mann-Kendall (``hotspots.emerging_hotspots``). This module adds the missing
forward-looking verb: **which H3 cells are at risk of BECOMING hotspots soon** —
cells that are not hot today but are trending toward the hot threshold.

Framing (a supervised early-warning classifier on the forecast panel):

* "Currently hot" at time ``t`` is a fast trailing-window proxy — a cell whose
  strictly-past ``TRAILING_WINDOW_DAYS`` count sits in the top
  ``HOTSPOT_LABEL_TOP_FRAC`` decile among that day's active cells. (We do NOT
  recompute full Gi* per time-slice; this proxy is monotone with intensity and
  cheap.)
* The EMERGENCE label at ``(cell, t)`` is positive when the cell is NOT currently
  hot but its FORWARD count over the next ``EMERGENCE_HORIZON_DAYS`` rises into
  the hot decile. The forward window is the LABEL ONLY — it may look ahead.
  FEATURES are the panel's strictly-past ``feature_cols`` (lags / rolling means /
  lagged neighbour activity); no future-derived feature is ever added.

Anti-leakage & honesty:

* A LightGBM **binary** classifier is trained on an EARLIER period and AUC is
  reported on a LATER period (temporal holdout — never a random split), against
  a trend/persistence baseline (rank by recent rolling-mean activity). CurbIQ
  reports the honest lift; the ranked ``emergence_risk`` stays useful even when
  the lift over the baseline is modest.
* The published ``emergence_risk`` scores each cell's LATEST available feature
  row with a model refit on all labelled rows. Currently-hot cells are scored
  too but excluded from ``predicted_emerging`` (they have already emerged).

The public entry point is :func:`run_emergence`, a pure function returning a
``{"summary": dict, "cells": DataFrame}`` dict (the artifact layer serializes it;
this module never writes files, serves, or rounds).
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from curbiq import config as C
from curbiq.forecast import make_features

# --------------------------------------------------------------------------- #
# Tunables — single source of truth in config.py (rationale documented there).
# Bound to local names for readability; values never diverge from config.
# --------------------------------------------------------------------------- #
EMERGENCE_HORIZON_DAYS = C.EMERGENCE_HORIZON_DAYS
EMERGENCE_TRAILING_WINDOW_DAYS = C.EMERGENCE_TRAILING_WINDOW_DAYS
EMERGENCE_TOP_FRAC = C.HOTSPOT_LABEL_TOP_FRAC
EMERGENCE_MIN_HOT_COUNT = C.EMERGENCE_MIN_HOT_COUNT
EMERGENCE_EVAL_MONTHS = C.EMERGENCE_EVAL_MONTHS
EMERGENCE_BAND_HIGH_PCTILE = C.EMERGENCE_BAND_HIGH_PCTILE
EMERGENCE_BAND_ELEVATED_PCTILE = C.EMERGENCE_BAND_ELEVATED_PCTILE
EMERGENCE_RISK_FLOOR = C.EMERGENCE_RISK_FLOOR
# predicted_emerging flags the top EMERGENCE_TOP_FRAC by risk among cells NOT
# currently hot (rank-based, gated by EMERGENCE_RISK_FLOOR). Classifier params:
_LGBM_CLF_PARAMS = C.EMERGENCE_LGBM_PARAMS


# --------------------------------------------------------------------------- #
# Labelling helpers
# --------------------------------------------------------------------------- #
def _hot_threshold_per_day(values: pd.Series) -> float:
    """Top-``EMERGENCE_TOP_FRAC`` decile threshold over POSITIVE values only.

    Restricting the quantile to positive counts stops the threshold collapsing to
    0 on a panel that is ~75% zeros (which would otherwise label every cell hot).
    Floored at ``EMERGENCE_MIN_HOT_COUNT`` so a quiet day cannot crown noise.
    """
    pos = values[values > 0]
    if pos.empty:
        return np.inf
    thr = float(pos.quantile(1.0 - EMERGENCE_TOP_FRAC))
    return max(thr, EMERGENCE_MIN_HOT_COUNT)


def _add_emergence_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach trailing/forward windowed counts and the binary emergence label.

    Adds (per ``(h3, dt)`` row):
      * ``trail_count``      strictly-past rolling sum over TRAILING_WINDOW_DAYS
      * ``currently_hot``    trail_count in top decile of that day's active cells
      * ``forward_count``    LABEL-ONLY forward sum over the next HORIZON_DAYS
      * ``forward_hot``      forward_count in top decile of the forward window
      * ``label``            (NOT currently_hot) AND forward_hot  -> emergence
      * ``label_valid``      both windows fully observed (drop edges for training)
    """
    panel = panel.sort_values(["h3", "dt"]).reset_index(drop=True)
    g = panel.groupby("h3", sort=False)["count"]

    # trailing (strictly past): shift(1) so today's own count never leaks in.
    panel["trail_count"] = g.transform(
        lambda s: s.shift(1).rolling(EMERGENCE_TRAILING_WINDOW_DAYS, min_periods=1).sum())
    # how many trailing days are actually observed (drops the warm-up edge).
    panel["_trail_obs"] = g.transform(
        lambda s: s.shift(1).rolling(EMERGENCE_TRAILING_WINDOW_DAYS, min_periods=1).count())

    # forward (label only, looks ahead): sum of the NEXT HORIZON_DAYS counts,
    # i.e. a forward rolling sum that excludes today. Reverse -> roll -> reverse.
    def _fwd_sum(s: pd.Series) -> pd.Series:
        rev = s.iloc[::-1]
        fwd = rev.shift(1).rolling(EMERGENCE_HORIZON_DAYS, min_periods=1).sum()
        return fwd.iloc[::-1]

    def _fwd_obs(s: pd.Series) -> pd.Series:
        rev = s.iloc[::-1]
        cnt = rev.shift(1).rolling(EMERGENCE_HORIZON_DAYS, min_periods=1).count()
        return cnt.iloc[::-1]

    panel["forward_count"] = g.transform(_fwd_sum)
    panel["_fwd_obs"] = g.transform(_fwd_obs)

    # per-day hot thresholds (independent for the trailing and forward windows).
    trail_thr = panel.groupby("dt")["trail_count"].transform(_hot_threshold_per_day)
    fwd_thr = panel.groupby("dt")["forward_count"].transform(_hot_threshold_per_day)
    panel["currently_hot"] = panel["trail_count"] >= trail_thr
    panel["forward_hot"] = panel["forward_count"] >= fwd_thr

    panel["label"] = (~panel["currently_hot"] & panel["forward_hot"]).astype(int)
    # a row is a valid training/eval example only when both windows are fully
    # observed AND the strictly-past features exist (lag28 / rmean drop warm-up).
    full_trail = panel["_trail_obs"] >= EMERGENCE_TRAILING_WINDOW_DAYS
    full_fwd = panel["_fwd_obs"] >= EMERGENCE_HORIZON_DAYS
    panel["label_valid"] = full_trail & full_fwd
    return panel.drop(columns=["_trail_obs", "_fwd_obs"])


def _risk_bands(risk: np.ndarray) -> np.ndarray:
    """Assign per-cell bands by percentile of the risk vector (scale-free).

    Cells with risk below ``EMERGENCE_RISK_FLOOR`` are always "low" regardless of
    their percentile (so a flat field of ~0 risk is never crowned "high").
    """
    risk = np.asarray(risk, dtype=float)
    bands = np.full(len(risk), "low", dtype=object)
    active = risk > EMERGENCE_RISK_FLOOR
    if active.sum() == 0:
        return bands
    hi_thr = np.quantile(risk, EMERGENCE_BAND_HIGH_PCTILE)
    el_thr = np.quantile(risk, EMERGENCE_BAND_ELEVATED_PCTILE)
    bands[active & (risk >= hi_thr)] = "high"
    bands[active & (risk < hi_thr) & (risk >= el_thr)] = "elevated"
    return bands


# --------------------------------------------------------------------------- #
# Honest temporal-holdout evaluation
# --------------------------------------------------------------------------- #
def _temporal_eval(train_lab: pd.DataFrame, eval_lab: pd.DataFrame,
                   feature_cols: list[str]) -> tuple[float, float]:
    """Fit on the earlier period, score the later period; return (model, baseline) AUC.

    Both AUCs use the same evaluation rows. The baseline ranks by recent activity
    (``rmean7`` — a persistence/trend proxy); if that column is unavailable it
    falls back to ``trail_count``. AUC is NaN when a side has a single class.
    """
    from sklearn.metrics import roc_auc_score

    y_eval = eval_lab["label"].to_numpy()
    if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
        return float("nan"), float("nan")

    # --- trend/persistence baseline (no model) ---
    base_col = "rmean7" if "rmean7" in eval_lab.columns else "trail_count"
    base_score = eval_lab[base_col].fillna(0.0).to_numpy()
    try:
        baseline_auc = float(roc_auc_score(y_eval, base_score))
    except ValueError:
        baseline_auc = float("nan")

    # --- model AUC (only if the train period has both classes) ---
    y_train = train_lab["label"].to_numpy()
    if len(train_lab) == 0 or y_train.sum() == 0 or y_train.sum() == len(y_train):
        return float("nan"), baseline_auc
    clf = LGBMClassifier(**_LGBM_CLF_PARAMS)
    clf.fit(train_lab[feature_cols], y_train)
    proba = clf.predict_proba(eval_lab[feature_cols])[:, 1]
    try:
        model_auc = float(roc_auc_score(y_eval, proba))
    except ValueError:
        model_auc = float("nan")
    return model_auc, baseline_auc


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_emergence(df: pd.DataFrame, res: int = C.H3_RES_PRIMARY) -> dict:
    """Predict which H3 cells are at risk of becoming hotspots in the near future.

    Parameters
    ----------
    df : processed violations DataFrame (must contain ``is_counted`` and the H3
        columns ``h3_r8``/``h3_r9``/``h3_r10`` plus the fields the forecast panel
        builder reads). Only ``is_counted`` rows are used (filtered downstream).
    res : H3 resolution (default ``C.H3_RES_PRIMARY`` = 9, aligning the output
        with the res-9 statistical-hotspot map for integration).

    Returns
    -------
    dict with ``summary`` (metadata + honest metrics) and ``cells`` (one row per
    H3 cell with ``emergence_risk`` and its band/flags). Numbers are NOT rounded.
    """
    # Strictly-past panel; enrich=False keeps this hermetic & fast (no network,
    # no metro/weather joins) — those enrichment cols add little to emergence.
    panel, feature_cols, meta = make_features(df, res=res, enrich=False)
    panel = _add_emergence_labels(panel)

    # rows usable for supervised learning: valid label window + observed features
    lab = panel[panel["label_valid"]].dropna(subset=feature_cols).copy()

    # temporal holdout: the LATEST EMERGENCE_EVAL_MONTHS month(s) that actually
    # contain valid labels become the eval period; everything earlier (minus a
    # HORIZON-day embargo so a train row's forward window can't overlap eval)
    # trains. We pick from labelled months — NOT the calendar-last month — because
    # the final ~HORIZON days have no observable forward window, so their label
    # set is structurally empty and would make a naive last-month holdout NaN.
    model_auc, baseline_auc = float("nan"), float("nan")
    if not lab.empty:
        lab_months = sorted(lab["month"].unique())
        if len(lab_months) > EMERGENCE_EVAL_MONTHS:
            eval_months = set(lab_months[-EMERGENCE_EVAL_MONTHS:])
            eval_lab = lab[lab["month"].isin(eval_months)]
            eval_start = eval_lab["dt"].min()
            embargo = eval_start - pd.Timedelta(days=EMERGENCE_HORIZON_DAYS)
            train_lab = lab[(~lab["month"].isin(eval_months)) & (lab["dt"] < embargo)]
            if not eval_lab.empty:
                model_auc, baseline_auc = _temporal_eval(train_lab, eval_lab, feature_cols)

    # --- final model: refit on ALL labelled rows, then score each cell's latest
    #     available feature row (strictly past) for the published risk. ---
    risk_by_cell: dict[str, float] = {}
    method = "lgbm_binary_emergence"
    if not lab.empty and 0 < lab["label"].sum() < len(lab):
        clf = LGBMClassifier(**_LGBM_CLF_PARAMS)
        clf.fit(lab[feature_cols], lab["label"].to_numpy())
        scored = panel.dropna(subset=feature_cols)
        latest = scored.sort_values("dt").groupby("h3").tail(1)
        proba = clf.predict_proba(latest[feature_cols])[:, 1]
        risk_by_cell = dict(zip(latest["h3"].to_numpy(), proba.astype(float)))
    else:
        # Degenerate label (too little history / no positives) -> fall back to a
        # normalized recent-activity score so the API still gets a ranked signal.
        method = "recent_activity_fallback"
        scored = panel.dropna(subset=["rmean7"]) if "rmean7" in panel.columns else panel
        latest = scored.sort_values("dt").groupby("h3").tail(1)
        base_col = "rmean7" if "rmean7" in latest.columns else "trail_count"
        raw = latest[base_col].fillna(0.0).to_numpy(dtype=float)
        hi = raw.max()
        norm = raw / hi if hi > 0 else np.zeros_like(raw)
        risk_by_cell = dict(zip(latest["h3"].to_numpy(), norm.astype(float)))

    # --- assemble per-cell output table ---------------------------------
    # latest strictly-past state per cell (currently_hot + recent activity).
    latest_state = (panel.sort_values("dt").groupby("h3")
                    .agg(currently_hot=("currently_hot", "last"),
                         recent_count=("trail_count", "last")))
    cells = sorted(meta["cells"])
    centroids = [h3.cell_to_latlng(c) for c in cells]
    risk = np.array([risk_by_cell.get(c, 0.0) for c in cells], dtype=float)
    cur_hot = latest_state["currently_hot"].reindex(cells).fillna(False).to_numpy().astype(bool)
    recent = latest_state["recent_count"].reindex(cells).fillna(0.0).to_numpy(dtype=float)

    # predicted_emerging: top EMERGENCE_TOP_FRAC by risk among NON-currently-hot
    # cells (rank-based), gated by the numeric floor.
    predicted_emerging = np.zeros(len(cells), dtype=bool)
    cand = (~cur_hot) & (risk > EMERGENCE_RISK_FLOOR)
    if cand.any():
        n_flag = int(np.ceil(EMERGENCE_TOP_FRAC * int(cand.sum())))
        cand_idx = np.where(cand)[0]
        top = cand_idx[np.argsort(risk[cand_idx], kind="stable")[-n_flag:]]
        predicted_emerging[top] = True

    out = pd.DataFrame({
        "h3": cells,
        "lat": [p[0] for p in centroids],
        "lon": [p[1] for p in centroids],
        "emergence_risk": risk,
        "risk_band": _risk_bands(risk),
        "currently_hotspot": cur_hot,
        "predicted_emerging": predicted_emerging,
        "recent_count": recent,
    }).sort_values("emergence_risk", ascending=False).reset_index(drop=True)

    pos = int(lab["label"].sum()) if not lab.empty else 0
    summary = {
        "method": method,
        "res": res,
        "horizon_days": EMERGENCE_HORIZON_DAYS,
        "n_cells": int(len(out)),
        "n_currently_hot": int(out["currently_hotspot"].sum()),
        "n_predicted_emerging": int(out["predicted_emerging"].sum()),
        "model_auc": model_auc,
        "baseline_auc": baseline_auc,
        "label_positive_rate": float(pos / len(lab)) if not lab.empty else 0.0,
    }
    return {"summary": summary, "cells": out}


if __name__ == "__main__":
    from curbiq.etl import load_processed

    df = load_processed()
    r = run_emergence(df)
    s = r["summary"]
    print("== emergence summary ==")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("\n== risk-band counts ==")
    print(r["cells"]["risk_band"].value_counts().to_dict())
    print("\n== top-10 emerging-risk cells (not currently hot) ==")
    emerging = r["cells"][~r["cells"]["currently_hotspot"]]
    cols = ["h3", "lat", "lon", "emergence_risk", "risk_band",
            "predicted_emerging", "recent_count"]
    print(emerging.head(10)[cols].to_string(index=False))

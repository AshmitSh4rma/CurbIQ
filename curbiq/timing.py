"""Enforcement *time-window* targeting.

``prioritize.py`` answers **where** to enforce; this module answers **when**.
For every significant hotspot cell — and for the city as a whole — it computes
the recommended enforcement HOURS: the contiguous time window that captures the
most violations, *weighted by* modeled congestion peak-overlap so that, when two
candidate windows capture similar volumes, the one overlapping the morning/
evening congestion peaks wins.

Why this is needed (and why it is honest about its data): the dataset's recorded
times are patrol-shift biased — police record what they happen to be out
patrolling. So we never claim the hour profile *measures* when illegal parking
happens. But the hour-of-day distribution still reveals where illegal parking
*concentrates*, and cross-referencing it with the modeled congestion peaks
surfaces the headline finding: the **17:00-21:00 IST evening peak is badly
under-enforced** relative to its congestion impact (see ``fairness.temporal_gap``
for the citywide risk-vs-enforcement gap that motivates this).

Pure & importable: ``run_timing(df, ...)`` returns a dict (a ``summary``
sub-dict + a ``cells`` DataFrame). It writes nothing and serves nothing — the
orchestrator (``artifacts.py``) serializes the result; the API/dashboard only
read JSON. Centroids are H3-derived (``h3.cell_to_latlng``) to honour the
DPDP-Act privacy contract; raw points never leave the processing layer.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from curbiq import config as C

# --------------------------------------------------------------------------- #
# Tunables — single source of truth in config.py (rationale documented there).
# Bound to local names for readability; values never diverge from config.
# --------------------------------------------------------------------------- #
WINDOW_HOURS = C.TIMING_WINDOW_HOURS
PEAK_WEIGHT = C.TIMING_PEAK_WEIGHT
MAX_CITY_WINDOWS = C.TIMING_MAX_CITY_WINDOWS
MIN_WINDOW_GAP_HOURS = C.TIMING_MIN_WINDOW_GAP_HOURS

_RES_COL = {8: "h3_r8", 9: "h3_r9", 10: "h3_r10"}
_HOURS = np.arange(24)


# --------------------------------------------------------------------------- #
# Peak-overlap weights (same peak logic as ``features.peak_overlap``)
# --------------------------------------------------------------------------- #
def _peak_weight_vector(is_weekend: bool = False) -> np.ndarray:
    """24-vector of congestion peak-overlap in [0, 1] per IST hour.

    Reuses ``features.peak_overlap`` so the timing weighting and the CIS share
    the single source of truth for "is this hour a congestion peak".
    """
    from curbiq.features import peak_overlap

    return np.array([peak_overlap(int(h), is_weekend) for h in range(24)], dtype=float)


def _best_window(hist: np.ndarray, weights: np.ndarray,
                 window: int = WINDOW_HOURS) -> tuple[int, int, float]:
    """Best contiguous ``window``-hour slot over a 24-h histogram.

    The window is chosen to maximize the *congestion-weighted* captured count
    ``sum(hist[h] * (1 + PEAK_WEIGHT * weights[h]))``; ties broken by raw
    captured count, then by earliest start. Hours wrap around midnight.

    Returns ``(start_hour, end_hour_exclusive, raw_share)`` where ``raw_share``
    is the *unweighted* fraction of the histogram's mass inside the window.
    """
    total = float(hist.sum())
    if total <= 0:
        return 0, window % 24, 0.0

    weighted = hist * (1.0 + PEAK_WEIGHT * weights)
    # circular prefix sums via tiling (window <= 24, so two copies suffice).
    w2 = np.concatenate([weighted, weighted])
    h2 = np.concatenate([hist, hist])
    best_start, best_w, best_raw = 0, -1.0, -1.0
    for s in range(24):
        ww = float(w2[s:s + window].sum())
        rr = float(h2[s:s + window].sum())
        if ww > best_w or (ww == best_w and rr > best_raw):
            best_start, best_w, best_raw = s, ww, rr
    end = (best_start + window) % 24
    return int(best_start), int(end), best_raw / total


def _window_share(hist: np.ndarray, start: int, window: int = WINDOW_HOURS) -> float:
    """Unweighted share of a histogram captured by ``window`` hours from ``start`` (wraps)."""
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    hours = [(start + i) % 24 for i in range(window)]
    return float(hist[hours].sum()) / total


def _peak_band_share(hist: np.ndarray, band: tuple[int, int]) -> float:
    """Unweighted share of a histogram inside a half-open IST hour band."""
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    lo, hi = band
    return float(hist[lo:hi].sum()) / total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_timing(df: pd.DataFrame, hot_df: pd.DataFrame | None = None,
               res: int = C.H3_RES_PRIMARY) -> dict:
    """Recommend enforcement time windows per hotspot cell and city-wide.

    Parameters
    ----------
    df
        Processed violations frame (must carry ``is_counted``, ``hour``,
        ``is_weekend`` and the res H3 column).
    hot_df
        Optional ``compute_hotspots`` output (indexed by h3). If given, per-cell
        windows are computed for the significant hotspot cells (``is_hotspot``);
        if None, for every cell with ``count >= K_ANON_MIN`` counted violations
        (the public k-anonymity floor).
    res
        H3 resolution; selects the ``h3_r{8,9,10}`` column.

    Returns
    -------
    dict
        ``{"summary": {...}, "cells": pd.DataFrame}`` — see module docstring /
        the project contract for the exact schema.
    """
    if res not in _RES_COL:
        raise ValueError(f"unsupported resolution {res}; use one of {list(_RES_COL)}")
    col = _RES_COL[res]
    d = df[df["is_counted"]]

    weekday_w = _peak_weight_vector(is_weekend=False)
    weekend_w = _peak_weight_vector(is_weekend=True)

    # --- citywide hour profiles ----------------------------------------- #
    global_hist = (d.groupby("hour", observed=True).size()
                   .reindex(range(24), fill_value=0).to_numpy().astype(float))
    weekday_hist = (d[~d["is_weekend"]].groupby("hour", observed=True).size()
                    .reindex(range(24), fill_value=0).to_numpy().astype(float))
    weekend_hist = (d[d["is_weekend"]].groupby("hour", observed=True).size()
                    .reindex(range(24), fill_value=0).to_numpy().astype(float))

    peak_hour_city = int(np.argmax(global_hist)) if global_hist.sum() > 0 else 0
    evening_peak_share = _peak_band_share(global_hist, C.EVENING_PEAK)
    morning_peak_share = _peak_band_share(global_hist, C.MORNING_PEAK)

    recommended_windows = _recommend_city_windows(global_hist, weekday_w)

    # --- per-cell hour histograms & best windows ------------------------ #
    cell_hist = (d.groupby([col, "hour"], observed=True).size()
                 .unstack("hour").reindex(columns=range(24), fill_value=0).fillna(0))
    cell_counts = cell_hist.sum(axis=1)

    if hot_df is not None:
        focus = [c for c in hot_df.index[hot_df["is_hotspot"].astype(bool)]
                 if c in cell_hist.index]
    else:
        focus = cell_counts.index[cell_counts >= C.K_ANON_MIN].tolist()

    rows = []
    for cell in focus:
        hist = cell_hist.loc[cell].to_numpy().astype(float)
        n = int(hist.sum())
        if n <= 0:
            continue
        lat, lon = h3.cell_to_latlng(cell)
        start, end, w_share = _best_window(hist, weekday_w, WINDOW_HOURS)
        rows.append({
            "h3": cell,
            "lat": float(lat),
            "lon": float(lon),
            "n": n,
            "peak_hour": int(np.argmax(hist)),
            "window_start": start,
            "window_end": end,
            "window_share": w_share,
            "morning_share": _peak_band_share(hist, C.MORNING_PEAK),
            "evening_share": _peak_band_share(hist, C.EVENING_PEAK),
        })

    cells = pd.DataFrame(rows, columns=[
        "h3", "lat", "lon", "n", "peak_hour", "window_start", "window_end",
        "window_share", "morning_share", "evening_share",
    ])
    if not cells.empty:
        cells = cells.sort_values("n", ascending=False).reset_index(drop=True)
        for c in ("n", "peak_hour", "window_start", "window_end"):
            cells[c] = cells[c].astype(int)

    summary = {
        "resolution": res,
        "window_hours": WINDOW_HOURS,
        "peak_weight": PEAK_WEIGHT,
        "global_hourly_ist": {int(h): int(v) for h, v in zip(_HOURS, global_hist)},
        "weekday_hourly": {int(h): int(v) for h, v in zip(_HOURS, weekday_hist)},
        "weekend_hourly": {int(h): int(v) for h, v in zip(_HOURS, weekend_hist)},
        "recommended_windows": recommended_windows,
        "evening_peak_share": evening_peak_share,
        "morning_peak_share": morning_peak_share,
        "peak_hour": peak_hour_city,
        "n_cells": int(len(cells)),
    }
    return {"summary": summary, "cells": cells}


def _recommend_city_windows(global_hist: np.ndarray, weights: np.ndarray,
                            window: int = WINDOW_HOURS) -> list[dict]:
    """Greedily pick up to ``MAX_CITY_WINDOWS`` non-overlapping shift windows.

    Each pick is the congestion-weighted best window over the *residual*
    histogram (already-covered hours zeroed), enforcing a minimum start-hour
    separation so the windows target different parts of the day. ``share`` is the
    unweighted share of all citywide violations the window captures. Windows are
    returned sorted by start hour and labeled morning/midday/evening/night.
    """
    total = float(global_hist.sum())
    if total <= 0:
        return []

    residual = global_hist.copy()
    starts: list[int] = []
    chosen: list[dict] = []
    for _ in range(MAX_CITY_WINDOWS):
        if residual.sum() <= 0:
            break
        s, _e, _share = _best_window(residual, weights, window)
        if any(_circular_gap(s, t) < MIN_WINDOW_GAP_HOURS for t in starts):
            # zero out this slot and retry so we don't loop on the same peak
            residual = _zero_window(residual, s, window)
            continue
        starts.append(s)
        share = _window_share(global_hist, s, window)  # share of the *full* profile
        chosen.append({
            "label": _window_label(s),
            "start_hour": int(s),
            "end_hour": int((s + window) % 24),
            "share": float(share),
        })
        residual = _zero_window(residual, s, window)

    chosen.sort(key=lambda w: w["start_hour"])
    return chosen


def _zero_window(hist: np.ndarray, start: int, window: int) -> np.ndarray:
    out = hist.copy()
    for i in range(window):
        out[(start + i) % 24] = 0.0
    return out


def _circular_gap(a: int, b: int) -> int:
    """Smallest forward/backward hour distance between two start hours on a 24-clock."""
    d = abs(a - b) % 24
    return min(d, 24 - d)


def _window_label(start: int) -> str:
    """Human label for a window by its start hour (rough IST shift naming)."""
    if 5 <= start < 11:
        return "morning"
    if 11 <= start < 16:
        return "midday"
    if 16 <= start < 21:
        return "evening"
    return "night"


if __name__ == "__main__":
    from curbiq.etl import load_processed
    from curbiq.hotspots import compute_hotspots

    df = load_processed()
    hot, _ = compute_hotspots(df)
    r = run_timing(df, hot)
    s = r["summary"]
    print("== timing summary ==")
    print(f"  resolution: {s['resolution']}   window_hours: {s['window_hours']}")
    print(f"  citywide peak_hour: {s['peak_hour']}:00 IST")
    print(f"  morning_peak_share: {s['morning_peak_share']:.3f}   "
          f"evening_peak_share: {s['evening_peak_share']:.3f}")
    print(f"  n_cells (hotspots windowed): {s['n_cells']}")
    print("  recommended city windows:")
    for w in s["recommended_windows"]:
        print(f"    [{w['label']:>8}] {w['start_hour']:02d}:00-{w['end_hour']:02d}:00  "
              f"captures {w['share']:.1%}")
    print("\n  citywide hourly profile (IST hour: count):")
    gh = s["global_hourly_ist"]
    print("   " + "  ".join(f"{h:02d}:{gh[h]}" for h in range(24)))
    print("\n== top-8 hotspot cells by volume ==")
    print(r["cells"].head(8)[["h3", "n", "peak_hour", "window_start", "window_end",
                              "window_share", "evening_share"]].round(3).to_string(index=False))

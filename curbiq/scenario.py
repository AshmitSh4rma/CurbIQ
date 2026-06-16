"""Congestion-ROI / what-if "delay recovery" layer.

``congestion.py`` *models* how much extra traffic delay each cell's illegal
parking induces (HCM capacity-loss -> BPR delay multiplier). This module turns
that model into an **actionable enforcement-ROI lens**: for every hotspot, how
much modeled traffic delay is *recoverable* if its illegal parking is cleared,
ranked highest-first, with a city rollup and a coverage-vs-delay-recovered curve.

The headline definition is deliberately the same one congestion.py already sums
into its city index, so the two reconcile exactly::

    recoverable_delay_cell = (bpr_delay_ratio - 1) * count          # per cell
    city_recoverable_delay_index = sum(recoverable_delay_cell)      # == congestion's
                                                                    #    city_delay_impact_index

Reading it as ROI: a parked-vehicle cell with delay multiplier ``M`` over
``count`` recorded events contributes ``(M - 1) * count`` units of modeled extra
delay; clearing the cell recovers that contribution. Sorting cells by it and
walking the cumulative curve answers "how few locations must we clear to recover
50% / 80% of the city's modeled parking-induced delay?" — the deployable
priority question behind the brief's "quantify impact on traffic flow".

This is a *modeled* recovery (no speed/flow ground truth exists), and assumes
clearing a cell removes its parking-induced capacity loss entirely; it is an
upper bound on what enforcement at that cell can recover. It is a pure function:
it RETURNS data and writes nothing.

A second, *non-reconciling* enrichment is also provided
(``recoverable_delay_weighted``) that up-weights cells on higher-capacity roads
(more through-lanes => a recovered delay-unit clears more vehicles); the headline
``recoverable_delay`` is left untouched so the city index still reconciles.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from curbiq import config as C

# --------------------------------------------------------------------------- #
# Tunables — single source of truth in config.py (rationale documented there).
# --------------------------------------------------------------------------- #
SCENARIO_CURVE_DENSE_HEAD = C.SCENARIO_CURVE_DENSE_HEAD
SCENARIO_CURVE_MAX_POINTS = C.SCENARIO_CURVE_MAX_POINTS
SCENARIO_COVERAGE_TARGETS = C.SCENARIO_COVERAGE_TARGETS
SCENARIO_WEIGHTED_LANE_REF = C.SCENARIO_WEIGHTED_LANE_REF
SCENARIO_WEIGHTED_LANE_EXP = C.SCENARIO_WEIGHTED_LANE_EXP

_METHOD = (
    "recoverable_delay = (bpr_delay_ratio - 1) * count per cell "
    "(HCM capacity-loss -> BPR delay multiplier); sum reconciles with "
    "congestion.city_delay_impact_index. Cells ranked desc; cumulative "
    "recovered-delay curve drives coverage targets."
)


def _coverage_curve(recoverable_sorted: np.ndarray, total: float) -> list[dict]:
    """Cumulative recovered-delay fraction vs #cells, down-sampled to <= cap points.

    ``recoverable_sorted`` must be the per-cell recoverable delay sorted
    descending. Returns a list of ``{n_cells, frac_cells, recovered_pct}`` dicts.
    """
    n = len(recoverable_sorted)
    if n == 0:
        return []
    cum = np.cumsum(recoverable_sorted)
    cum_pct = cum / total if total > 0 else np.zeros(n)

    # ranks (1-based) we sample at: every cell for the dense head, then a sparse
    # tail; always include the final cell so the curve ends at recovered=1.0.
    head = min(SCENARIO_CURVE_DENSE_HEAD, n)
    ranks = list(range(1, head + 1))
    if n > head:
        remaining = SCENARIO_CURVE_MAX_POINTS - head
        if remaining >= 1:
            step = max(1, int(np.ceil((n - head) / remaining)))
            ranks.extend(range(head + step, n + 1, step))
        if ranks[-1] != n:
            ranks.append(n)

    out = []
    for r in ranks:
        out.append({
            "n_cells": int(r),
            "frac_cells": float(r / n),
            "recovered_pct": float(cum_pct[r - 1]),
        })
    return out


def run_scenario(
    df: pd.DataFrame,
    cis_df: pd.DataFrame | None = None,
    res: int = C.H3_RES_PRIMARY,
) -> dict:
    """Congestion-ROI / what-if delay-recovery layer (see module docstring).

    Parameters
    ----------
    df : processed violations frame (``is_counted`` filtered internally by the
        congestion model when ``cis_df`` must be computed).
    cis_df : optional pre-computed Congestion Impact Score frame from
        ``compute_congestion`` (reset_index, has ``h3``). If ``None`` it is
        computed here; if provided it is used as-is to avoid recompute.
    res : H3 resolution (must match the one used to build ``cis_df``).

    Returns
    -------
    dict with keys ``summary`` (dict), ``cells`` (DataFrame, ranked desc),
    ``curve`` (list of dicts). See module docstring / task contract.
    """
    if cis_df is None:
        from curbiq.congestion import compute_congestion
        cis_df, _ = compute_congestion(df, res)

    # --- per-cell recoverable delay (the reconciling headline definition) ----
    c = cis_df.copy()
    # (bpr_delay_ratio - 1) is the extra-delay multiplier; * count gives the
    # event-weighted modeled extra delay that clearing the cell recovers.
    c["recoverable_delay"] = (c["bpr_delay_ratio"] - 1.0) * c["count"]
    # Guard against any tiny negative from float noise (ratio >= 1 by construction).
    c["recoverable_delay"] = c["recoverable_delay"].clip(lower=0.0)

    city_total = float(c["recoverable_delay"].sum())

    # H3 centroids (privacy: never emit raw point means).
    latlng = c["h3"].map(lambda h: h3.cell_to_latlng(h))
    c["lat"] = latlng.map(lambda t: t[0]).astype(float)
    c["lon"] = latlng.map(lambda t: t[1]).astype(float)

    # share of the city total + capacity-weighted (non-reconciling) variant
    c["recoverable_pct"] = (c["recoverable_delay"] / city_total
                            if city_total > 0 else 0.0)
    lanes = pd.to_numeric(c.get("lanes", C.DEFAULT_LANES), errors="coerce").fillna(
        C.DEFAULT_LANES)
    lane_w = (lanes / SCENARIO_WEIGHTED_LANE_REF) ** SCENARIO_WEIGHTED_LANE_EXP
    c["recoverable_delay_weighted"] = c["recoverable_delay"] * lane_w

    # --- rank descending + cumulative recovered fraction --------------------
    c = c.sort_values("recoverable_delay", ascending=False,
                      kind="mergesort").reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1, dtype=int)
    cum = c["recoverable_delay"].cumsum()
    c["cum_pct"] = (cum / city_total if city_total > 0
                    else pd.Series(np.zeros(len(c)), index=c.index)).astype(float)

    # smallest N whose cumulative recovered fraction >= each target
    def _cells_for(target: float) -> int:
        if len(c) == 0:
            return 0
        reached = c.index[c["cum_pct"] >= target]
        return int(reached[0] + 1) if len(reached) else int(len(c))

    t50, t80 = SCENARIO_COVERAGE_TARGETS
    cells_for_50 = _cells_for(t50)
    cells_for_80 = _cells_for(t80)

    # --- coverage-vs-delay-recovered curve ----------------------------------
    curve = _coverage_curve(c["recoverable_delay"].to_numpy(), city_total)

    # --- output cells frame (typed, unrounded) ------------------------------
    cells = pd.DataFrame({
        "h3": c["h3"].astype(str),
        "lat": c["lat"].astype(float),
        "lon": c["lon"].astype(float),
        "recoverable_delay": c["recoverable_delay"].astype(float),
        "recoverable_delay_weighted": c["recoverable_delay_weighted"].astype(float),
        "recoverable_pct": c["recoverable_pct"].astype(float),
        "cum_pct": c["cum_pct"].astype(float),
        "rank": c["rank"].astype(int),
        "extra_delay_pct": c["extra_delay_pct"].astype(float),
        "count": c["count"].astype(int),
    })

    summary = {
        "resolution": int(res),
        "modeled": True,
        "city_recoverable_delay_index": city_total,
        "n_cells": int(len(cells)),
        "cells_for_50pct": cells_for_50,
        "cells_for_80pct": cells_for_80,
        "top_cell_recoverable": float(cells["recoverable_delay"].iloc[0])
        if len(cells) else 0.0,
        "method": _METHOD,
    }
    return {"summary": summary, "cells": cells, "curve": curve}


if __name__ == "__main__":
    from curbiq.etl import load_processed
    from curbiq.congestion import compute_congestion

    df = load_processed()
    cis, cis_summary = compute_congestion(df)
    r = run_scenario(df, cis_df=cis)

    print("== scenario (delay-recovery ROI) summary ==")
    for k, v in r["summary"].items():
        print(f"  {k}: {v}")

    # reconciliation check vs congestion's city delay-impact index
    city_idx = cis_summary["city_delay_impact_index"]
    rec_idx = r["summary"]["city_recoverable_delay_index"]
    print("\n== reconciliation with congestion.city_delay_impact_index ==")
    print(f"  congestion city_delay_impact_index : {city_idx:.6f}")
    print(f"  scenario  city_recoverable_delay   : {rec_idx:.6f}")
    print(f"  abs diff                            : {abs(city_idx - rec_idx):.3e}")
    assert abs(city_idx - rec_idx) <= 1e-6 * max(1.0, abs(city_idx)), \
        "city index reconciliation FAILED"
    print("  reconciliation OK")

    print(f"\n== curve points: {len(r['curve'])} (head/tail) ==")
    for pt in r["curve"][:3] + r["curve"][-2:]:
        print(f"  n_cells={pt['n_cells']:>5}  frac_cells={pt['frac_cells']:.4f}  "
              f"recovered_pct={pt['recovered_pct']:.4f}")

    print("\n== top-8 cells by recoverable delay ==")
    print(r["cells"].head(8)[["h3", "recoverable_delay", "recoverable_pct",
                              "cum_pct", "rank", "extra_delay_pct", "count"]]
          .round(4).to_string(index=False))

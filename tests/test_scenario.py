"""Tests for the congestion-ROI / delay-recovery layer (curbiq.scenario).

Fast + deterministic: a tiny synthetic processed frame is built inline (no full
dataset load). Cells are seeded across a few road classes / maneuver rates so
the recoverable-delay ranking and the congestion reconciliation are non-trivial.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest

from curbiq import config as C
from curbiq.congestion import compute_congestion
from curbiq.scenario import run_scenario

CENTER = (12.9716, 77.5946)


def _synth_df(seed: int = 7) -> pd.DataFrame:
    """A small, deterministic processed frame spanning several res-9 cells.

    Distinct road classes + per-cell event volumes produce a spread of BPR delay
    multipliers and counts, so recoverable delay differs across cells.
    """
    rng = np.random.default_rng(seed)
    center = h3.latlng_to_cell(*CENTER, C.H3_RES_PRIMARY)
    cells = sorted(h3.grid_disk(center, 2))  # ~19 res-9 cells

    # vary road class (=> lanes + rc_loss) and event count across cells
    road_classes = ["arterial", "junction", "residential", "collector",
                    "ring_arterial", "local_connector"]

    rows = []
    rid = 0
    for i, cell in enumerate(cells):
        rc = road_classes[i % len(road_classes)]
        clat, clon = h3.cell_to_latlng(cell)
        n_events = 3 + (i * 4) % 37            # 3..~39 events, deterministic
        for _ in range(n_events):
            # peak vs off-peak hours so maneuver_rate_hr varies
            hour = int(rng.integers(0, 24))
            day = int(rng.integers(0, 14))     # spread across two weeks
            rows.append({
                "id": rid,
                "latitude": clat + float(rng.normal(0, 1e-4)),
                "longitude": clon + float(rng.normal(0, 1e-4)),
                "road_class": rc,
                "blockage": float(rng.uniform(0.3, 3.0)),
                "confidence": 1.0,
                "is_counted": True,
                "peak_overlap": 1.0 if (8 <= hour < 11 or 17 <= hour < 21) else 0.0,
                "hour": hour,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                "has_junction": rc == "junction",
                "junction_id": f"J{i}" if rc == "junction" else None,
                "h3_r8": h3.cell_to_parent(cell, 8),
                "h3_r9": cell,
                "h3_r10": h3.cell_to_children(cell, 10)[0],
            })
            rid += 1
    # a few non-counted rows that the congestion model must ignore
    for _ in range(5):
        rows.append({
            "id": rid, "latitude": CENTER[0], "longitude": CENTER[1],
            "road_class": "arterial", "blockage": 1.0, "confidence": 0.0,
            "is_counted": False, "peak_overlap": 1.0, "hour": 9,
            "date": pd.Timestamp("2024-01-01"), "has_junction": False,
            "junction_id": None, "h3_r8": h3.cell_to_parent(center, 8),
            "h3_r9": center, "h3_r10": h3.cell_to_children(center, 10)[0],
        })
        rid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synth_df():
    return _synth_df()


@pytest.fixture(scope="module")
def result(synth_df):
    return run_scenario(synth_df)


# --------------------------------------------------------------------------- #
# Return contract
# --------------------------------------------------------------------------- #
class TestContract:
    def test_top_level_keys(self, result):
        assert set(result.keys()) == {"summary", "cells", "curve"}

    def test_summary_keys(self, result):
        for k in ("city_recoverable_delay_index", "n_cells", "cells_for_50pct",
                  "cells_for_80pct", "top_cell_recoverable", "method"):
            assert k in result["summary"], f"missing summary key {k}"
        assert isinstance(result["summary"]["method"], str)

    def test_cells_columns(self, result):
        cols = set(result["cells"].columns)
        required = {"h3", "lat", "lon", "recoverable_delay", "recoverable_pct",
                    "cum_pct", "rank", "extra_delay_pct", "count"}
        assert required <= cols

    def test_cells_dtypes(self, result):
        cells = result["cells"]
        assert cells["h3"].map(type).eq(str).all()
        assert pd.api.types.is_float_dtype(cells["lat"])
        assert pd.api.types.is_float_dtype(cells["lon"])
        assert pd.api.types.is_float_dtype(cells["recoverable_delay"])
        assert pd.api.types.is_integer_dtype(cells["rank"])
        assert pd.api.types.is_integer_dtype(cells["count"])

    def test_curve_element_shape(self, result):
        curve = result["curve"]
        assert isinstance(curve, list) and len(curve) >= 1
        for pt in curve:
            assert set(pt.keys()) == {"n_cells", "frac_cells", "recovered_pct"}
            assert isinstance(pt["n_cells"], int)
        assert len(curve) <= 100  # sampling cap

    def test_n_cells_matches_table(self, result):
        assert result["summary"]["n_cells"] == len(result["cells"])

    def test_lat_lon_are_h3_centroids(self, result):
        # privacy: emitted lat/lon must equal h3.cell_to_latlng(cell), not raw means
        for _, row in result["cells"].iterrows():
            clat, clon = h3.cell_to_latlng(row["h3"])
            assert row["lat"] == pytest.approx(clat)
            assert row["lon"] == pytest.approx(clon)


# --------------------------------------------------------------------------- #
# Numeric properties
# --------------------------------------------------------------------------- #
class TestNumerics:
    def test_recoverable_pct_sums_to_one(self, result):
        assert result["cells"]["recoverable_pct"].sum() == pytest.approx(1.0)

    def test_cum_pct_monotone_nondecreasing(self, result):
        cum = result["cells"].sort_values("rank")["cum_pct"].to_numpy()
        assert np.all(np.diff(cum) >= -1e-12)

    def test_cum_pct_ends_at_one(self, result):
        last = result["cells"].sort_values("rank")["cum_pct"].iloc[-1]
        assert last == pytest.approx(1.0)

    def test_rank_is_unique_1_to_n(self, result):
        ranks = sorted(result["cells"]["rank"].tolist())
        assert ranks == list(range(1, len(ranks) + 1))

    def test_sorted_descending_by_recoverable(self, result):
        rec = result["cells"].sort_values("rank")["recoverable_delay"].to_numpy()
        assert np.all(np.diff(rec) <= 1e-12)

    def test_cells_for_50_le_80(self, result):
        s = result["summary"]
        assert s["cells_for_50pct"] <= s["cells_for_80pct"]

    def test_cells_for_targets_reach_thresholds(self, result):
        cells = result["cells"].sort_values("rank")
        s = result["summary"]
        assert cells["cum_pct"].iloc[s["cells_for_50pct"] - 1] >= 0.50 - 1e-12
        assert cells["cum_pct"].iloc[s["cells_for_80pct"] - 1] >= 0.80 - 1e-12

    def test_top_cell_recoverable_matches_rank1(self, result):
        top = result["cells"].sort_values("rank")["recoverable_delay"].iloc[0]
        assert result["summary"]["top_cell_recoverable"] == pytest.approx(top)

    def test_recoverable_delay_nonnegative(self, result):
        assert (result["cells"]["recoverable_delay"] >= 0).all()

    def test_curve_recovered_monotone_and_ends_at_one(self, result):
        rp = [pt["recovered_pct"] for pt in result["curve"]]
        assert np.all(np.diff(rp) >= -1e-12)
        assert rp[-1] == pytest.approx(1.0)
        assert result["curve"][-1]["n_cells"] == result["summary"]["n_cells"]


# --------------------------------------------------------------------------- #
# Reconciliation with the congestion city index (the load-bearing invariant)
# --------------------------------------------------------------------------- #
class TestReconciliation:
    def test_sum_equals_congestion_city_index(self, synth_df, result):
        _, cis_summary = compute_congestion(synth_df, C.H3_RES_PRIMARY)
        city_idx = cis_summary["city_delay_impact_index"]
        rec_idx = result["summary"]["city_recoverable_delay_index"]
        assert rec_idx == pytest.approx(city_idx, rel=1e-9, abs=1e-9)
        # and it equals the per-cell sum, too
        assert result["cells"]["recoverable_delay"].sum() == pytest.approx(city_idx)

    def test_precomputed_cis_matches_internal_compute(self, synth_df):
        cis_df, _ = compute_congestion(synth_df, C.H3_RES_PRIMARY)
        r_pre = run_scenario(synth_df, cis_df=cis_df)
        r_auto = run_scenario(synth_df, cis_df=None)
        assert (r_pre["summary"]["city_recoverable_delay_index"]
                == pytest.approx(r_auto["summary"]["city_recoverable_delay_index"]))
        assert len(r_pre["cells"]) == len(r_auto["cells"])

    def test_deterministic(self, synth_df):
        a = run_scenario(synth_df)
        b = run_scenario(synth_df)
        pd.testing.assert_frame_equal(a["cells"], b["cells"])
        assert a["summary"] == b["summary"]

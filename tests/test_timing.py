"""Timing-targeting tests (curbiq.timing): per-cell & citywide enforcement windows.

Fast & deterministic: every test builds a tiny synthetic violations frame inline
(no full-dataset load). We construct three res-9 cells around the city centre and
plant their violations at known hours so the recommended windows are predictable.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest

from curbiq import config as C
from curbiq.timing import (
    WINDOW_HOURS,
    _best_window,
    _peak_weight_vector,
    run_timing,
)

CENTER = (12.9716, 77.5946)


def _cells(n: int) -> list[str]:
    """``n`` distinct res-9 cells: the centre plus its grid-disk neighbours."""
    center = h3.latlng_to_cell(*CENTER, C.H3_RES_PRIMARY)
    ring = sorted(h3.grid_disk(center, 2))
    # keep centre first, then unique others
    ordered = [center] + [c for c in ring if c != center]
    return ordered[:n]


def _make_df(specs: list[tuple[str, list[int], int]]) -> pd.DataFrame:
    """Build a processed-style frame.

    ``specs`` is a list of ``(h3_cell, hours, reps)``: plant ``reps`` counted
    violations at each hour in ``hours`` for that cell. All weekday, all counted.
    """
    rows = []
    rid = 0
    for cell, hours, reps in specs:
        lat, lon = h3.cell_to_latlng(cell)
        for hr in hours:
            for _ in range(reps):
                rows.append({
                    "id": rid,
                    "latitude": lat,
                    "longitude": lon,
                    "hour": int(hr),
                    "is_weekend": False,
                    "is_counted": True,
                    "h3_r8": h3.cell_to_parent(cell, 8),
                    "h3_r9": cell,
                    "h3_r10": h3.cell_to_children(cell, 10)[0],
                })
                rid += 1
    df = pd.DataFrame(rows)
    df["hour"] = df["hour"].astype("int16")
    return df


# --------------------------------------------------------------------------- #
# Helper-level unit tests
# --------------------------------------------------------------------------- #
class TestBestWindow:
    def test_empty_histogram(self):
        start, end, share = _best_window(np.zeros(24), _peak_weight_vector())
        assert (start, end) == (0, WINDOW_HOURS % 24)
        assert share == 0.0

    def test_all_mass_at_one_hour_picks_window_covering_it(self):
        hist = np.zeros(24)
        hist[18] = 10.0
        start, end, share = _best_window(hist, _peak_weight_vector(), WINDOW_HOURS)
        covered = [(start + i) % 24 for i in range(WINDOW_HOURS)]
        assert 18 in covered
        assert share == pytest.approx(1.0)

    def test_share_is_unweighted_fraction(self):
        hist = np.zeros(24)
        hist[10] = 3.0   # inside morning peak
        hist[14] = 1.0   # midday
        _s, _e, share = _best_window(hist, _peak_weight_vector(), WINDOW_HOURS)
        # best 4h window centred on the peak should capture the 3 at h10 -> >=0.75
        assert share >= 0.75 - 1e-9

    def test_peak_weight_breaks_near_ties_toward_congestion_peak(self):
        # Two equal spikes: one at 03:00 (night), one at 18:00 (evening peak).
        hist = np.zeros(24)
        hist[3] = 5.0
        hist[18] = 5.0
        start, _e, _share = _best_window(hist, _peak_weight_vector(), WINDOW_HOURS)
        covered = [(start + i) % 24 for i in range(WINDOW_HOURS)]
        assert 18 in covered and 3 not in covered

    def test_window_wraps_midnight(self):
        hist = np.zeros(24)
        hist[23] = 4.0
        hist[0] = 4.0
        hist[1] = 4.0
        start, _e, share = _best_window(hist, _peak_weight_vector(), WINDOW_HOURS)
        covered = [(start + i) % 24 for i in range(WINDOW_HOURS)]
        assert {23, 0, 1}.issubset(set(covered))
        assert share == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Public contract: run_timing
# --------------------------------------------------------------------------- #
class TestRunTimingContract:
    @pytest.fixture
    def result(self):
        cells = _cells(3)
        df = _make_df([
            (cells[0], [18], 12),                 # evening-concentrated hotspot
            (cells[1], [9, 10], 6),               # morning-concentrated
            (cells[2], [13, 14, 20], 4),          # mixed
        ])
        return run_timing(df), df, cells

    def test_top_level_keys(self, result):
        r, _df, _cells = result
        assert set(r.keys()) == {"summary", "cells"}
        assert isinstance(r["cells"], pd.DataFrame)

    def test_summary_keys(self, result):
        r, _df, _cells = result
        s = r["summary"]
        for key in ("global_hourly_ist", "weekday_hourly", "weekend_hourly",
                    "recommended_windows", "evening_peak_share",
                    "morning_peak_share", "peak_hour", "n_cells"):
            assert key in s, f"missing summary key {key}"

    def test_global_hourly_covers_24_hours(self, result):
        r, _df, _cells = result
        gh = r["summary"]["global_hourly_ist"]
        assert set(gh.keys()) == set(range(24))
        assert all(isinstance(v, int) and v >= 0 for v in gh.values())

    def test_cells_required_columns(self, result):
        r, _df, _cells = result
        required = {"h3", "lat", "lon", "n", "peak_hour", "window_start",
                    "window_end", "window_share", "morning_share", "evening_share"}
        assert required.issubset(set(r["cells"].columns))

    def test_hours_within_bounds(self, result):
        r, _df, _cells = result
        c = r["cells"]
        for col in ("peak_hour", "window_start", "window_end"):
            assert c[col].between(0, 23).all(), f"{col} out of 0-23"

    def test_shares_in_unit_interval(self, result):
        r, _df, _cells = result
        c = r["cells"]
        for col in ("window_share", "morning_share", "evening_share"):
            assert c[col].between(0.0, 1.0).all(), f"{col} not in [0,1]"

    def test_n_cells_matches_rows(self, result):
        r, _df, _cells = result
        assert r["summary"]["n_cells"] == len(r["cells"])

    def test_recommended_windows_well_formed(self, result):
        r, _df, _cells = result
        rw = r["summary"]["recommended_windows"]
        assert 1 <= len(rw) <= 3
        for w in rw:
            assert set(w.keys()) == {"label", "start_hour", "end_hour", "share"}
            assert 0 <= w["start_hour"] <= 23
            assert 0 <= w["end_hour"] <= 23
            assert 0.0 <= w["share"] <= 1.0

    def test_centroids_are_h3_derived(self, result):
        r, _df, _cells = result
        # lat/lon must equal the H3 centroid of the cell (privacy contract), not
        # the mean of the planted points.
        for _, row in r["cells"].iterrows():
            clat, clon = h3.cell_to_latlng(row["h3"])
            assert row["lat"] == pytest.approx(clat)
            assert row["lon"] == pytest.approx(clon)


# --------------------------------------------------------------------------- #
# Behavioural assertions on planted distributions
# --------------------------------------------------------------------------- #
class TestRunTimingBehaviour:
    def test_evening_cell_peaks_at_18_with_evening_window(self):
        cells = _cells(1)
        df = _make_df([(cells[0], [18], 20)])
        r = run_timing(df)
        row = r["cells"].iloc[0]
        assert row["peak_hour"] == 18
        covered = [(row["window_start"] + i) % 24 for i in range(WINDOW_HOURS)]
        assert 18 in covered
        # all mass is at 18:00 -> entirely inside EVENING_PEAK band
        assert row["evening_share"] == pytest.approx(1.0)
        assert row["window_share"] == pytest.approx(1.0)

    def test_morning_cell_peaks_in_morning_band(self):
        cells = _cells(1)
        df = _make_df([(cells[0], [9, 10], 10)])
        r = run_timing(df)
        row = r["cells"].iloc[0]
        assert C.MORNING_PEAK[0] <= row["peak_hour"] < C.MORNING_PEAK[1]
        assert row["morning_share"] == pytest.approx(1.0)

    def test_hot_df_focuses_on_significant_cells_only(self):
        cells = _cells(3)
        df = _make_df([
            (cells[0], [18], 15),
            (cells[1], [9], 15),
            (cells[2], [20], 15),
        ])
        # mark only the first two cells as hotspots
        hot_df = pd.DataFrame(
            {"is_hotspot": [True, True, False]},
            index=pd.Index(cells, name="h3"),
        )
        r = run_timing(df, hot_df=hot_df)
        assert set(r["cells"]["h3"]) == {cells[0], cells[1]}
        assert r["summary"]["n_cells"] == 2

    def test_k_anon_floor_when_no_hot_df(self):
        cells = _cells(2)
        # cell 0 has >= K_ANON_MIN, cell 1 below the floor -> dropped.
        df = _make_df([
            (cells[0], [18], C.K_ANON_MIN + 2),
            (cells[1], [12], 1),
        ])
        r = run_timing(df)
        assert set(r["cells"]["h3"]) == {cells[0]}

    def test_deterministic(self):
        cells = _cells(3)
        df = _make_df([
            (cells[0], [18, 19], 7),
            (cells[1], [8, 9], 5),
            (cells[2], [14], 6),
        ])
        a = run_timing(df)
        b = run_timing(df)
        assert a["summary"] == b["summary"]
        pd.testing.assert_frame_equal(a["cells"], b["cells"])

    def test_evening_peak_share_reflects_planted_evening_mass(self):
        cells = _cells(2)
        # 30 violations at 18:00 (evening) + 10 at 03:00 (night) -> 0.75 evening.
        df = _make_df([
            (cells[0], [18], 30),
            (cells[1], [3], 10),
        ])
        r = run_timing(df)
        assert r["summary"]["evening_peak_share"] == pytest.approx(30 / 40)
        assert r["summary"]["peak_hour"] == 18

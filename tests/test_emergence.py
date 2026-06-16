"""Fast, deterministic unit tests for curbiq.emergence on a synthetic panel.

These never touch the real dataset. We build a tiny H3 lattice over ~10 months
with a population of cells whose violation rate RISES on staggered onsets (so
many cells transition from quiet -> busy across the window, generating genuine
forward-emergence labels in every month), a couple of persistent-hot anchors,
and a flat near-dead background. That exercises the full model path (LightGBM
classifier + temporal-holdout AUC) and lets us assert the core invariants:

  * the return contract (keys / columns / scalar types);
  * emergence_risk in [0, 1], finite; valid risk bands; boolean flags;
  * lat/lon are exact H3 centroids (privacy contract);
  * predicted_emerging never overlaps currently_hotspot;
  * the labelling + currently-hot proxy behave (rising cells get positive labels,
    persistent cells are flagged hot), and a rising cell out-ranks dead background;
  * determinism (seeded params -> identical output on identical input).

The classifier uses fixed, deterministic LightGBM params, so the run is
reproducible. Runtime is a few seconds (tiny grid).
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest

from curbiq import config as C
from curbiq import emergence as E
from curbiq.emergence import (_add_emergence_labels, _hot_threshold_per_day,
                              _risk_bands, run_emergence)
from curbiq.forecast import make_features

CENTER = (12.9716, 77.5946)
VALID_BANDS = {"high", "elevated", "low"}
N_RISERS = 14                 # cells 0..13 ramp; 14,15 persistent-hot; 16+ dead bg
REQUIRED_CELL_COLS = [
    "h3", "lat", "lon", "emergence_risk", "risk_band",
    "currently_hotspot", "predicted_emerging", "recent_count",
]
REQUIRED_SUMMARY_KEYS = [
    "method", "res", "horizon_days", "n_cells", "n_currently_hot",
    "n_predicted_emerging", "model_auc", "baseline_auc", "label_positive_rate",
]


def _make_synthetic(n_days: int = 300, seed: int = 21) -> tuple[pd.DataFrame, list[str]]:
    """Deterministic processed-style frame at res 9 (see module docstring)."""
    rng = np.random.default_rng(seed)
    c9 = h3.latlng_to_cell(*CENTER, 9)
    cells = sorted(h3.grid_disk(c9, 3))[:24]
    days = pd.date_range("2023-08-01", periods=n_days, freq="D")
    # staggered onsets spread across ~18%..76% of the window
    onsets = [int(n_days * (0.18 + 0.045 * i)) for i in range(N_RISERS)]

    rows = []
    rid = 0
    for di, day in enumerate(days):
        for ci, cell in enumerate(cells):
            ll = h3.cell_to_latlng(cell)
            if ci < N_RISERS:                         # gradual riser
                o = onsets[ci]
                rate = 0.3 if di < o else 0.3 + 7.0 * min(1.0, (di - o) / 30.0)
            elif ci in (N_RISERS, N_RISERS + 1):      # persistent hot anchors
                rate = 9.0
            else:                                     # flat near-dead background
                rate = 0.3
            n = int(rng.poisson(max(rate, 0.03)))
            for _ in range(n):
                rows.append({
                    "id": rid,
                    "latitude": ll[0] + rng.normal(0, 1e-4),
                    "longitude": ll[1] + rng.normal(0, 1e-4),
                    "date": day.strftime("%Y-%m-%d"),
                    "hour": int(rng.integers(0, 24)),
                    "vehicle_type": "CAR",
                    "vehicle_category": "light_4w",
                    "primary_offence": "NO PARKING",
                    "has_junction": False,
                    "police_station": "PS1",
                    "road_weight": 0.5,
                    "confidence": 1.0,
                    "is_counted": True,
                    "h3_r8": h3.cell_to_parent(cell, 8),
                    "h3_r9": cell,
                    "h3_r10": h3.cell_to_children(cell, 10)[0],
                })
                rid += 1
    return pd.DataFrame(rows), cells


@pytest.fixture(scope="module")
def synthetic():
    return _make_synthetic()


@pytest.fixture(scope="module")
def result(synthetic) -> dict:
    df, _ = synthetic
    return run_emergence(df, res=9)


# --------------------------------------------------------------------------- #
# Return contract
# --------------------------------------------------------------------------- #
class TestContract:
    def test_top_level_keys(self, result):
        assert set(result.keys()) == {"summary", "cells"}
        assert isinstance(result["summary"], dict)
        assert isinstance(result["cells"], pd.DataFrame)

    def test_summary_keys_present(self, result):
        for k in REQUIRED_SUMMARY_KEYS:
            assert k in result["summary"], f"missing summary key {k}"

    def test_cells_required_columns(self, result):
        for c in REQUIRED_CELL_COLS:
            assert c in result["cells"].columns, f"missing cells column {c}"

    def test_summary_scalar_types(self, result):
        s = result["summary"]
        assert isinstance(s["method"], str)
        assert isinstance(s["res"], int) and s["res"] == 9
        assert isinstance(s["horizon_days"], int)
        assert s["horizon_days"] == E.EMERGENCE_HORIZON_DAYS
        for k in ("n_cells", "n_currently_hot", "n_predicted_emerging"):
            assert isinstance(s[k], int)
        assert isinstance(s["model_auc"], float)
        assert isinstance(s["baseline_auc"], float)
        assert isinstance(s["label_positive_rate"], float)

    def test_summary_counts_match_table(self, result):
        s, cells = result["summary"], result["cells"]
        assert s["n_cells"] == len(cells)
        assert s["n_currently_hot"] == int(cells["currently_hotspot"].sum())
        assert s["n_predicted_emerging"] == int(cells["predicted_emerging"].sum())


# --------------------------------------------------------------------------- #
# Value ranges, dtypes, privacy contract
# --------------------------------------------------------------------------- #
class TestValues:
    def test_risk_in_unit_interval(self, result):
        r = result["cells"]["emergence_risk"].to_numpy()
        assert np.all(np.isfinite(r))
        assert r.min() >= 0.0 and r.max() <= 1.0

    def test_risk_bands_valid(self, result):
        assert set(result["cells"]["risk_band"].unique()) <= VALID_BANDS

    def test_bool_columns_are_boolean(self, result):
        cells = result["cells"]
        assert cells["currently_hotspot"].dtype == bool
        assert cells["predicted_emerging"].dtype == bool

    def test_latlon_are_h3_centroids(self, result):
        # lat/lon MUST equal the H3 centroid of the cell — never raw point means.
        for _, row in result["cells"].iterrows():
            clat, clon = h3.cell_to_latlng(row["h3"])
            assert row["lat"] == pytest.approx(clat)
            assert row["lon"] == pytest.approx(clon)

    def test_label_positive_rate_in_unit_interval(self, result):
        assert 0.0 <= result["summary"]["label_positive_rate"] <= 1.0

    def test_predicted_emerging_excludes_currently_hot(self, result):
        cells = result["cells"]
        assert not (cells["predicted_emerging"] & cells["currently_hotspot"]).any()

    def test_auc_in_range_or_nan(self, result):
        for key in ("model_auc", "baseline_auc"):
            v = result["summary"][key]
            assert np.isnan(v) or (0.0 <= v <= 1.0)

    def test_recent_count_nonnegative(self, result):
        assert (result["cells"]["recent_count"].to_numpy() >= 0).all()


# --------------------------------------------------------------------------- #
# Model path is actually exercised on this learnable synthetic
# --------------------------------------------------------------------------- #
class TestModelPath:
    def test_model_path_used(self, result):
        # the staggered-riser population yields positive labels in every month,
        # so the supervised classifier (not the fallback) drives the score.
        assert result["summary"]["method"] == "lgbm_binary_emergence"
        assert result["summary"]["label_positive_rate"] > 0.0

    def test_temporal_holdout_auc_is_finite_and_valid(self, result):
        # the honest temporal-holdout AUC is computable (not NaN) and a probability.
        auc = result["summary"]["model_auc"]
        assert not np.isnan(auc)
        assert 0.5 <= auc <= 1.0   # a learnable signal -> at least chance-level

    def test_baseline_auc_finite(self, result):
        assert not np.isnan(result["summary"]["baseline_auc"])


# --------------------------------------------------------------------------- #
# Core labelling + currently-hot proxy behave as designed
# --------------------------------------------------------------------------- #
class TestLabelling:
    def test_persistent_cells_flagged_currently_hot(self, synthetic, result):
        df, cells = synthetic
        anchors = [cells[N_RISERS], cells[N_RISERS + 1]]
        cur = result["cells"].set_index("h3")["currently_hotspot"]
        assert all(bool(cur.get(a, False)) for a in anchors)
        assert result["summary"]["n_currently_hot"] >= 1

    def test_rising_cells_generate_positive_labels(self, synthetic):
        # the emergence label captures the quiet -> hot transition for risers.
        df, cells = synthetic
        panel, fcols, _ = make_features(df, res=9, enrich=False)
        panel = _add_emergence_labels(panel)
        # at least one mid/late riser should accrue positive emergence labels
        riser_pos = sum(int(panel[panel["h3"] == cells[i]]["label"].sum())
                        for i in range(N_RISERS))
        assert riser_pos > 0
        assert int(panel["label"].sum()) > 0

    def test_rising_cell_outranks_dead_background(self, synthetic, result):
        # a late riser (recently active, not yet hot) should carry more emergence
        # risk than a flat near-dead background cell.
        df, cells = synthetic
        risk = result["cells"].set_index("h3")["emergence_risk"]
        late_riser = cells[N_RISERS - 1]
        dead = [float(risk[cells[i]]) for i in range(N_RISERS + 2, len(cells))
                if cells[i] in risk.index]
        assert late_riser in risk.index and dead
        assert float(risk[late_riser]) >= max(dead)


class TestDeterminism:
    def test_identical_inputs_identical_output(self, synthetic):
        df, _ = synthetic
        a = run_emergence(df, res=9)
        b = run_emergence(df, res=9)
        assert a["summary"] == b["summary"]
        pd.testing.assert_frame_equal(a["cells"], b["cells"])


# --------------------------------------------------------------------------- #
# Helper-level units
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_hot_threshold_floor(self):
        # all-tiny positive counts -> threshold floored at EMERGENCE_MIN_HOT_COUNT
        s = pd.Series([1.0, 1.0, 2.0, 0.0, 0.0])
        assert _hot_threshold_per_day(s) >= E.EMERGENCE_MIN_HOT_COUNT

    def test_hot_threshold_all_zero_is_inf(self):
        # no positive activity -> infinite threshold -> nobody is "hot"
        assert np.isinf(_hot_threshold_per_day(pd.Series([0.0, 0.0, 0.0])))

    def test_risk_bands_percentile_split(self):
        # a clear gradient -> top decile "high", next band "elevated", rest "low".
        risk = np.linspace(0.0, 1.0, 100)
        bands = _risk_bands(risk)
        assert set(bands) <= VALID_BANDS
        assert bands[-1] == "high"          # top risk
        assert bands[0] == "low"            # zero risk -> always low (floor)

    def test_risk_bands_all_zero_all_low(self):
        # a flat field of ~0 risk is never crowned high/elevated.
        bands = _risk_bands(np.zeros(50))
        assert set(bands) == {"low"}

    def test_top_frac_constant_aligned_with_config(self):
        # the emergence top-fraction reuses the repo-wide hotspot decile.
        assert E.EMERGENCE_TOP_FRAC == C.HOTSPOT_LABEL_TOP_FRAC

"""Spatial-statistics tests (curbiq.spatial) on a tiny known H3 lattice."""
from __future__ import annotations

import numpy as np
import pytest

from curbiq import spatial as S


# --------------------------------------------------------------------------- #
# Getis-Ord Gi*
# --------------------------------------------------------------------------- #
class TestGetisOrd:
    def test_high_positive_z_for_clear_cluster(self, hex_lattice, lattice_center,
                                                clustered_field):
        cells, idx = hex_lattice
        W = S.build_weights(cells, idx, k=1, include_self=True)
        z = S.getis_ord_gi_star(clustered_field, W)
        zc = float(z[idx[lattice_center]])
        # The centre sits inside a dense high-value blob -> strongly significant hot.
        assert zc > 3.29

    def test_near_zero_z_for_uniform_field(self, hex_lattice, uniform_field):
        cells, idx = hex_lattice
        W = S.build_weights(cells, idx, k=1, include_self=True)
        z = S.getis_ord_gi_star(uniform_field, W)
        # A flat field has zero variance -> all z forced to 0 (no clustering).
        assert np.allclose(z, 0.0, atol=1e-9)

    def test_cold_spot_is_negative(self, hex_lattice, lattice_center):
        cells, idx = hex_lattice
        # Centre + neighbours LOW, everywhere else high -> centre is a cold spot.
        import h3
        cold = {lattice_center} | set(h3.grid_disk(lattice_center, 1))
        x = np.array([0.0 if c in cold else 10.0 for c in cells])
        W = S.build_weights(cells, idx, k=1, include_self=True)
        z = S.getis_ord_gi_star(x, W)
        assert float(z[idx[lattice_center]]) < -3.29


# --------------------------------------------------------------------------- #
# Global Moran's I
# --------------------------------------------------------------------------- #
class TestGlobalMoran:
    def test_positive_for_clustered(self, hex_lattice, clustered_field):
        cells, idx = hex_lattice
        W = S.build_weights(cells, idx, k=1, include_self=False)
        res = S.global_morans_i(clustered_field, W)
        assert res.I > 0.2          # clear positive autocorrelation
        assert res.z > 2.0          # significant
        assert res.expected == pytest.approx(-1.0 / (len(cells) - 1))

    def test_random_field_near_expected(self, hex_lattice):
        cells, idx = hex_lattice
        W = S.build_weights(cells, idx, k=1, include_self=False)
        rng = np.random.default_rng(123)
        x = rng.normal(size=len(cells))
        res = S.global_morans_i(x, W)
        # I should sit close to E[I] for a spatially random field; |z| modest.
        assert abs(res.I - res.expected) < 0.2
        assert abs(res.z) < 2.0

    def test_clustered_more_autocorrelated_than_random(self, hex_lattice,
                                                        clustered_field):
        cells, idx = hex_lattice
        W = S.build_weights(cells, idx, k=1, include_self=False)
        rng = np.random.default_rng(7)
        rand = rng.normal(size=len(cells))
        assert (S.global_morans_i(clustered_field, W).I
                > S.global_morans_i(rand, W).I)


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR
# --------------------------------------------------------------------------- #
class TestBenjaminiHochberg:
    def test_known_critical_p(self):
        # thresholds q*i/n = [.01,.02,.03,.04,.05]; last p <= thresh is index 1.
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
        mask, pcrit = S.benjamini_hochberg(p, q=0.05)
        assert pcrit == pytest.approx(0.008)
        assert mask.tolist() == [True, True, False, False, False]

    def test_all_significant(self):
        p = np.array([0.0001, 0.0002, 0.0003])
        mask, pcrit = S.benjamini_hochberg(p, q=0.05)
        assert mask.all()

    def test_none_significant(self):
        p = np.array([0.6, 0.7, 0.8, 0.99])
        mask, pcrit = S.benjamini_hochberg(p, q=0.05)
        assert not mask.any()
        assert pcrit == 0.0

    def test_monotone_step_up_selection(self):
        # A large p that beats its (later) threshold pulls in all smaller ps.
        p = np.array([0.001, 0.2, 0.049])
        mask, pcrit = S.benjamini_hochberg(p, q=0.05)
        # sorted: .001(thr .0167), .049(thr .0333) no, .2 no -> kmax=0, pcrit=.001
        assert pcrit == pytest.approx(0.001)
        assert mask.tolist() == [True, False, False]

    def test_critical_p_monotone_in_q(self):
        p = np.array([0.001, 0.01, 0.02, 0.03, 0.04])
        _, pc_small = S.benjamini_hochberg(p, q=0.05)
        _, pc_large = S.benjamini_hochberg(p, q=0.20)
        assert pc_large >= pc_small

    def test_empty_input(self):
        mask, pcrit = S.benjamini_hochberg(np.zeros(0))
        assert mask.shape == (0,)
        assert pcrit == 0.0


# --------------------------------------------------------------------------- #
# Mann-Kendall trend
# --------------------------------------------------------------------------- #
class TestMannKendall:
    def test_detects_increasing(self):
        mk = S.mann_kendall(np.arange(1, 13, dtype=float))
        assert mk["trend"] == "intensifying"
        assert mk["z"] > 0
        assert mk["tau"] == pytest.approx(1.0)
        assert mk["p"] < 0.05

    def test_detects_decreasing(self):
        mk = S.mann_kendall(np.arange(12, 0, -1, dtype=float))
        assert mk["trend"] == "diminishing"
        assert mk["z"] < 0
        assert mk["tau"] == pytest.approx(-1.0)

    def test_flat_series_stable(self):
        mk = S.mann_kendall(np.full(10, 5.0))
        assert mk["trend"] == "stable"

    def test_too_short_is_insufficient(self):
        mk = S.mann_kendall(np.array([1.0, 2.0, 3.0]))
        assert mk["trend"] == "insufficient"

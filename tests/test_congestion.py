"""Congestion-model tests (curbiq.congestion): HCM capacity loss + BPR delay."""
from __future__ import annotations

import numpy as np
import pytest

from curbiq import config as C
from curbiq.congestion import bpr_delay_ratio, haversine_m, hcm_capacity_loss


# --------------------------------------------------------------------------- #
# HCM 2000 Ch.16 on-street-parking capacity loss
# --------------------------------------------------------------------------- #
class TestHcmCapacityLoss:
    def test_worked_example_n2_nm60(self):
        # fp = (2 - 0.1 - 18*60/3600) / 2 = (2 - 0.1 - 0.3)/2 = 0.80 -> loss 0.20
        loss = hcm_capacity_loss(np.array([60.0]), np.array([2]))
        assert loss[0] == pytest.approx(0.20, abs=0.01)

    def test_worked_example_n2_nm180(self):
        # fp = (2 - 0.1 - 18*180/3600)/2 = (2 - 0.1 - 0.9)/2 = 0.50 -> loss 0.50
        loss = hcm_capacity_loss(np.array([180.0]), np.array([2]))
        assert loss[0] == pytest.approx(0.50, abs=0.01)

    def test_zero_maneuvers_gives_tiny_loss(self):
        # fp = (2 - 0.1 - 0)/2 = 0.95 -> loss 0.05
        loss = hcm_capacity_loss(np.array([0.0]), np.array([2]))
        assert loss[0] == pytest.approx(0.05, abs=1e-6)

    def test_monotone_increasing_in_maneuver_rate(self):
        rates = np.array([0.0, 30.0, 60.0, 120.0, 180.0])
        loss = hcm_capacity_loss(rates, np.full(rates.shape, 2))
        assert np.all(np.diff(loss) >= 0)

    def test_decreasing_in_lanes(self):
        # More through-lanes absorb the same maneuver rate -> less proportional loss.
        nm = np.array([90.0, 90.0, 90.0])
        loss = hcm_capacity_loss(nm, np.array([1, 2, 3]))
        assert loss[0] >= loss[1] >= loss[2]

    def test_capacity_loss_capped_at_max(self):
        # N=1, Nm large -> fp floored, raw loss ~0.95 but capped to MAX_CAPACITY_LOSS.
        loss = hcm_capacity_loss(np.array([180.0, 1000.0]), np.array([1, 1]))
        assert np.all(loss <= C.MAX_CAPACITY_LOSS + 1e-9)
        assert loss[0] == pytest.approx(C.MAX_CAPACITY_LOSS)
        assert loss[1] == pytest.approx(C.MAX_CAPACITY_LOSS)

    def test_maneuver_rate_is_capped(self):
        # Nm > HCM_MANEUVER_RATE_CAP is clipped, so loss saturates.
        a = hcm_capacity_loss(np.array([C.HCM_MANEUVER_RATE_CAP]), np.array([2]))
        b = hcm_capacity_loss(np.array([C.HCM_MANEUVER_RATE_CAP * 5]), np.array([2]))
        assert a[0] == pytest.approx(b[0])

    def test_loss_never_negative(self):
        loss = hcm_capacity_loss(np.array([0.0, 1.0, 50.0]), np.array([3, 3, 3]))
        assert np.all(loss >= 0.0)


# --------------------------------------------------------------------------- #
# BPR delay ratio
# --------------------------------------------------------------------------- #
class TestBprDelayRatio:
    def test_no_loss_is_unit_ratio(self):
        assert bpr_delay_ratio(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_at_least_one(self):
        losses = np.linspace(0.0, C.MAX_CAPACITY_LOSS, 13)
        assert np.all(bpr_delay_ratio(losses) >= 1.0 - 1e-9)

    def test_monotone_increasing_in_loss(self):
        losses = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ratios = bpr_delay_ratio(losses)
        assert np.all(np.diff(ratios) > 0)

    def test_formula_at_half_loss(self):
        # vc_park = 0.8 / (1 - 0.5) = 1.6 ; base = 1 + .15*.8^4 ; park = 1 + .15*1.6^4
        base = 1.0 + C.BPR_ALPHA * C.BASELINE_VC_RATIO ** C.BPR_BETA
        park = 1.0 + C.BPR_ALPHA * (C.BASELINE_VC_RATIO / 0.5) ** C.BPR_BETA
        expected = park / base
        assert bpr_delay_ratio(np.array([0.5]))[0] == pytest.approx(expected)

    def test_vc_clamped(self):
        # An (impossible) loss near 1.0 would blow up v/c; the clamp keeps it finite.
        r = bpr_delay_ratio(np.array([0.99]))
        assert np.isfinite(r[0]) and r[0] > 1.0


# --------------------------------------------------------------------------- #
# Haversine geometry sanity
# --------------------------------------------------------------------------- #
class TestHaversine:
    def test_zero_distance(self):
        assert haversine_m(12.97, 77.59, 12.97, 77.59) == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_lat_approx_111km(self):
        d = haversine_m(12.0, 77.0, 13.0, 77.0)
        assert d == pytest.approx(111_000, rel=0.02)

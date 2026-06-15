"""Forecast-metric tests (curbiq.forecast): PAI/PEI and score_block."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from curbiq import config as C
from curbiq.forecast import pai_pei, score_block


def _single_day_frame(actual, pred, dt="2024-01-01"):
    n = len(actual)
    return pd.DataFrame({"dt": [dt] * n, "actual": actual, "pred": pred})


# --------------------------------------------------------------------------- #
# pai_pei analytic checks
# --------------------------------------------------------------------------- #
class TestPaiPei:
    def test_pai_at_20pct_perfect_ranking(self):
        # 10 cells; all violations in 2 cells; model ranks those 2 highest.
        # m = round(0.2*10) = 2 -> captures 100% -> PAI = 1.0 / 0.20 = 5.0.
        day = _single_day_frame(
            actual=[50, 50, 0, 0, 0, 0, 0, 0, 0, 0],
            pred=[9.0, 8.0, 1, 1, 1, 1, 1, 1, 1, 1])
        pai, pei = pai_pei(day, 0.20)
        assert pai == pytest.approx(5.0)
        assert pei == pytest.approx(1.0)   # model captures as much as the oracle

    def test_pai_at_5pct_top_cell_only(self):
        # m = max(1, round(0.05*10)) = 1 -> captures 50/100 = 0.5 -> PAI = 0.5/0.05 = 10.
        day = _single_day_frame(
            actual=[50, 50, 0, 0, 0, 0, 0, 0, 0, 0],
            pred=[9.0, 8.0, 1, 1, 1, 1, 1, 1, 1, 1])
        pai, _ = pai_pei(day, 0.05)
        assert pai == pytest.approx(10.0)

    def test_uniform_predictions_give_no_lift(self):
        # 20 cells uniform actual, model flat -> top-k captures exactly its area share.
        # nlargest is stable on ties (keeps first-seen order) -> capture == frac -> PAI 1.
        day = _single_day_frame(actual=[5] * 20, pred=[1.0] * 20)
        pai, _ = pai_pei(day, 0.20)
        assert pai == pytest.approx(1.0, abs=1e-9)

    def test_skips_zero_total_days(self):
        # A day with no violations is skipped; the remaining good day drives PAI.
        good = _single_day_frame([10, 0, 0, 0], [3.0, 1, 1, 1], dt="2024-01-02")
        empty = _single_day_frame([0, 0, 0, 0], [1.0, 1, 1, 1], dt="2024-01-03")
        df = pd.concat([good, empty], ignore_index=True)
        pai, _ = pai_pei(df, 0.25)   # m = max(1, round(.25*4)) = 1 -> top cell captures all
        assert pai == pytest.approx(4.0)

    def test_anti_ranking_below_one(self):
        # Model ranks the empty cells highest -> captures less than its area share.
        day = _single_day_frame(
            actual=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pred=[0.0, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        pai, pei = pai_pei(day, 0.20)
        assert pai < 1.0
        assert pei == pytest.approx(0.0)   # captured 0 of the oracle's hotspot mass


# --------------------------------------------------------------------------- #
# score_block keys + values
# --------------------------------------------------------------------------- #
class TestScoreBlock:
    def _frame(self):
        actual = list(np.r_[np.zeros(18), [10.0, 12.0]])
        pred = list(np.r_[np.full(18, 0.1), [9.0, 11.0]])
        return _single_day_frame(actual, pred)

    def test_required_keys_present(self):
        out = score_block(self._frame())
        for key in ("mae", "rmse", "r2", "poisson_deviance"):
            assert key in out
        for frac in C.PAI_AREA_FRACS:
            assert f"pai@{int(frac*100)}" in out
            assert f"pei@{int(frac*100)}" in out

    def test_classification_keys_present_when_mixed_labels(self):
        out = score_block(self._frame())
        # With a clear top-decile split present, AUC metrics are emitted.
        assert "roc_auc" in out
        assert "pr_auc" in out
        assert 0.0 <= out["roc_auc"] <= 1.0

    def test_perfect_prediction_metrics(self):
        a = [0.0, 0.0, 5.0, 10.0]
        df = _single_day_frame(a, a)
        out = score_block(df)
        assert out["mae"] == pytest.approx(0.0)
        assert out["rmse"] == pytest.approx(0.0)
        assert out["r2"] == pytest.approx(1.0)

    def test_no_auc_when_no_positive_labels(self):
        # All-zero actuals -> no hotspot label -> classification metrics omitted.
        df = _single_day_frame([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0])
        out = score_block(df)
        assert "roc_auc" not in out
        assert out["r2"] == 0.0   # zero variance guard

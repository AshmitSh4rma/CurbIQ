"""Tests for patrol-routing optimization (time matrix, 2-opt, VRP orchestration)."""
import numpy as np
import pandas as pd

from curbiq import patrol


def _cells(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "h3": [f"c{i}" for i in range(n)],
        "lat": 12.97 + rng.uniform(0, 0.03, n),
        "lon": 77.59 + rng.uniform(0, 0.03, n),
        "priority_score": rng.uniform(10, 100, n),
        "top_offence": ["NO PARKING"] * n,
    })


def test_time_matrix_dwell_and_diag():
    lat = np.array([12.97, 12.98])
    lon = np.array([77.59, 77.60])
    tm = patrol._time_matrix(lat, lon, speed_kmph=20, dwell_s=480)
    assert tm.shape == (2, 2)
    assert tm[0, 0] == 0 and tm[1, 1] == 0
    assert tm[0, 1] > 480          # travel + dwell at non-depot destination
    assert tm[1, 0] < tm[0, 1]     # arriving at depot (idx 0) carries no dwell


def test_two_opt_returns_permutation():
    tm = np.array([[0, 1, 10, 1], [1, 0, 1, 10], [10, 1, 0, 1], [1, 10, 1, 0]], float)
    r = patrol._two_opt([1, 2, 3], tm, depot=0)
    assert set(r) == {1, 2, 3}


def test_optimize_patrols_end_to_end():
    cells = _cells(10)
    r = patrol.optimize_patrols(cells, n_units=2, top_k=10, shift_hours=6)
    assert r["solver"] in ("ortools", "greedy")
    assert r["stops_covered"] >= 1
    # each covered stop belongs to exactly one route
    assert sum(rt["n_stops"] for rt in r["routes"]) == r["stops_covered"]
    assert 0 <= r["coverage_pct"] <= 100
    for rt in r["routes"]:
        for s in rt["stops"]:
            assert ":" in s["eta"] and "priority" in s

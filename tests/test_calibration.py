"""Tests for the congestion calibration / probe-speed validation module."""
import h3
import numpy as np
import pandas as pd

from curbiq import calibration as cal


def test_congestion_from_speeds():
    pct = cal.congestion_from_speeds([30, 60, 0], [60, 60, 60])
    assert abs(pct[0] - 50) < 1e-6
    assert abs(pct[1] - 0) < 1e-6
    assert abs(pct[2] - 100) < 1e-6


def _toy_cis(n=120, seed=0):
    rng = np.random.default_rng(seed)
    center = h3.latlng_to_cell(12.97, 77.59, 9)
    cells = list(h3.grid_disk(center, 10))[:n]   # distinct H3 cells (no collisions)
    n = len(cells)
    return pd.DataFrame({
        "h3": cells,
        "z_density": rng.normal(size=n), "z_junction": rng.normal(size=n),
        "z_peak": rng.normal(size=n), "z_road": rng.normal(size=n),
        "count": rng.integers(5, 500, n), "rc_loss": rng.uniform(0.15, 0.4, n),
        "junction_proximity": rng.uniform(0, 1, n), "peak_share": rng.uniform(0, 1, n),
    })


def test_synthetic_probe_bounds():
    df = _toy_cis()
    p = cal.synthetic_probe(df)
    assert len(p) == len(df)
    assert {"h3", "congestion_pct"}.issubset(p.columns)
    assert p["congestion_pct"].between(0, 100).all()


def test_calibrate_improves_and_weights_normalized():
    df = _toy_cis(150)
    # ground truth driven almost entirely by z_density -> density should dominate
    noise = np.random.default_rng(1).normal(0, 1, len(df))
    probe = pd.DataFrame({"h3": df["h3"], "congestion_pct": 50 + 10 * df["z_density"] + noise})
    r = cal.calibrate(df, probe)
    w = r["calibrated_weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert r["spearman_calibrated"] >= r["spearman_default"] - 1e-9
    # strong z_density signal -> calibration should reach high agreement
    assert r["spearman_calibrated"] > 0.7
    assert w["severity_density"] >= 0.30   # density up-weighted above the 0.25 uniform


def test_load_probe_csv(tmp_path):
    df = _toy_cis(40)
    csv = tmp_path / "probe.csv"
    pd.DataFrame({"h3": df["h3"], "observed_speed_kmph": 20,
                  "freeflow_speed_kmph": 40}).to_csv(csv, index=False)
    out = cal.load_probe_csv(str(csv))
    assert "congestion_pct" in out.columns
    assert abs(out["congestion_pct"].iloc[0] - 50) < 1e-6

"""Tests for hotspot validation against an enforcement-point reference set."""
import pandas as pd

from curbiq import validate_geo as vg


def test_btp_reference_from_data():
    df = pd.DataFrame({
        "id": [1, 2, 3], "is_counted": [True, True, True],
        "has_junction": [True, True, False], "junction_id": ["J1", "J1", "None"],
        "latitude": [12.98, 12.982, 12.90], "longitude": [77.60, 77.60, 77.70],
    })
    ref = vg.btp_reference_from_data(df)
    assert len(ref) == 1
    assert ref.iloc[0]["name"] == "J1"
    assert {"lat", "lon", "name"}.issubset(ref.columns)


def test_validate_precision_recall_and_novel():
    ref = pd.DataFrame({"name": ["J1"], "lat": [12.980], "lon": [77.600]})
    cells = pd.DataFrame({
        "h3": ["a", "b", "c"],
        "lat": [12.9803, 12.998, 12.970],     # a ~35 m, b ~2 km, c ~11 km from ref
        "lon": [77.6001, 77.600, 77.700],
        "priority_score": [100, 90, 80],
        "is_hotspot": [True, True, False],
        "count": [500, 400, 100],
        "top_offence": ["NO PARKING", "WRONG PARKING", "NO PARKING"],
    })
    r = vg.validate(cells, ref, top_ns=(2,), radii=(150, 300), novel_radius=500)
    # top-2 = {a,b}; only a is within 150 m -> precision 0.5
    assert r["precision_at_n"]["top2"]["150m"] == 0.5
    # b is a hotspot ~2 km from any reference -> flagged novel
    assert r["n_novel_hotspots"] >= 1
    assert any(nv["h3"] == "b" for nv in r["novel_hotspots"])
    # the reference junction has hotspot a within 300 m -> recall 1.0
    assert r["recall"]["300m"] == 1.0

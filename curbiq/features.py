"""Feature engineering for parking-violation records.

Pure, testable functions that enrich a cleaned DataFrame with the temporal,
spatial, road-class, vehicle-footprint, offence-severity and confidence
features used by every downstream module (hotspots, congestion, forecasting).
"""
from __future__ import annotations

import json
from functools import lru_cache

import h3
import numpy as np
import pandas as pd

from curbiq import config as C


# --------------------------------------------------------------------------- #
# Offence / violation parsing
# --------------------------------------------------------------------------- #
def parse_int_list(raw: object) -> list[int]:
    """Parse an offence_code cell like ``"[112,104]"`` into ``[112, 104]``."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if s in C.NULL_TOKENS:
        return []
    try:
        val = json.loads(s)
    except (ValueError, TypeError):
        return []
    out = []
    for x in val if isinstance(val, list) else [val]:
        try:
            out.append(int(x))
        except (ValueError, TypeError):
            continue
    return out


def parse_str_list(raw: object) -> list[str]:
    """Parse a violation_type cell like ``'["NO PARKING"]'`` into a str list."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if s in C.NULL_TOKENS:
        return []
    try:
        val = json.loads(s)
    except (ValueError, TypeError):
        return [s]
    return [str(x) for x in (val if isinstance(val, list) else [val])]


def primary_offence(codes: list[int]) -> int | None:
    """The parking offence with the highest carriageway severity, else first code."""
    if not codes:
        return None
    parking = [c for c in codes if c in C.PARKING_OFFENCE_CODES]
    if parking:
        return max(parking, key=lambda c: C.OFFENCE_CARRIAGEWAY_SEVERITY.get(c, C.DEFAULT_SEVERITY))
    return codes[0]


def max_carriageway_severity(codes: list[int]) -> float:
    parking = [C.OFFENCE_CARRIAGEWAY_SEVERITY.get(c, C.DEFAULT_SEVERITY)
               for c in codes if c in C.PARKING_OFFENCE_CODES]
    return max(parking) if parking else C.DEFAULT_SEVERITY


# --------------------------------------------------------------------------- #
# Road-class inference from free-text address
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=100_000)
def classify_road(location: str | None) -> tuple[str, float]:
    """Map a free-text address to (road_class, criticality_weight)."""
    if not location:
        return C.DEFAULT_ROAD_CLASS, C.DEFAULT_ROAD_WEIGHT
    for pattern, name, weight in C._COMPILED_ROAD_PATTERNS:
        if pattern.search(location):
            return name, weight
    return C.DEFAULT_ROAD_CLASS, C.DEFAULT_ROAD_WEIGHT


# --------------------------------------------------------------------------- #
# Temporal helpers
# --------------------------------------------------------------------------- #
def peak_overlap(hour: int, is_weekend: bool) -> float:
    """Congestion peak-overlap factor in [0, 1] for a given IST hour.

    Weekend peaks are softer than weekday commute peaks.
    """
    m0, m1 = C.MORNING_PEAK
    e0, e1 = C.EVENING_PEAK
    in_morning = m0 <= hour < m1
    in_evening = e0 <= hour < e1
    if in_morning or in_evening:
        return 0.7 if is_weekend else 1.0
    # shoulders of the peaks
    if (m0 - 1) <= hour <= (m1 + 1) or (e0 - 1) <= hour <= (e1 + 1):
        return 0.6
    # daytime inter-peak
    if 11 <= hour < 17:
        return 0.5
    # night / early morning
    return 0.25


def time_of_day_bucket(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 8:
        return "early_morning"
    if 8 <= hour < 11:
        return "morning_peak"
    if 11 <= hour < 17:
        return "midday"
    if 17 <= hour < 21:
        return "evening_peak"
    return "late_evening"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered feature columns to a cleaned DataFrame (in place-ish)."""
    df = df.copy()

    # --- offences ---------------------------------------------------------
    codes = df["offence_code"].map(parse_int_list)
    df["offence_codes"] = codes
    df["violation_types"] = df["violation_type"].map(parse_str_list)
    df["n_offences"] = codes.map(len)
    df["primary_code"] = codes.map(primary_offence)
    df["primary_offence"] = df["primary_code"].map(
        lambda c: C.OFFENCE_LABELS.get(c) if c is not None else None)
    df["is_parking"] = codes.map(lambda cs: any(c in C.PARKING_OFFENCE_CODES for c in cs))
    df["n_parking_offences"] = codes.map(
        lambda cs: sum(1 for c in cs if c in C.PARKING_OFFENCE_CODES))
    df["carriageway_severity"] = codes.map(max_carriageway_severity)

    # --- vehicle ----------------------------------------------------------
    vt = df["vehicle_type"].fillna("OTHERS").astype(str).str.upper().str.strip()
    df["vehicle_type"] = vt
    df["footprint_pcu"] = vt.map(C.VEHICLE_FOOTPRINT_PCU).fillna(C.DEFAULT_FOOTPRINT_PCU)
    df["vehicle_category"] = vt.map(C.VEHICLE_CATEGORY).fillna("other")

    # obstruction magnitude = footprint x how much the offence blocks a live lane
    df["blockage"] = df["footprint_pcu"] * df["carriageway_severity"]

    # --- road class -------------------------------------------------------
    loc = df["location"].astype("object").where(df["location"].notna(), None)
    road = loc.map(classify_road)
    df["road_class"] = road.map(lambda t: t[0])
    df["road_weight"] = road.map(lambda t: t[1]).astype(float)

    # --- junction ---------------------------------------------------------
    jn = df["junction_name"].fillna("No Junction").astype(str).str.strip()
    df["has_junction"] = (jn != "No Junction") & (jn != "")
    df["junction_id"] = jn.where(df["has_junction"], None)

    # --- validation confidence -------------------------------------------
    vs = df["validation_status"].fillna("NULL").astype(str)
    df["validation_status"] = vs
    df["confidence"] = vs.map(C.VALIDATION_CONFIDENCE).fillna(C.DEFAULT_CONFIDENCE)
    df["is_counted"] = vs != "duplicate"

    # --- temporal (IST) ---------------------------------------------------
    ist = df["created_ist"]
    df["date"] = ist.dt.date.astype("string")
    df["hour"] = ist.dt.hour.astype("int16")
    df["dow"] = ist.dt.dayofweek.astype("int16")          # 0 = Monday
    df["is_weekend"] = df["dow"] >= 5
    df["month"] = ist.dt.strftime("%Y-%m")
    df["week"] = ist.dt.strftime("%G-W%V")
    df["peak_overlap"] = [
        peak_overlap(int(h), bool(w)) for h, w in zip(df["hour"], df["is_weekend"])
    ]
    df["tod_bucket"] = df["hour"].map(time_of_day_bucket)

    # --- H3 spatial indexing ---------------------------------------------
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    df["h3_r9"] = [h3.latlng_to_cell(la, lo, C.H3_RES_PRIMARY) for la, lo in zip(lat, lon)]
    df["h3_r10"] = [h3.latlng_to_cell(la, lo, C.H3_RES_FINE) for la, lo in zip(lat, lon)]
    df["h3_r8"] = [h3.latlng_to_cell(la, lo, C.H3_RES_COARSE) for la, lo in zip(lat, lon)]

    return df

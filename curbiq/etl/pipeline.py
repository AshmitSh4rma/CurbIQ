"""Raw CSV -> cleaned, feature-engineered parquet.

``build_processed()`` is the one-time (idempotent) transform; ``load_processed()``
is the fast read used by analytics and the API.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from curbiq import config as C
from curbiq.features import engineer

# Columns we actually read (the rest are 100% null or redundant).
USECOLS = [
    "id", "latitude", "longitude", "location",
    "vehicle_number", "vehicle_type",
    "violation_type", "offence_code",
    "created_datetime", "modified_datetime",
    "device_id", "created_by_id",
    "center_code", "police_station", "junction_name",
    "data_sent_to_scita",
    "updated_vehicle_number", "updated_vehicle_type",
    "validation_status",
]


def _nullify(s: pd.Series) -> pd.Series:
    """Replace the dataset's textual null sentinels with real NA."""
    return s.astype("object").where(~s.astype("object").isin(C.NULL_TOKENS), other=pd.NA)


def build_processed(raw_csv=C.RAW_CSV, out=C.PROCESSED_PARQUET, verbose=True) -> pd.DataFrame:
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[etl] {msg}")

    log(f"reading {raw_csv} ...")
    df = pd.read_csv(
        raw_csv,
        usecols=USECOLS,
        dtype="string",
        engine="c",
        on_bad_lines="warn",
    )
    log(f"raw rows: {len(df):,}")

    # --- coerce geo & drop out-of-bounds ---------------------------------
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    in_bbox = (
        df["latitude"].between(C.BBOX["lat_min"], C.BBOX["lat_max"])
        & df["longitude"].between(C.BBOX["lon_min"], C.BBOX["lon_max"])
    )
    dropped = int((~in_bbox).sum())
    df = df[in_bbox].copy()
    log(f"dropped {dropped:,} out-of-bbox / bad-geo rows -> {len(df):,}")

    # --- nullify sentinels on text columns -------------------------------
    for col in ("location", "vehicle_type", "updated_vehicle_type",
                "vehicle_number", "updated_vehicle_number",
                "junction_name", "police_station", "validation_status",
                "center_code"):
        df[col] = _nullify(df[col])

    # --- coalesce corrected vehicle fields over originals ----------------
    df["vehicle_type"] = df["updated_vehicle_type"].fillna(df["vehicle_type"])
    df["vehicle_number"] = df["updated_vehicle_number"].fillna(df["vehicle_number"])
    df = df.drop(columns=["updated_vehicle_type", "updated_vehicle_number"])

    # --- timestamps: parse UTC, convert to IST ---------------------------
    created = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    n_bad_dt = int(created.isna().sum())
    df = df[created.notna()].copy()
    created = created[created.notna()]
    # store UTC (tz-aware) and IST wall-clock (tz-naive, so .dt.hour gives IST).
    # NB: never use .values here -- it strips tz and silently reverts to UTC.
    df["created_utc"] = created
    df["created_ist"] = created.dt.tz_convert(C.LOCAL_TZ).dt.tz_localize(None)
    log(f"dropped {n_bad_dt:,} rows with unparseable created_datetime -> {len(df):,}")

    # normalise scita flag
    df["data_sent_to_scita"] = df["data_sent_to_scita"].astype("string").str.upper().eq("TRUE")

    log("engineering features ...")
    df = engineer(df)

    # --- final column selection ------------------------------------------
    keep = [
        "id", "latitude", "longitude", "location",
        "vehicle_number", "vehicle_type", "vehicle_category", "footprint_pcu",
        "violation_types", "offence_codes", "primary_code", "primary_offence",
        "is_parking", "n_offences", "n_parking_offences", "carriageway_severity",
        "blockage",
        "road_class", "road_weight",
        "police_station", "center_code", "junction_id", "has_junction",
        "validation_status", "confidence", "is_counted", "data_sent_to_scita",
        "device_id", "created_by_id",
        "created_utc", "created_ist",
        "date", "hour", "dow", "is_weekend", "month", "week",
        "peak_overlap", "tod_bucket",
        "h3_r8", "h3_r9", "h3_r10",
    ]
    df = df[keep]

    # list/object columns must be python lists for parquet round-trip
    df["offence_codes"] = df["offence_codes"].map(list)
    df["violation_types"] = df["violation_types"].map(list)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    size_mb = out.stat().st_size / 1e6
    log(f"wrote {out} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    log(f"parking-relevant rows: {int(df['is_parking'].sum()):,} / {len(df):,}")
    return df


def load_processed(out=C.PROCESSED_PARQUET, build_if_missing=True) -> pd.DataFrame:
    if not out.exists():
        if build_if_missing:
            return build_processed(out=out)
        raise FileNotFoundError(f"{out} not found; run `python -m curbiq.etl.pipeline`")
    return pd.read_parquet(out, engine="pyarrow")


if __name__ == "__main__":
    build_processed()

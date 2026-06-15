"""End-to-end CurbIQ build: ETL -> analytics -> versioned artifacts + model.

Usage:
    python build_all.py                # build artifacts (ETL only if parquet missing)
    python build_all.py --rebuild-etl  # force re-run the ETL from the raw CSV
"""
from __future__ import annotations

import argparse
import time

from curbiq import config as C
from curbiq.artifacts import build_artifacts
from curbiq.etl import build_processed, load_processed


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the full CurbIQ pipeline.")
    ap.add_argument("--rebuild-etl", action="store_true",
                    help="force re-run the ETL transform from the raw CSV")
    ap.add_argument("--probe", default=None,
                    help="path to a probe-speed CSV to calibrate the congestion score "
                         "against (TomTom/Google/Uber Movement); else a synthetic probe is used")
    ap.add_argument("--enforcement-points", default=None,
                    help="path to the official BTP enforcement-point CSV (name,lat,lon) to "
                         "validate hotspots against; else the dataset's BTP junctions are used")
    args = ap.parse_args()

    t0 = time.time()
    if args.rebuild_etl or not C.PROCESSED_PARQUET.exists():
        df = build_processed()
    else:
        df = load_processed()
        print(f"[build] loaded processed parquet: {len(df):,} rows")

    build_artifacts(df, probe_path=args.probe, enforcement_points=args.enforcement_points)
    print(f"[build] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

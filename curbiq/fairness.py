"""Fairness, bias and privacy layer.

Recorded violations are *enforcement events*, not ground-truth occurrence, so a
naive system launders patrol bias into "where the problems are". This module
makes that bias visible and ships DPDP-Act-2023-compliant privacy helpers.

* ``temporal_gap``  — enforcement share vs congestion-risk share by IST hour;
  exposes the evening blind spot (peak congestion, near-zero enforcement).
* ``spatial_equity``— per-station enforcement coverage relative to modeled need,
  with 4/5ths disparate-impact and statistical-parity flags.
* privacy helpers   — SHA-256+salt plate hashing and k-anonymity suppression.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from curbiq import config as C
from curbiq.features import peak_overlap


# --------------------------------------------------------------------------- #
# Privacy (India DPDP Act 2023)
# --------------------------------------------------------------------------- #
def hash_plate(plate: str, salt: str = C.PLATE_SALT) -> str:
    return hashlib.sha256(f"{salt}:{plate}".encode()).hexdigest()[:16]


def k_anon_suppress(frame: pd.DataFrame, count_col: str = "count",
                    k: int = C.K_ANON_MIN) -> tuple[pd.DataFrame, dict]:
    """Drop any public cell/bucket with count < k (k-anonymity)."""
    keep = frame[count_col] >= k
    report = {
        "k": k,
        "n_total": int(len(frame)),
        "n_suppressed": int((~keep).sum()),
        "frac_suppressed": float((~keep).mean()) if len(frame) else 0.0,
    }
    return frame[keep].copy(), report


# --------------------------------------------------------------------------- #
# Temporal under-enforcement
# --------------------------------------------------------------------------- #
def temporal_gap(df: pd.DataFrame) -> dict:
    d = df[df["is_counted"]]
    enf = d.groupby("hour", observed=True).size().reindex(range(24), fill_value=0)
    enf_share = enf / enf.sum()
    risk = pd.Series({h: peak_overlap(h, False) for h in range(24)})
    risk_share = risk / risk.sum()
    gap = (risk_share - enf_share)               # +ve => under-enforced for its risk
    return {
        "hour": list(range(24)),
        "enforcement_share": enf_share.round(4).tolist(),
        "risk_share": risk_share.round(4).tolist(),
        "under_enforcement_gap": gap.round(4).tolist(),
        "most_under_enforced_hours": gap.sort_values(ascending=False).head(4).index.tolist(),
        "evening_peak_enforcement_share": float(enf_share.reindex(range(17, 21)).sum()),
    }


# --------------------------------------------------------------------------- #
# Spatial equity across police stations
# --------------------------------------------------------------------------- #
def spatial_equity(df: pd.DataFrame, min_records: int = 200) -> dict:
    d = df[df["is_counted"]]
    g = d.groupby("police_station", observed=True).agg(
        observed=("id", "size"),
        need=("blockage", "sum"),          # severity-weighted density = modeled need
    )
    g = g[g["observed"] >= min_records]
    g["observed_share"] = g["observed"] / g["observed"].sum()
    g["need_share"] = g["need"] / g["need"].sum()
    g["coverage_ratio"] = g["observed_share"] / g["need_share"]   # >1 over-, <1 under-enforced

    cr = g["coverage_ratio"]
    di = float(cr.min() / cr.max()) if cr.max() > 0 else 0.0
    parity = float((g["observed_share"] - g["need_share"]).abs().max())
    g = g.sort_values("coverage_ratio")
    return {
        "n_stations": int(len(g)),
        "disparate_impact_ratio": di,
        "disparate_impact_flag": bool(di < C.DISPARATE_IMPACT_FLAG),
        "statistical_parity_diff": parity,
        "statistical_parity_flag": bool(parity > C.STAT_PARITY_FLAG),
        "most_under_enforced": g.head(5).reset_index()[
            ["police_station", "observed", "coverage_ratio"]].round(3).to_dict("records"),
        "most_over_enforced": g.tail(5).reset_index()[
            ["police_station", "observed", "coverage_ratio"]].round(3).to_dict("records"),
    }


# --------------------------------------------------------------------------- #
# Data-quality guardrails
# --------------------------------------------------------------------------- #
def data_quality(df: pd.DataFrame) -> dict:
    vs = df["validation_status"].value_counts(normalize=True).round(4).to_dict()
    rej = df["validation_status"].eq("rejected").mean()
    scita = df["data_sent_to_scita"].mean()
    return {
        "validation_status_distribution": vs,
        "rejected_fraction": float(rej),
        "scita_true_fraction": float(scita),
        # guard against constant-flag false signals (the all-TRUE trap)
        "scita_is_near_constant": bool(scita > 0.95 or scita < 0.05),
        "note": "rejected tickets down-weighted (confidence) and duplicates excluded from counts",
    }


def run_fairness(df: pd.DataFrame) -> dict:
    return {
        "temporal": temporal_gap(df),
        "spatial_equity": spatial_equity(df),
        "data_quality": data_quality(df),
        "privacy_policy": {
            "public_granularity": "H3 cell ids / centroids only — never raw point lat/lon",
            "k_anonymity": C.K_ANON_MIN,
            "plate_handling": "SHA-256 + salt" if C.HASH_PLATES else "dropped",
            "named_junction_exception": ("BTP-named junction centroids are published as "
                                         "operational geography (already-public junction "
                                         "labels averaged over >=425 points) — not personal data"),
            "regulation": "India DPDP Act 2023",
        },
    }


if __name__ == "__main__":
    from curbiq.etl import load_processed

    df = load_processed()
    r = run_fairness(df)
    t = r["temporal"]
    print("== temporal under-enforcement ==")
    print(f"  evening-peak (17-20h) enforcement share: {t['evening_peak_enforcement_share']:.4f}")
    print(f"  most under-enforced hours (IST): {t['most_under_enforced_hours']}")
    se = r["spatial_equity"]
    print("\n== spatial equity ==")
    print(f"  stations: {se['n_stations']}  DI ratio: {se['disparate_impact_ratio']:.3f} "
          f"(flag={se['disparate_impact_flag']})  parity diff: {se['statistical_parity_diff']:.3f}")
    print(f"  most under-enforced vs need: {[s['police_station'] for s in se['most_under_enforced']]}")
    print(f"  most over-enforced vs need:  {[s['police_station'] for s in se['most_over_enforced']]}")
    print("\n== data quality ==")
    print(f"  {r['data_quality']['validation_status_distribution']}")

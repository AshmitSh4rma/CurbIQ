"""External feature enrichment: weather (Open-Meteo) + metro proximity (Overpass).

Both are free, key-less sources. Results are cached under ``data/external/`` so
the network is hit at most once. If a source is unreachable the pipeline
degrades gracefully (weather -> NaN, metro -> small built-in fallback list).
"""
from __future__ import annotations

import httpx
import pandas as pd

from curbiq import config as C

EXTERNAL = C.DATA_DIR / "external"
WEATHER_CSV = EXTERNAL / "weather.csv"
METRO_CSV = EXTERNAL / "metro_stations.csv"

# Minimal fallback if Overpass is unreachable (major Namma Metro interchanges/stops).
_FALLBACK_METRO = [
    ("Nadaprabhu Kempegowda (Majestic)", 12.9756, 77.5726), ("MG Road", 12.9756, 77.6197),
    ("Cubbon Park", 12.9796, 77.5969), ("Vidhana Soudha", 12.9794, 77.5905),
    ("Indiranagar", 12.9784, 77.6386), ("Baiyappanahalli", 12.9905, 77.6536),
    ("Trinity", 12.9728, 77.6202), ("Halasuru", 12.9760, 77.6266),
    ("Rashtreeya Vidyalaya Road", 12.9215, 77.5800), ("Jayanagar", 12.9300, 77.5800),
    ("Jayadeva Hospital", 12.9167, 77.6000), ("Yeshwanthpur", 13.0234, 77.5500),
    ("Peenya", 13.0290, 77.5190), ("Nagasandra", 13.0480, 77.5000),
    ("Banashankari", 12.9250, 77.5730), ("Silk Institute", 12.8600, 77.5300),
    ("Whitefield (Kadugodi)", 12.9957, 77.7579), ("KR Pura", 12.9990, 77.6780),
    ("Electronic City", 12.8450, 77.6600), ("Bommasandra", 12.8100, 77.6700),
]


def fetch_weather(start: str, end: str, lat: float = 12.97, lon: float = 77.59) -> pd.DataFrame:
    r = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Kolkata"}, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": d["time"], "temp_max": d["temperature_2m_max"],
                       "temp_min": d["temperature_2m_min"], "precip": d["precipitation_sum"]})
    df["is_rainy"] = (df["precip"].fillna(0) >= 2.5).astype(int)
    return df


def fetch_metro() -> pd.DataFrame:
    q = ('[out:json][timeout:40];area["name"="Bengaluru"]->.a;'
         'node["station"="subway"](area.a);out;')
    r = httpx.get("https://overpass-api.de/api/interpreter", params={"data": q},
                  headers={"User-Agent": "CurbIQ/1.0 (traffic-analytics)"}, timeout=80)
    r.raise_for_status()
    rows = [{"name": e.get("tags", {}).get("name", "metro"), "lat": e["lat"], "lon": e["lon"]}
            for e in r.json().get("elements", []) if "lat" in e]
    return pd.DataFrame(rows)


def load_or_fetch_weather(start: str, end: str, lat=12.97, lon=77.59) -> pd.DataFrame | None:
    if WEATHER_CSV.exists():
        return pd.read_csv(WEATHER_CSV)
    try:
        df = fetch_weather(start, end, lat, lon)
    except Exception as e:                       # offline -> skip weather features
        print(f"[enrich] weather fetch failed ({e}); skipping weather features")
        return None
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(WEATHER_CSV, index=False)
    return df


def load_or_fetch_metro() -> pd.DataFrame:
    if METRO_CSV.exists():
        return pd.read_csv(METRO_CSV)
    try:
        df = fetch_metro()
        assert len(df) >= 10
        source = "overpass"
    except Exception as e:
        print(f"[enrich] Overpass fetch failed ({e}); using built-in metro fallback")
        df = pd.DataFrame(_FALLBACK_METRO, columns=["name", "lat", "lon"])
        source = "fallback"
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRO_CSV, index=False)
    print(f"[enrich] metro stations: {len(df)} ({source})")
    return df


if __name__ == "__main__":
    from curbiq.etl import load_processed

    df = load_processed()
    start = df["created_ist"].min().date().isoformat()
    end = df["created_ist"].max().date().isoformat()
    w = load_or_fetch_weather(start, end)
    m = load_or_fetch_metro()
    print(f"weather rows: {0 if w is None else len(w)}  ({start} -> {end})")
    if w is not None:
        print(f"  temp_max {w['temp_max'].min():.1f}-{w['temp_max'].max():.1f}C, "
              f"rainy days: {int(w['is_rainy'].sum())}/{len(w)}")
    print(f"metro stations: {len(m)}")

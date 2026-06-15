"""CurbIQ API — a read-only server over precomputed artifacts.

All heavy spatial/ML compute happens offline in ``curbiq.artifacts``; this layer
only loads the resulting JSON into memory at startup and serves immutable slices
with strong ETag + Cache-Control headers (CDN-friendly, 304-aware). No spatial
math ever runs in a request handler.

Run:
    uvicorn curbiq.api.main:app --reload --port 8000
"""
from __future__ import annotations

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from curbiq import config as C

# artifact name -> file (the public API surface)
ARTIFACTS = {
    "manifest": "manifest.json",
    "kpis": "kpis.json",
    "cells": "cells.json",
    "forecast": "forecast_cells.json",
    "zones": "zones.json",
    "junctions": "junctions.json",
    "emerging": "emerging.json",
    "priority": "priority.json",
    "fairness": "fairness.json",
    "calibration": "calibration.json",
    "geo-validation": "geo_validation.json",
    "patrol": "patrol.json",
    "timeseries": "timeseries.json",
    "weekly": "weekly.json",
    "model-metrics": "model_metrics.json",
}

_CACHE: dict[str, tuple[bytes, str]] = {}


def _load_artifacts(outdir: Path = C.ARTIFACTS_DIR) -> None:
    _CACHE.clear()
    for name, fname in ARTIFACTS.items():
        path = outdir / fname
        if path.exists():
            raw = path.read_bytes()
            etag = '"%s"' % hashlib.md5(raw).hexdigest()
            _CACHE[name] = (raw, etag)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    missing = [n for n in ARTIFACTS if n not in _CACHE]
    if missing:
        print(f"[api] WARNING missing artifacts {missing} — run `python build_all.py`")
    else:
        print(f"[api] loaded {len(_CACHE)} artifacts from {C.ARTIFACTS_DIR}")
    yield
    _CACHE.clear()


app = FastAPI(
    title="CurbIQ API",
    version="1.0.0",
    description="Illegal-parking & congestion-impact intelligence (read-only artifact server).",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "HEAD", "OPTIONS"], allow_headers=["*"])

# Optional API-key gate: if CURBIQ_API_KEY is set, /api/* requires X-API-Key.
API_KEY = os.environ.get("CURBIQ_API_KEY")


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/api/"):
        if request.headers.get("x-api-key") != API_KEY:
            return Response(b'{"error":"unauthorized"}', status_code=401,
                            media_type="application/json")
    return await call_next(request)


def _serve(name: str, request: Request) -> Response:
    entry = _CACHE.get(name)
    if entry is None:
        return Response(content=b'{"error":"artifact not built"}', status_code=503,
                        media_type="application/json")
    raw, etag = entry
    headers = {"ETag": etag, "Cache-Control": "public, max-age=300"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=raw, media_type="application/json", headers=headers)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts": sorted(_CACHE.keys()),
            "auth": "api-key" if API_KEY else "open"}


@app.get("/version")
def version() -> Response:
    entry = _CACHE.get("manifest")
    body = entry[0] if entry else b'{"name":"CurbIQ","version":"unknown"}'
    return Response(content=body, media_type="application/json")


@app.get("/api/{name}")
def api(name: str, request: Request) -> Response:
    if name not in ARTIFACTS:
        return Response(
            content=b'{"error":"unknown artifact","available":%s}'
            % str(sorted(ARTIFACTS)).replace("'", '"').encode(),
            status_code=404, media_type="application/json")
    return _serve(name, request)


# Serve the build-free dashboard last so /api/* and /health take precedence.
if C.WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(C.WEB_DIR), html=True), name="web")

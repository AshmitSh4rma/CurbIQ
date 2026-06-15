#!/usr/bin/env bash
# CurbIQ one-shot launcher.
# Sets up the virtualenv, installs deps, fetches the dataset, builds the
# analytics artifacts (only if missing), and serves the dashboard.
#
#   ./run.sh                 # set up everything and open the dashboard
#   ./run.sh --rebuild       # force-rebuild artifacts from the raw CSV
#   ./run.sh --port 9000      # serve on a different port
#   ./run.sh --no-open        # don't try to open a browser
#   ./run.sh --reinstall      # reinstall Python dependencies
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
REBUILD=0; OPEN=1; REINSTALL=0; WITH_CV=0
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
DATASET_URL="https://uc.hackerearth.com/he-public-ap-south-1/jan%20to%20may%20police%20violation_anonymized791b166.csv"

usage() {
  echo "Usage: ./run.sh [--rebuild] [--reinstall] [--with-cv] [--no-open] [--port N] [--host H]"; exit 0;
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)   REBUILD=1 ;;
    --reinstall) REINSTALL=1 ;;
    --with-cv)   WITH_CV=1 ;;
    --no-open)   OPEN=0 ;;
    --port)      PORT="${2:?}"; shift ;;
    --host)      HOST="${2:?}"; shift ;;
    -h|--help)   usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
  shift
done

log()  { printf '\033[36m[curbiq]\033[0m %s\n' "$*"; }
fail() { printf '\033[31m[curbiq] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1) virtualenv ------------------------------------------------------------
if [[ ! -x "$PY" ]]; then
  log "creating virtualenv at .venv ..."
  PYBIN="$(command -v python3.13 || command -v python3 || command -v python || true)"
  [[ -n "$PYBIN" ]] || fail "no python3 found on PATH"
  "$PYBIN" -m venv "$VENV"
fi

# 2) dependencies ----------------------------------------------------------
if [[ "$REINSTALL" == 1 ]] || ! "$PY" -c 'import fastapi, uvicorn, lightgbm, h3, pandas, scipy, sklearn' 2>/dev/null; then
  log "installing dependencies (first run can take a few minutes) ..."
  "$PY" -m pip install -q --no-cache-dir --upgrade pip
  "$PY" -m pip install -q --no-cache-dir -r "$ROOT/requirements.txt" || fail "pip install failed"
fi

# 2b) optional live-CV extras (onnxruntime + SSD-MobileNet model) ----------
if [[ "$WITH_CV" == 1 ]]; then
  log "installing onnxruntime + fetching SSD-MobileNet model for live CV ..."
  "$PY" -m pip install -q --no-cache-dir onnxruntime || fail "onnxruntime install failed"
  bash "$ROOT/scripts/get_cv_model.sh"
fi

# 3) dataset ---------------------------------------------------------------
RAW_GZ="$ROOT/data/raw/police_violations.csv.gz"
RAW_CSV="$ROOT/data/raw/police_violations.csv"
if [[ ! -f "$RAW_GZ" && ! -f "$RAW_CSV" ]]; then
  log "raw dataset not found — downloading (~105 MB) ..."
  mkdir -p "$ROOT/data/raw"
  if curl -fSL "$DATASET_URL" -o "$RAW_CSV"; then
    gzip -f "$RAW_CSV"   # config expects police_violations.csv.gz (pandas reads it transparently)
    log "dataset saved -> $RAW_GZ"
  else
    fail "dataset download failed. Put the CSV at $RAW_CSV (or .csv.gz) and re-run."
  fi
fi

# 4) build artifacts -------------------------------------------------------
export PYTHONPATH="$ROOT"
if [[ "$REBUILD" == 1 || ! -f "$ROOT/data/artifacts/manifest.json" ]]; then
  log "building artifacts: ETL -> hotspots/congestion/forecast/prioritize -> JSON + model ..."
  "$PY" build_all.py $( [[ "$REBUILD" == 1 ]] && echo --rebuild-etl )
else
  log "artifacts already built (use --rebuild to regenerate)"
fi

# 5) serve -----------------------------------------------------------------
URL="http://${HOST}:${PORT}"
log "dashboard -> ${URL}   (Ctrl-C to stop)"
if [[ "$OPEN" == 1 ]]; then
  ( sleep 2
    { command -v xdg-open >/dev/null && xdg-open "$URL"; } \
      || { command -v open >/dev/null && open "$URL"; } || true
  ) >/dev/null 2>&1 &
fi
exec "$PY" -m uvicorn curbiq.api.main:app --host "$HOST" --port "$PORT"

#!/usr/bin/env bash
# Scheduled artifact refresh — rebuild from the (possibly updated) raw dataset.
# Cron example (nightly at 02:00):
#   0 2 * * * /home/ashmit/Claude/CurbIQ/scripts/refresh.sh
#
# The API serves whatever is in data/artifacts/; because build_artifacts() is
# idempotent and versioned, the running server picks up the new manifest on its
# next restart (or wire a reload hook). No API/frontend contract changes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/data/refresh.log"
{
  echo "[refresh] start $(date -u +%FT%TZ)"
  PYTHONPATH="$ROOT" "$ROOT/.venv/bin/python" "$ROOT/build_all.py" --rebuild-etl
  echo "[refresh] done  $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1

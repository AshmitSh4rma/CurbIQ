# CurbIQ — developer & ops tasks.
# All Python runs through the project venv with PYTHONPATH set to the repo root,
# matching the documented run convention.

# --- configuration ---------------------------------------------------------
VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
PYTHONPATH := .
RUN        := PYTHONPATH=$(PYTHONPATH) $(PYTHON)
HOST       ?= 0.0.0.0
PORT       ?= 8000
IMAGE      := curbiq:1.0.0

export PYTHONPATH

.DEFAULT_GOAL := help
.PHONY: help venv install etl build serve test fmt clean docker-build docker-up

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## create the Python 3.13 virtualenv
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv  ## install pinned dependencies into the venv
	$(PIP) install -r requirements-lock.txt

etl:  ## force re-run the ETL from the raw CSV -> data/processed/violations.parquet
	$(RUN) build_all.py --rebuild-etl

build:  ## run the full pipeline: ETL (if needed) -> analytics -> artifacts + model
	$(RUN) build_all.py

serve:  ## run the read-only API + dashboard at http://localhost:$(PORT)
	PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/uvicorn curbiq.api.main:app --host $(HOST) --port $(PORT)

test:  ## run the test suite
	$(RUN) -m pytest

fmt:  ## format + lint-fix with ruff (no-op if ruff is not installed)
	@$(RUN) -m ruff format . 2>/dev/null && $(RUN) -m ruff check --fix . 2>/dev/null \
	  || echo "ruff not installed — skipping (pip install ruff)"

clean:  ## remove caches, bytecode, and generated artifacts/processed data
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[co]' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info build dist
	rm -f data/processed/*.parquet data/artifacts/*.json models/*.txt models/feature_cols.json

docker-build:  ## build the production Docker image
	docker build -t $(IMAGE) .

docker-up:  ## build artifacts (builder) then serve the API via docker compose
	docker compose up --build

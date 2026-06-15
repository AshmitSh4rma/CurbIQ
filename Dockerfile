# CurbIQ — lean production image
# Build:  docker build -t curbiq:1.0.0 .
# Run:    docker run --rm -p 8000:8000 curbiq:1.0.0
#
# The image ships the code + the lean Python runtime only. The processed
# parquet, JSON artifacts and the LightGBM model are NOT baked in (they are
# large and regenerable). Provide them in ONE of two ways:
#
#   (a) build at run time inside the container
#       docker run --rm -v "$PWD/data:/app/data" curbiq:1.0.0 \
#           python build_all.py
#       (the artifacts then land in the mounted data/ volume; rerun without
#        the override to serve the API)
#
#   (b) mount pre-built artifacts from the host
#       docker run --rm -p 8000:8000 \
#           -v "$PWD/data/artifacts:/app/data/artifacts:ro" \
#           -v "$PWD/models:/app/models:ro" \
#           curbiq:1.0.0
#
# See docker-compose.yml for the builder + api split that automates (a)->(b).

FROM python:3.13-slim AS base

# Python: no .pyc, unbuffered logs, predictable, no pip version chatter.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Build toolchain is only needed if a wheel is missing for the target platform.
# All listed deps publish manylinux wheels for cp313, so this is normally a
# no-op; kept minimal and removed in the same layer to keep the image lean.
# If a future dep needs to compile, build-essential is here; otherwise the apt
# block can be dropped entirely.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# libgomp1: OpenMP runtime required by LightGBM and scikit-learn at run time.

# Dependencies first for layer caching: the lockfile changes rarely.
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# Project source (data/raw, data/processed, data/artifacts, models are excluded
# via .dockerignore and supplied as volumes / built at run time).
COPY . .

# Run as an unprivileged user; pre-create the writable data + model dirs so a
# run-time build (option (a)) can write even when nothing is mounted.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/artifacts /app/models \
    && useradd --create-home --uid 10001 curbiq \
    && chown -R curbiq:curbiq /app
USER curbiq

EXPOSE 8000

# Lightweight liveness probe against the API's /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)" || exit 1

CMD ["uvicorn", "curbiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

"""API tests (curbiq.api.main) using FastAPI's TestClient.

These hit the read-only artifact server against the already-built artifacts in
data/artifacts/*.json — they never rebuild anything.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from curbiq.api.main import ARTIFACTS, app


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan handler that loads artifacts.
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["artifacts"], list)

    def test_health_lists_loaded_artifacts(self, client):
        body = client.get("/health").json()
        # kpis should be present given artifacts are built in this repo.
        assert "kpis" in body["artifacts"]


class TestKpis:
    def test_kpis_200_with_etag_and_cache_control(self, client):
        r = client.get("/api/kpis")
        assert r.status_code == 200
        assert "etag" in r.headers
        assert r.headers["cache-control"] == "public, max-age=300"
        # body is valid JSON
        json.loads(r.content)

    def test_etag_matches_on_repeat(self, client):
        e1 = client.get("/api/kpis").headers["etag"]
        e2 = client.get("/api/kpis").headers["etag"]
        assert e1 == e2

    def test_304_on_matching_if_none_match(self, client):
        etag = client.get("/api/kpis").headers["etag"]
        r = client.get("/api/kpis", headers={"If-None-Match": etag})
        assert r.status_code == 304
        assert r.content == b""
        # 304 must still carry the cache headers
        assert r.headers["etag"] == etag
        assert r.headers["cache-control"] == "public, max-age=300"

    def test_200_on_stale_if_none_match(self, client):
        r = client.get("/api/kpis", headers={"If-None-Match": '"stale-etag"'})
        assert r.status_code == 200


class TestUnknownArtifact:
    def test_unknown_artifact_404(self, client):
        r = client.get("/api/does-not-exist")
        assert r.status_code == 404
        body = r.json()
        assert body["error"] == "unknown artifact"
        assert isinstance(body["available"], list)


class TestAllKnownArtifacts:
    @pytest.mark.parametrize("name", sorted(ARTIFACTS))
    def test_each_known_artifact_served(self, client, name):
        r = client.get(f"/api/{name}")
        # Built repo -> 200; if an artifact were missing the server returns 503,
        # never a 404/500. Accept both documented states but require valid JSON.
        assert r.status_code in (200, 503)
        json.loads(r.content)
        if r.status_code == 200:
            assert "etag" in r.headers

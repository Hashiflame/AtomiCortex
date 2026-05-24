"""
Tests for AtomiCortex API security: auth, CORS allowlist, rate limiting.

Covers Phase 5.2 hardening of ``src/api/main.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import _reset_rate_buckets, app

_API_KEY = "secret-test-key-abcdef"


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    """Configure env for auth tests; isolated empty DB."""
    db = tmp_path / "atomicortex.db"
    db.touch()
    monkeypatch.setenv("ATOMICORTEX_DB_PATHS", str(db))
    monkeypatch.setenv("ATOMICORTEX_API_KEY", _API_KEY)
    monkeypatch.setenv(
        "API_CORS_ORIGINS",
        "http://localhost:3000,https://atomicortex.app",
    )
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1000")
    _reset_rate_buckets()
    return TestClient(app)


# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------
def test_protected_endpoint_without_key_returns_401(auth_env):
    r = auth_env.get("/api/v1/stats")
    assert r.status_code == 401
    assert "Missing" in r.json()["detail"]


def test_protected_endpoint_with_wrong_key_returns_403(auth_env):
    r = auth_env.get("/api/v1/stats", headers={"X-API-Key": "nope"})
    assert r.status_code == 403
    assert "Invalid" in r.json()["detail"]


def test_protected_endpoint_with_correct_key_returns_200(auth_env):
    r = auth_env.get("/api/v1/stats", headers={"X-API-Key": _API_KEY})
    assert r.status_code == 200


def test_health_endpoint_accessible_without_auth(auth_env):
    """Health must remain open for monitoring / k8s probes."""
    r = auth_env.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_all_protected_endpoints_require_key(auth_env):
    protected = [
        "/api/v1/stats",
        "/api/v1/stats/4h",
        "/api/v1/signals",
        "/api/v1/signals/open",
        "/api/v1/equity-curve",
        "/api/v1/monthly-stats",
        "/api/v1/live",
    ]
    for path in protected:
        r = auth_env.get(path)
        assert r.status_code == 401, f"{path} should require API key"


# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------
def test_cors_never_returns_wildcard(auth_env):
    r = auth_env.get(
        "/api/v1/health", headers={"Origin": "https://evil.com"},
    )
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-origin" not in r.headers


def test_cors_allows_whitelisted_origin(auth_env):
    origin = "https://atomicortex.app"
    r = auth_env.get("/api/v1/health", headers={"Origin": origin})
    assert r.headers.get("access-control-allow-origin") == origin
    assert r.headers.get("vary") == "Origin"


def test_cors_blocks_unknown_origin(auth_env):
    r = auth_env.get(
        "/api/v1/health", headers={"Origin": "https://attacker.io"},
    )
    assert "access-control-allow-origin" not in r.headers


# ----------------------------------------------------------------------
# Rate limiting
# ----------------------------------------------------------------------
def test_rate_limit_blocks_excess_requests(tmp_path, monkeypatch):
    db = tmp_path / "atomicortex.db"
    db.touch()
    monkeypatch.setenv("ATOMICORTEX_DB_PATHS", str(db))
    monkeypatch.setenv("ATOMICORTEX_API_KEY", _API_KEY)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "5")
    _reset_rate_buckets()
    client = TestClient(app)

    statuses = [client.get("/api/v1/health").status_code for _ in range(7)]
    assert statuses[:5] == [200] * 5
    assert 429 in statuses[5:], f"expected 429 after limit, got {statuses}"


def test_rate_limit_resets_between_tests(tmp_path, monkeypatch):
    """Verify _reset_rate_buckets() clears the bucket state."""
    db = tmp_path / "atomicortex.db"
    db.touch()
    monkeypatch.setenv("ATOMICORTEX_DB_PATHS", str(db))
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "3")
    _reset_rate_buckets()
    client = TestClient(app)
    for _ in range(3):
        assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/health").status_code == 429

    _reset_rate_buckets()
    assert client.get("/api/v1/health").status_code == 200

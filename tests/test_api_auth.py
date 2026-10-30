"""
Tests for API-key auth on the FastAPI app (app/core/security.py +
app/api/main.py). Uses FastAPI's TestClient - no real server, no real
network calls, no real Groq/EDGAR calls needed for these specific checks.
"""
import os
import pytest

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_check_requires_no_auth():
    r = client.get("/health")
    assert r.status_code == 200


def test_cache_stats_requires_no_auth():
    r = client.get("/cache-stats")
    assert r.status_code == 200
    assert "total_entries" in r.json()


def test_analyze_rejects_missing_api_key():
    r = client.post("/analyze", json={"company_id": "0001318605", "tag": "Assets"})
    assert r.status_code == 401


def test_analyze_rejects_wrong_api_key():
    r = client.post(
        "/analyze",
        json={"company_id": "0001318605", "tag": "Assets"},
        headers={"X-API-Key": "definitely-not-the-right-key"},
    )
    assert r.status_code == 401


def test_analyze_fails_closed_when_api_key_unset(monkeypatch):
    from app.config.settings import settings
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.delenv("ALLOW_NO_AUTH", raising=False)
    r = client.post("/analyze", json={"company_id": "0001318605", "tag": "Assets"})
    assert r.status_code == 503


def test_analyze_allows_no_auth_only_with_explicit_flag(monkeypatch):
    from app.config.settings import settings
    from app.core.security import require_api_key
    import asyncio
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setenv("ALLOW_NO_AUTH", "true")
    assert asyncio.run(require_api_key(None)) is None

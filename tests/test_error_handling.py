"""A failing /analyze must not leak internal error text to the client."""
from fastapi.testclient import TestClient

import app.api.main as main_module

client = TestClient(main_module.app)


def test_500_does_not_leak_exception_text(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret-internal-detail /srv/path key=abc123")

    monkeypatch.setattr(main_module, "analyze_company", boom)
    r = client.post(
        "/analyze",
        json={"company_id": "0001318605", "tag": "Assets"},
        headers={"X-API-Key": "test-secret-123"},
    )
    assert r.status_code == 500
    assert "secret-internal-detail" not in r.text
    assert r.json()["detail"] == "Internal error while analyzing."
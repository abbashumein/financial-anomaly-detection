"""Confirms fetch_company_facts is actually cached - investigating one
company across 5 metric tags (as get_anomalous_metrics does) should hit
SEC's API once, not five times."""
import pytest
from app.services import edgar_client, cache


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.clear_cache()
    yield
    cache.clear_cache()


class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "entityName": "Test Co",
            "facts": {"us-gaap": {
                "Assets": {"units": {"USD": [
                    {"end": f"2024-0{i}-01", "val": 100 + i * 5} for i in range(1, 6)
                ]}},
                "Liabilities": {"units": {"USD": [
                    {"end": f"2024-0{i}-01", "val": 50 + i * 2} for i in range(1, 6)
                ]}},
            }},
        }


def test_repeated_calls_for_same_company_hit_network_once(monkeypatch):
    call_count = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        call_count["n"] += 1
        return _FakeResp()

    monkeypatch.setattr("app.services.edgar_client.requests.get", fake_get)

    edgar_client.get_live_sequence("0001318605", "Assets")
    edgar_client.get_live_sequence("0001318605", "Liabilities")
    edgar_client.get_live_sequence("0001318605", "Assets")

    assert call_count["n"] == 1, (
        f"expected 1 cached network call for repeated requests on the same "
        f"company, got {call_count['n']} - caching may be broken"
    )


def test_different_companies_are_not_cross_cached(monkeypatch):
    call_count = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        call_count["n"] += 1
        return _FakeResp()

    monkeypatch.setattr("app.services.edgar_client.requests.get", fake_get)

    edgar_client.get_live_sequence("0000000001", "Assets")
    edgar_client.get_live_sequence("0000000002", "Assets")  # different company

    assert call_count["n"] == 2, "different companies must not share a cache entry"

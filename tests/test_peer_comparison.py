"""
Tests for app/services/peer_comparison.py and the compare_to_peers agent
tool. Mocks the two SEC network calls (submissions.json for SIC lookup,
efts.sec.gov for peer discovery) - everything downstream (VAE scoring
via the real tool_score_company_metric) is real logic.
"""
import numpy as np
import pytest
from app.services import rag_agent, peer_comparison, edgar_client


FAKE_SUBMISSIONS = {
    "sic": "3711",
    "sicDescription": "Motor Vehicles & Passenger Car Bodies",
    "name": "Target Motors Inc",
}

FAKE_SEARCH_HITS = {
    "hits": {"hits": [
        {"_source": {"ciks": ["0000000002"], "display_names": ["Peer One Motors  (P1)  (CIK 0000000002)"]}},
        {"_source": {"ciks": ["0000000003"], "display_names": ["Peer Two Autos  (P2)  (CIK 0000000003)"]}},
        {"_source": {"ciks": ["0000000001"], "display_names": ["Target Motors Inc"]}},  # self - must be excluded
    ]}
}


def _fake_sequence(values):
    arr = np.array(values, dtype="float32")
    lo, hi = arr.min(), arr.max()
    scaled = (arr - lo) / (hi - lo) if hi != lo else np.zeros_like(arr)
    padded = np.pad(scaled, (0, 20 - len(scaled)), "constant").astype("float32")
    return padded


FAKE_SEQUENCES = {
    "0000000001": {"entity_name": "Target Motors Inc", "sequence": _fake_sequence([1, 90, 1, 90, 1, 90]),
                   "raw_values": [1, 90, 1, 90, 1, 90], "dates": ["2024-01-01"] * 6, "unit": "USD", "n_points": 6},
    "0000000002": {"entity_name": "Peer One Motors", "sequence": _fake_sequence([1, 2, 3, 4, 5, 6]),
                   "raw_values": [1, 2, 3, 4, 5, 6], "dates": ["2024-01-01"] * 6, "unit": "USD", "n_points": 6},
    "0000000003": {"entity_name": "Peer Two Autos", "sequence": _fake_sequence([1, 2, 3, 4, 5, 6]),
                   "raw_values": [1, 2, 3, 4, 5, 6], "dates": ["2024-01-01"] * 6, "unit": "USD", "n_points": 6},
}


@pytest.fixture(autouse=True)
def _clear_cache():
    from app.services import cache
    cache.clear_cache()
    yield
    cache.clear_cache()


@pytest.fixture
def fake_sec(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return FAKE_SUBMISSIONS if "submissions" in url else FAKE_SEARCH_HITS
        return R()
    monkeypatch.setattr("app.services.peer_comparison.requests.get", fake_get)

    def fake_get_live_sequence(company_id, tag):
        return FAKE_SEQUENCES[peer_comparison.normalize_cik(company_id)]
    monkeypatch.setattr(edgar_client, "get_live_sequence", fake_get_live_sequence)


def test_compare_to_peers_excludes_self_from_peer_list(fake_sec):
    result = rag_agent.tool_compare_to_peers("0000000001", "Liabilities")
    assert "error" not in result
    peer_names = [p["entity_name"] for p in result["peers"]]
    assert "Target Motors Inc" not in peer_names
    assert len(result["peers"]) == 2


def test_compare_to_peers_flags_target_as_more_anomalous(fake_sec):
    result = rag_agent.tool_compare_to_peers("0000000001", "Liabilities")
    assert result["target_score"] > result["peer_average_score"]
    assert result["more_anomalous_than_peers"] is True


def test_compare_to_peers_handles_no_sic_found(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        class R:
            status_code = 404
            def raise_for_status(self):
                raise Exception("404 not found")
            def json(self):
                return {}
        return R()
    monkeypatch.setattr("app.services.peer_comparison.requests.get", fake_get)

    result = rag_agent.tool_compare_to_peers("0000000001", "Liabilities")
    assert "error" in result


def test_compare_to_peers_handles_no_peers_found(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                if "submissions" in url:
                    return FAKE_SUBMISSIONS
                return {"hits": {"hits": []}}  # no peers found
        return R()
    monkeypatch.setattr("app.services.peer_comparison.requests.get", fake_get)

    result = rag_agent.tool_compare_to_peers("0000000001", "Liabilities")
    assert "error" in result

"""
Tests for get_full_history (edgar_client.py) and the get_historical_trend
agent tool - distinguishes a sudden one-period spike from gradual,
steady drift, using more history than the VAE's 6-quarter window sees.
"""
import pytest
from app.services import edgar_client, rag_agent


GRADUAL_FACTS = {
    "entityName": "Gradual Co",
    "facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2023-01-01", "val": 100},
        {"end": "2023-04-01", "val": 105},
        {"end": "2023-07-01", "val": 110},
        {"end": "2023-10-01", "val": 116},
        {"end": "2024-01-01", "val": 122},
        {"end": "2024-04-01", "val": 128},
    ]}}}},
}

SUDDEN_FACTS = {
    "entityName": "Sudden Co",
    "facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2023-01-01", "val": 100},
        {"end": "2023-04-01", "val": 102},
        {"end": "2023-07-01", "val": 101},
        {"end": "2023-10-01", "val": 350},
        {"end": "2024-01-01", "val": 352},
        {"end": "2024-04-01", "val": 349},
    ]}}}},
}


def test_get_full_history_returns_unwindowed_values(monkeypatch):
    """Unlike build_sequence (windowed to 6, scaled), get_full_history
    should return ALL raw values, unscaled."""
    monkeypatch.setattr(edgar_client, "fetch_company_facts", lambda cid: GRADUAL_FACTS)
    history = edgar_client.get_full_history("0000000001", "Assets")
    assert history["values"] == [100, 105, 110, 116, 122, 128]
    assert history["entity_name"] == "Gradual Co"


def test_gradual_growth_classified_as_gradual_drift(monkeypatch):
    monkeypatch.setattr(edgar_client, "fetch_company_facts", lambda cid: GRADUAL_FACTS)
    result = rag_agent.tool_get_historical_trend("0000000001", "Assets")
    assert result["pattern"] == "gradual_drift"


def test_sudden_jump_classified_as_sudden_spike(monkeypatch):
    monkeypatch.setattr(edgar_client, "fetch_company_facts", lambda cid: SUDDEN_FACTS)
    result = rag_agent.tool_get_historical_trend("0000000002", "Assets")
    assert result["pattern"] == "sudden_spike"
    assert result["largest_change_date"] == "2023-10-01"


def test_insufficient_history_returns_error(monkeypatch):
    short_facts = {
        "entityName": "Short Co",
        "facts": {"us-gaap": {"Assets": {"units": {"USD": [
            {"end": "2023-01-01", "val": 100},
            {"end": "2023-04-01", "val": 105},
        ]}}}},
    }
    monkeypatch.setattr(edgar_client, "fetch_company_facts", lambda cid: short_facts)
    result = rag_agent.tool_get_historical_trend("0000000003", "Assets")
    assert "error" in result


def test_build_sequence_still_works_after_refactor(monkeypatch):
    """Regression check: the shared _get_ordered_points refactor must not
    have broken the original windowed+scaled build_sequence path."""
    padded, raw_values, dates, unit = edgar_client.build_sequence(GRADUAL_FACTS, "Assets")
    assert raw_values == [100, 105, 110, 116, 122, 128]
    assert len(padded) == edgar_client.MAX_LEN
    assert unit == "USD"

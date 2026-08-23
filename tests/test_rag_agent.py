"""
Tests for app/services/rag_agent.py's tool functions.

These run against the REAL trained VAE model in models/vae_model.pt and
the REAL vae_scorer.score_sequence(). The only thing mocked is the SEC
EDGAR network call (edgar_client.get_live_sequence) - CI boxes and
sandboxes generally can't reach data.sec.gov, and we don't want tests
to depend on SEC's servers being up anyway. Everything downstream of
that one mock is your actual code.
"""
import numpy as np
import pytest

from app.services import rag_agent, edgar_client


def _fake_sequence(entity_name, values, unit="USD"):
    """Build a fake get_live_sequence() return value from raw quarterly
    numbers - mirrors exactly what edgar_client.build_sequence() would
    hand back, just without hitting the network."""
    arr = np.array(values, dtype="float32")
    lo, hi = arr.min(), arr.max()
    scaled = (arr - lo) / (hi - lo) if hi != lo else np.zeros_like(arr)
    padded = np.pad(scaled, (0, 20 - len(scaled)), "constant").astype("float32")
    return {
        "entity_name": entity_name,
        "sequence": padded,
        "raw_values": values,
        "dates": [f"2024-0{i+1}-01" for i in range(len(values))],
        "unit": unit,
        "n_points": len(values),
    }


@pytest.fixture
def fake_edgar(monkeypatch):
    """Swap in a controllable fake for the network call. Tests configure
    FAKE_DATA per-tag; any tag not in it raises the same error the real
    client raises for an undisclosed tag, so error-handling paths get
    exercised too."""
    FAKE_DATA = {}

    def fake_get_live_sequence(company_id, tag):
        if tag not in FAKE_DATA:
            raise edgar_client.EdgarLookupError(f"Tag '{tag}' not disclosed by this company.")
        return FAKE_DATA[tag]

    monkeypatch.setattr(edgar_client, "get_live_sequence", fake_get_live_sequence)
    return FAKE_DATA


# ---------- score_company_metric (pre-existing tool - regression check) ----------

def test_score_company_metric_smooth_sequence_scores_low(fake_edgar):
    """A smooth, steadily-growing sequence should NOT be flagged HIGH risk -
    this is the exact bug class the project's README calls out (stable
    companies falsely scoring HIGH). Guards against that regression."""
    fake_edgar["Assets"] = _fake_sequence("Smooth Co", [100, 105, 110, 115, 120, 125])

    result = rag_agent.tool_score_company_metric("0000000001", "Assets")

    assert "error" not in result
    assert result["entity_name"] == "Smooth Co"
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")  # always one of these
    assert isinstance(result["reconstruction_error"], float)
    assert result["risk_level"] != "HIGH", (
        f"Smooth, non-anomalous sequence scored HIGH (error={result['reconstruction_error']}) - "
        "possible regression of the distribution-mismatch bug."
    )


def test_score_company_metric_missing_tag_returns_error_not_crash(fake_edgar):
    """Tags the company doesn't disclose should come back as a clean
    error dict, never an exception that would crash the agent loop."""
    result = rag_agent.tool_score_company_metric("0000000001", "SomeTagThatDoesntExist")
    assert "error" in result


# ---------- get_anomalous_metrics (new V2 tool) ----------

def test_get_anomalous_metrics_ranks_most_erratic_highest(fake_edgar):
    """The metric with the most erratic pattern should rank first."""
    fake_edgar["Assets"] = _fake_sequence("Test Co", [100, 105, 110, 115, 120, 125])           # smooth
    fake_edgar["Liabilities"] = _fake_sequence("Test Co", [10, 90, 12, 88, 11, 91])              # erratic
    fake_edgar["StockholdersEquity"] = _fake_sequence("Test Co", [50, 50, 50, 50, 50, 50])       # flat
    fake_edgar["Revenues"] = _fake_sequence("Test Co", [20, 25, 30, 35, 40, 45])                 # smooth

    result = rag_agent.tool_get_anomalous_metrics(
        "0000000001", tags=["Assets", "Liabilities", "StockholdersEquity", "Revenues"]
    )

    tags_in_order = [m["tag"] for m in result["ranked_metrics"]]
    assert tags_in_order[0] == "Liabilities", f"expected erratic metric to rank first, got order: {tags_in_order}"
    assert len(result["ranked_metrics"]) == 4
    # confirm it's actually sorted descending by error, not accidentally right
    errors = [m["reconstruction_error"] for m in result["ranked_metrics"]]
    assert errors == sorted(errors, reverse=True)


def test_get_anomalous_metrics_skips_undisclosed_tags_without_crashing(fake_edgar):
    """A company that doesn't disclose one of the basket tags should have
    that tag land in skipped_tags, not blow up the whole call."""
    fake_edgar["Assets"] = _fake_sequence("Partial Co", [100, 105, 110, 115, 120, 125])
    # Liabilities intentionally NOT in fake_edgar - simulates an undisclosed tag

    result = rag_agent.tool_get_anomalous_metrics("0000000001", tags=["Assets", "Liabilities"])

    assert len(result["ranked_metrics"]) == 1
    assert result["ranked_metrics"][0]["tag"] == "Assets"
    assert len(result["skipped_tags"]) == 1
    assert result["skipped_tags"][0]["tag"] == "Liabilities"


def test_get_anomalous_metrics_uses_default_basket_when_no_tags_given(fake_edgar):
    """Calling with no explicit tags should fall back to DEFAULT_METRIC_BASKET,
    not crash or return nothing."""
    for tag in rag_agent.DEFAULT_METRIC_BASKET:
        fake_edgar[tag] = _fake_sequence("Default Co", [1, 2, 3, 4, 5, 6])

    result = rag_agent.tool_get_anomalous_metrics("0000000001")

    assert len(result["ranked_metrics"]) == len(rag_agent.DEFAULT_METRIC_BASKET)
    assert result["skipped_tags"] == []

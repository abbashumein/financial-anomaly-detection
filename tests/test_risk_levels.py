"""Risk levels must cover MEDIUM and HIGH, not only LOW."""
import numpy as np

from app.services import rag_agent, edgar_client
from app.services.vae_scorer import risk_level_for, P90, P95


def test_risk_level_boundaries():
    assert risk_level_for(P90 - 0.001) == "LOW"
    assert risk_level_for((P90 + P95) / 2) == "MEDIUM"
    assert risk_level_for(P95 + 0.001) == "HIGH"


def test_erratic_sequence_scores_above_low(monkeypatch):
    values = [1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100]
    arr = np.array(values, dtype="float32")
    scaled = (arr - arr.min()) / (arr.max() - arr.min())
    padded = np.pad(scaled, (0, 20 - len(scaled)), "constant").astype("float32")
    fake = {
        "entity_name": "Erratic Co",
        "sequence": padded,
        "raw_values": values,
        "dates": [f"2024-{i+1:02d}-01" for i in range(len(values))],
        "unit": "USD",
        "n_points": len(values),
    }
    monkeypatch.setattr(edgar_client, "get_live_sequence", lambda cid, tag: fake)

    result = rag_agent.tool_score_company_metric("0000000002", "Assets")

    assert "error" not in result
    assert result["risk_level"] in ("MEDIUM", "HIGH"), result["reconstruction_error"]
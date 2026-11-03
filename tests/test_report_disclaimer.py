"""Reports must carry the not-fraud disclaimer; prompt must forbid fraud claims."""
import json
from types import SimpleNamespace

from app.services import rag_agent


def _conclude_response(risk, finding):
    call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="conclude", arguments=json.dumps({"risk_level": risk, "finding": finding})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_report_always_carries_disclaimer(monkeypatch):
    resp = _conclude_response("HIGH", "possible earnings manipulation")
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    monkeypatch.setattr(rag_agent, "groq_client", fake)
    result = rag_agent.analyze_company("0000000001", "Assets")
    assert "not confirmed fraud" in result["final_report"]


def test_prompt_forbids_fraud_claims_and_names_window():
    assert "Never allege fraud" in rag_agent.SYSTEM_PROMPT
    assert "6 quarters" in rag_agent.SYSTEM_PROMPT

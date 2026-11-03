"""Agent must be able to finish a longer investigation (8 tool calls + conclude)."""
import json
from types import SimpleNamespace

from app.services import rag_agent


def _call(name, args, i):
    return SimpleNamespace(
        id=f"call{i}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(calls):
    msg = SimpleNamespace(content=None, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_agent_concludes_after_eight_tool_calls(monkeypatch):
    script = [
        _response([_call("deep_search", {"tag": "Assets"}, i)]) for i in range(8)
    ]
    script.append(
        _response([_call("conclude", {"risk_level": "MEDIUM", "finding": "ok"}, 99)])
    )
    it = iter(script)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: next(it)))
    )
    monkeypatch.setattr(rag_agent, "groq_client", fake_client)
    monkeypatch.setitem(rag_agent.DISPATCH, "deep_search", lambda args: {"ok": True})

    result = rag_agent.analyze_company("0000000001", "Assets")

    assert result["risk_level"] == "MEDIUM"
    assert len(result["agent_trace"]) == 8
    assert "did not conclude" not in result["final_report"]
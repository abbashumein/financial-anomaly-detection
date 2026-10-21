"""Tests for the agent loop in app/services/rag_agent.py.

The Groq client is replaced by a scripted fake, so no network or API key is
needed. These tests check that bad tool calls from the LLM do not crash
the loop, and that /analyze is not an `async def` that blocks the event loop.
"""
import inspect
import json
from types import SimpleNamespace

import pytest

from app.api import main as api_main
from app.services import rag_agent


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(*tool_calls):
    msg = SimpleNamespace(content=None, tool_calls=list(tool_calls))
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedGroq:
    """Fake groq client. Returns the scripted responses in order and keeps
    a copy of the messages it was given on each call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_messages = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen_messages.append(list(kwargs["messages"]))
        return self._responses.pop(0)


def _conclude(call_id="c1"):
    return _tool_call(call_id, "conclude", {"risk_level": "LOW", "finding": "ok"})


def test_missing_tool_argument_does_not_crash_the_loop(monkeypatch):
    """LLM calls get_sec_filing_context without the required 'tag'.
    Before the fix this raised KeyError and became an HTTP 500."""
    fake = ScriptedGroq([
        _response(_tool_call("t1", "get_sec_filing_context", {"company_id": "0000000001"})),
        _response(_conclude()),
    ])
    monkeypatch.setattr(rag_agent, "groq_client", fake)

    result = rag_agent.analyze_company("0000000001", "Assets")

    assert result["risk_level"] == "LOW"
    assert result["agent_trace"] == ["get_sec_filing_context"]
    # the LLM must be TOLD about the error on its next step
    tool_msgs = [m for m in fake.seen_messages[1] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "missing argument" in tool_msgs[0]["content"]


def test_tool_that_raises_becomes_an_error_result(monkeypatch):
    monkeypatch.setitem(rag_agent.DISPATCH, "get_historical_trend", lambda a: 1 / 0)
    fake = ScriptedGroq([
        _response(_tool_call("t1", "get_historical_trend", {"company_id": "1", "tag": "Assets"})),
        _response(_conclude()),
    ])
    monkeypatch.setattr(rag_agent, "groq_client", fake)

    result = rag_agent.analyze_company("0000000001", "Assets")

    assert result["risk_level"] == "LOW"
    tool_msgs = [m for m in fake.seen_messages[1] if m.get("role") == "tool"]
    assert "ZeroDivisionError" in tool_msgs[0]["content"]


def test_run_tool_unknown_tool_returns_error():
    assert "error" in rag_agent.run_tool("not_a_real_tool", {})


def test_run_tool_never_raises_on_empty_args():
    for name in rag_agent.DISPATCH:
        out = rag_agent.run_tool(name, {})
        assert isinstance(out, dict)


def test_analyze_endpoint_is_sync_so_it_runs_in_a_threadpool():
    """/analyze does blocking work (Groq, SEC requests, torch). As
    `async def` it would freeze the event loop for every other request."""
    assert not inspect.iscoroutinefunction(api_main.analyze)

"""
Regression test for a real production bug: when the LLM replies with no
tool call, msg.tool_calls is None. The old code did
    {"role": "assistant", "content": ..., "tool_calls": None}
and sent that back to Groq on the NEXT turn, which Groq's stricter
message validation rejects with a 400 error
("messages.N.tool_calls: Value is not nullable"). This only appears on
the SECOND api call in a conversation, which is why it wasn't caught
until real usage - a single mocked response can't catch it.

This test uses a fake Groq client that validates every outgoing
messages array on every call, mimicking Groq's real API behavior, so
this exact bug can never silently come back.
"""
import pytest
from app.services import rag_agent, edgar_client


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _validate_no_null_tool_calls(messages):
    """Mimics Groq's real validation: a message dict must not contain
    'tool_calls': None - the key should be OMITTED entirely instead."""
    for m in messages:
        if m.get("role") == "assistant" and "tool_calls" in m and m["tool_calls"] is None:
            raise AssertionError(
                f"BUG REPRODUCED: assistant message has tool_calls=None instead of "
                f"omitting the key entirely - this is exactly what breaks against the "
                f"real Groq API. Message: {m}"
            )


@pytest.fixture
def fake_groq_conversation(monkeypatch):
    """Simulates: tool call -> no-tool-call reply (the trigger case) ->
    conclude tool call. Validates message shape on every single call."""
    call_sequence = [
        _FakeResponse(_FakeMessage(
            content=None,
            tool_calls=[_FakeToolCall("score_company_metric", '{"company_id": "0000000001", "tag": "Assets"}')],
        )),
        _FakeResponse(_FakeMessage(content="Let me think about this.", tool_calls=None)),  # the trigger case
        _FakeResponse(_FakeMessage(
            content=None,
            tool_calls=[_FakeToolCall("conclude", '{"risk_level": "LOW", "finding": "Nothing unusual."}', call_id="call_2")],
        )),
    ]
    call_count = {"n": 0}

    class FakeCompletions:
        @staticmethod
        def create(messages, **kwargs):
            _validate_no_null_tool_calls(messages)  # this is the actual regression check
            response = call_sequence[call_count["n"]]
            call_count["n"] += 1
            return response

    class FakeChat:
        completions = FakeCompletions()

    class FakeGroqClient:
        chat = FakeChat()

    monkeypatch.setattr(rag_agent, "groq_client", FakeGroqClient())

    def fake_get_live_sequence(company_id, tag):
        import numpy as np
        return {
            "entity_name": "Test Co",
            "sequence": np.zeros(20, dtype="float32"),
            "raw_values": [1, 2, 3, 4, 5, 6],
            "dates": ["2024-01-01"] * 6,
            "unit": "USD",
            "n_points": 6,
        }
    monkeypatch.setattr(edgar_client, "get_live_sequence", fake_get_live_sequence)

    return call_count


def test_agent_survives_a_no_tool_call_turn_without_400_error(fake_groq_conversation):
    """The actual regression test: a conversation that includes a
    no-tool-call turn (which triggers the bug) must complete successfully,
    not raise the message-validation error."""
    result = rag_agent.analyze_company("0000000001", "Assets")
    assert result["risk_level"] == "LOW"
    assert fake_groq_conversation["n"] == 3, "expected all 3 turns to complete"

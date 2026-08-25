"""
Tests for app/services/graph_rag.py - the small in-memory knowledge
graph layer. Covers both halves: deterministic edges (free, no LLM)
and LLM-based entity extraction (mocked network call, but the caching
behavior itself is genuinely verified by counting real call attempts).
"""
import pytest
from app.services import graph_rag


@pytest.fixture(autouse=True)
def _reset():
    graph_rag.reset_graph()
    yield
    graph_rag.reset_graph()


def test_deterministic_edges_are_reachable_at_one_hop():
    graph_rag.add_company("0000000001", "Target Motors Inc")
    graph_rag.add_metric_relationship("0000000001", "NetIncomeLoss", "HIGH")
    graph_rag.add_filing_relationship("0000000001", "https://sec.gov/filing1.htm", "10-K", "NetIncomeLoss")
    graph_rag.add_peer_relationship("0000000001", "0000000002", "Peer Motors Inc")

    connections = graph_rag.find_connections("0000000001", max_hops=1)
    targets = {c["target"] for c in connections}

    assert "NetIncomeLoss" in targets
    assert "https://sec.gov/filing1.htm" in targets
    assert "0000000002" in targets


def test_shortest_path_is_preferred_over_longer_path():
    """Company -> Metric directly (1 hop) AND Company -> Filing ->
    Metric (2 hops) both exist - traversal must report the shorter one."""
    graph_rag.add_company("0000000001", "Target Motors Inc")
    graph_rag.add_metric_relationship("0000000001", "NetIncomeLoss", "HIGH")
    graph_rag.add_filing_relationship("0000000001", "https://sec.gov/filing1.htm", "10-K", "NetIncomeLoss")

    connections = graph_rag.find_connections("0000000001", max_hops=2)
    metric_conn = next(c for c in connections if c["target"] == "NetIncomeLoss")
    assert metric_conn["hops"] == 1


def test_max_hops_limits_traversal_depth():
    graph_rag.add_company("0000000001", "Target Motors Inc")
    graph_rag.add_filing_relationship("0000000001", "https://sec.gov/filing1.htm", "10-K", "NetIncomeLoss")
    # NetIncomeLoss is 2 hops away via the filing
    connections_0hop = graph_rag.find_connections("0000000001", max_hops=0)
    assert connections_0hop == []


def test_unknown_company_returns_empty_connections():
    assert graph_rag.find_connections("nonexistent-cik") == []


class _FakeGroqResponse:
    def __init__(self, content):
        class Choice:
            def __init__(self, content):
                class Message:
                    def __init__(self, content):
                        self.content = content
                self.message = Message(content)
        self.choices = [Choice(content)]


@pytest.fixture
def fake_groq(monkeypatch):
    call_log = {"count": 0}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    call_log["count"] += 1
                    return _FakeGroqResponse(
                        '[{"entity": "Acme Leasing LLC", "type": "Subsidiary", '
                        '"relationship": "wholly-owned subsidiary"}]'
                    )

    monkeypatch.setattr(graph_rag, "_get_groq", lambda: FakeClient())
    return call_log


def test_extraction_parses_entities_from_excerpt(fake_groq):
    triples = graph_rag.extract_entities_from_excerpt(
        "...through its subsidiary, Acme Leasing LLC...", "0000000001"
    )
    assert len(triples) == 1
    assert triples[0]["entity"] == "Acme Leasing LLC"
    assert fake_groq["count"] == 1


def test_extraction_is_cached_by_excerpt_content(fake_groq):
    excerpt = "...through its subsidiary, Acme Leasing LLC..."
    graph_rag.extract_entities_from_excerpt(excerpt, "0000000001")
    graph_rag.extract_entities_from_excerpt(excerpt, "0000000001")  # identical text again

    assert fake_groq["count"] == 1, "identical excerpt must be served from cache, not re-sent to the LLM"


def test_extracted_entity_becomes_reachable_graph_node(fake_groq):
    graph_rag.add_company("0000000001", "Target Motors Inc")
    graph_rag.extract_entities_from_excerpt(
        "...through its subsidiary, Acme Leasing LLC...", "0000000001"
    )
    connections = graph_rag.find_connections("0000000001", max_hops=1)
    names = [c["target_name"] for c in connections]
    assert "Acme Leasing LLC" in names


def test_extraction_handles_malformed_llm_output_gracefully(monkeypatch):
    class BadClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeGroqResponse("not valid json at all")

    monkeypatch.setattr(graph_rag, "_get_groq", lambda: BadClient())

    triples = graph_rag.extract_entities_from_excerpt("some excerpt text", "0000000001")
    assert triples == []  # fails closed, doesn't crash

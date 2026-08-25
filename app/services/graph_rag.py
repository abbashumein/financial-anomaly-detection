# app/services/graph_rag.py
"""
Small in-memory knowledge graph, built from the SAME evidence the other
V2 tools already gather - not a second, separate retrieval system.

Two ways edges get added:
  1. DETERMINISTIC (free, instant, no LLM) - wired directly from
     structured tool outputs already computed: Company -[REPORTED]->
     Metric, Company -[FILED]-> Filing, Company -[PEER_OF]-> Peer.
  2. LLM-EXTRACTED (one Groq call per UNIQUE filing excerpt, cached
     forever by content hash so the same text is never billed twice) -
     pulls entities MENTIONED INSIDE the filing text that the
     deterministic layer can't see (e.g. "...through its subsidiary,
     Acme Leasing LLC...").

Why a graph instead of more flat retrieval: it enables MULTI-HOP
questions flat vector search can't answer - e.g. "is this company
connected, through a shared subsidiary or industry peer, to another
company with a similar anomaly?" That's a graph traversal, not a single
similarity search.

Uses networkx (in-memory, no server, no hosting cost) - proportionate
infra for a single-process portfolio deployment; Neo4j would be solving
a scale problem this project doesn't have.
"""
import hashlib
import json
import networkx as nx
from groq import Groq

from app.config.settings import settings

_graph = nx.MultiDiGraph()
_extraction_cache: dict[str, list[dict]] = {}  # excerpt hash -> extracted triples

_groq_client = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def add_company(cik: str, name: str) -> None:
    _graph.add_node(cik, type="Company", name=name)


def add_metric_relationship(company_cik: str, tag: str, risk_level: str) -> None:
    _graph.add_node(tag, type="Metric")
    _graph.add_edge(company_cik, tag, relationship="REPORTED", risk_level=risk_level)


def add_filing_relationship(company_cik: str, filing_url: str, form: str, tag: str) -> None:
    _graph.add_node(filing_url, type="Filing", form=form)
    _graph.add_edge(company_cik, filing_url, relationship="FILED")
    _graph.add_edge(filing_url, tag, relationship="DISCUSSES")


def add_peer_relationship(company_cik: str, peer_cik: str, peer_name: str) -> None:
    _graph.add_node(peer_cik, type="Company", name=peer_name)
    _graph.add_edge(company_cik, peer_cik, relationship="PEER_OF")


def _excerpt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


EXTRACTION_PROMPT = """Extract named entities mentioned in this SEC filing excerpt that are DIFFERENT from the main company (e.g. subsidiaries, other named companies, executives, related entities). Return ONLY a JSON array, no other text, no markdown fences:
[{"entity": "name", "type": "Subsidiary|Company|Person", "relationship": "short description of the connection"}]
If none found, return exactly: []

Excerpt:
"""


def extract_entities_from_excerpt(excerpt: str, company_cik: str) -> list[dict]:
    """LLM-based extraction, cached by excerpt content hash - the same
    filing text is NEVER sent to the LLM twice, protecting the free tier."""
    key = _excerpt_hash(excerpt)
    if key in _extraction_cache:
        return _extraction_cache[key]

    try:
        response = _get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + excerpt}],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
            if raw.endswith("```"):
                raw = raw[:-3]
        triples = json.loads(raw.strip())
    except Exception:
        triples = []

    _extraction_cache[key] = triples

    for t in triples:
        entity_id = f"entity:{t['entity']}"
        _graph.add_node(entity_id, type=t.get("type", "Entity"), name=t["entity"])
        _graph.add_edge(company_cik, entity_id, relationship=t.get("relationship", "MENTIONED_WITH"))

    return triples


def find_connections(company_cik: str, max_hops: int = 3) -> list[dict]:
    """Multi-hop traversal: everything reachable from this company
    within max_hops steps, and the relationship chain that connects
    them. This is the actual GraphRAG payoff - a flat vector search
    returns similar TEXT, this returns a REASONING PATH."""
    if company_cik not in _graph:
        return []

    results = []
    for target in _graph.nodes:
        if target == company_cik:
            continue
        try:
            path = nx.shortest_path(_graph, company_cik, target)
        except nx.NetworkXNoPath:
            continue
        hops = len(path) - 1
        if hops > max_hops:
            continue

        relationships = []
        for i in range(len(path) - 1):
            edge_data = _graph.get_edge_data(path[i], path[i + 1])
            relationships.append(list(edge_data.values())[0].get("relationship", "?"))

        results.append({
            "target": target,
            "target_type": _graph.nodes[target].get("type"),
            "target_name": _graph.nodes[target].get("name", target),
            "hops": hops,
            "path_relationships": relationships,
        })

    results.sort(key=lambda r: r["hops"])
    return results


def graph_stats() -> dict:
    return {"nodes": _graph.number_of_nodes(), "edges": _graph.number_of_edges()}


def reset_graph() -> None:
    """Mainly for tests - clears graph state between investigations."""
    global _graph, _extraction_cache
    _graph = nx.MultiDiGraph()
    _extraction_cache = {}

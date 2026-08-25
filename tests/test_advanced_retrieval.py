"""
Tests for app/services/advanced_retrieval.py - the metadata-filter +
hybrid-search + rerank pipeline that replaced blind top-k semantic search.

The CrossEncoder's real model weights come from Hugging Face at runtime
(first call downloads ~80MB) - that download itself isn't something a
unit test should depend on, so _reranker is swapped for a fake scorer.
Everything else (BM25 index, metadata filtering, merge/dedupe logic,
sort-by-score reordering) is real.
"""
import pytest
from app.services import advanced_retrieval


DOCS = [
    "Company: Acme Corp Metric: Revenues Score: 0.91",
    "Company: Beta Inc Metric: Liabilities Score: 0.20",
    "Company: Acme Corp Metric: Liabilities Score: 0.15",
    "Company: Gamma LLC Metric: Revenues Score: 0.85",
    "Company: Delta Co Metric: Revenues Score: 0.10",
]
METAS = [
    {"company": "Acme Corp", "tag": "Revenues"},
    {"company": "Beta Inc", "tag": "Liabilities"},
    {"company": "Acme Corp", "tag": "Liabilities"},
    {"company": "Gamma LLC", "tag": "Revenues"},
    {"company": "Delta Co", "tag": "Revenues"},
]


@pytest.fixture(autouse=True)
def _build_index():
    advanced_retrieval.build_bm25_index(DOCS, METAS)
    yield


class _FakeReranker:
    """Deterministic fake: prefers any doc containing 'Acme'."""
    def predict(self, pairs):
        return [5.0 if "Acme" in doc else 1.0 for _, doc in pairs]


@pytest.fixture
def fake_reranker(monkeypatch):
    monkeypatch.setattr(advanced_retrieval, "_reranker", _FakeReranker())


def test_keyword_search_ranks_exact_match_first():
    results = advanced_retrieval.keyword_search("Acme Corp Revenues", top_k=5)
    assert results[0]["doc"] == "Company: Acme Corp Metric: Revenues Score: 0.91"


def test_keyword_search_metadata_filter_excludes_wrong_tag():
    results = advanced_retrieval.keyword_search("Acme Corp", tag="Liabilities", top_k=5)
    assert len(results) == 1
    assert all(r["meta"]["tag"] == "Liabilities" for r in results)


def test_keyword_search_returns_empty_before_index_built():
    # simulate a fresh, unbuilt index
    advanced_retrieval._bm25_index = None
    assert advanced_retrieval.keyword_search("anything") == []


def test_rerank_reorders_by_cross_encoder_score(fake_reranker):
    candidates = [{"doc": d, "meta": m} for d, m in zip(DOCS, METAS)]
    top = advanced_retrieval.rerank("Acme Corp anomaly", candidates, top_n=2)
    assert len(top) == 2
    assert all("Acme" in c["doc"] for c in top)
    assert top[0]["rerank_score"] == 5.0


def test_rerank_handles_empty_candidates(fake_reranker):
    assert advanced_retrieval.rerank("query", [], top_n=3) == []


def test_hybrid_search_dedupes_overlapping_semantic_and_keyword_hits():
    class _FakeCollection:
        def query(self, query_texts, n_results, where=None):
            # semantic search "finds" the same top doc keyword search will too
            return {
                "documents": [[DOCS[0]]],
                "metadatas": [[METAS[0]]],
            }

    results = advanced_retrieval.hybrid_search(_FakeCollection(), "Acme Corp Revenues", top_k=5)
    doc_texts = [r["doc"] for r in results]
    assert doc_texts.count(DOCS[0]) == 1, "same doc from semantic + keyword search should be de-duped, not doubled"

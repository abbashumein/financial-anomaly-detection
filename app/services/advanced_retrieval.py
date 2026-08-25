# app/services/advanced_retrieval.py
"""
Upgrades retrieval from "blind top-k semantic search" to a proper
3-stage pipeline:

  1. METADATA FILTERING - narrow candidates to the same metric tag
     BEFORE any similarity search, so a Revenues anomaly is never
     compared against a Liabilities case just because the text
     happens to embed nearby.

  2. HYBRID RETRIEVAL - combine semantic search (Chroma embeddings -
     good at "conceptually similar") with keyword search (BM25 - good
     at exact terms like company names, which embeddings often blur)
     and merge the candidate pools.

  3. RERANKING - a CrossEncoder reads the query and EACH candidate
     TOGETHER (slower but far more accurate than embedding cosine
     similarity alone) and only the top N survive to reach the LLM.

This is the standard "retrieve broad, rerank precise" pattern that
separates real RAG from a single vector-similarity call.
"""
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

_bm25_index = None
_bm25_docs = None
_bm25_metas = None
_reranker = None  # lazy-loaded - downloading the model at import time would
                   # slow down every single startup, even requests that
                   # never call this tool.


def build_bm25_index(docs: list[str], metas: list[dict]) -> None:
    global _bm25_index, _bm25_docs, _bm25_metas
    tokenized = [d.lower().split() for d in docs]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_docs = docs
    _bm25_metas = metas


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def keyword_search(query: str, tag: str | None = None, top_k: int = 10) -> list[dict]:
    """BM25 keyword search - catches exact-term matches (company names,
    tag names) that semantic embeddings sometimes miss."""
    if _bm25_index is None:
        return []

    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked_indices:
        if scores[i] <= 0:
            break  # BM25Okapi returns 0 for no term overlap at all - stop, not just skip
        if tag and _bm25_metas[i].get("tag") != tag:
            continue
        results.append({"doc": _bm25_docs[i], "meta": _bm25_metas[i]})
        if len(results) >= top_k:
            break
    return results


def semantic_search(collection, query: str, tag: str | None = None, top_k: int = 10) -> list[dict]:
    """Chroma embedding similarity search, with a metadata filter applied
    BEFORE the similarity search runs - not after."""
    where = {"tag": tag} if tag else None
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return [{"doc": d, "meta": m} for d, m in zip(docs, metas)]


def hybrid_search(collection, query: str, tag: str | None = None, top_k: int = 10) -> list[dict]:
    """Merge semantic + keyword candidate pools, de-duplicated by doc text."""
    semantic_results = semantic_search(collection, query, tag=tag, top_k=top_k)
    keyword_results = keyword_search(query, tag=tag, top_k=top_k)

    merged = {}
    for item in semantic_results + keyword_results:
        merged[item["doc"]] = item  # de-dupe by text; fine if metadata is identical either way
    return list(merged.values())


def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """Cross-encoder reranking: score each (query, candidate) PAIR
    together, rather than comparing independently-computed embeddings."""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["doc"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]


def retrieve(collection, query: str, tag: str | None = None, top_n: int = 3) -> list[dict]:
    """The full pipeline in one call: filtered hybrid search -> rerank."""
    candidates = hybrid_search(collection, query, tag=tag, top_k=10)
    return rerank(query, candidates, top_n=top_n)

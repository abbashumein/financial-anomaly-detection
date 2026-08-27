# eval_harness.py
"""
Real evaluation harness - produces an honest performance metrics table by
actually running test companies through the agent and measuring results.

Usage:
    python eval_harness.py

Requires: real GROQ_API_KEY and EDGAR_USER_AGENT set (via .env), and
real internet access (this hits live SEC + Groq APIs - each run costs
real Groq tokens, though Groq's free tier covers this easily).

Design note: unlike the CDSS example, there's no labeled ground truth for
"is this company actually committing fraud" - so accuracy here means
"did the agent behave correctly given the evidence" (called the right
tools, didn't crash, cited real evidence) rather than "did it correctly
classify fraud." That distinction matters and should be stated in your
README alongside this table - see the Performance Metrics section.
"""
import time
import json
from app.services.rag_agent import (
    analyze_company, collection,
    tool_get_anomalous_metrics, tool_get_sec_filing_context,
    tool_compare_to_peers, tool_get_historical_trend,
    tool_retrieve_similar_cases, tool_graph_investigate,
)
from app.services import advanced_retrieval

# ---- Test cases: adjust/add real CIKs you want to evaluate ----
# tag choices should match DEFAULT_METRIC_BASKET tags for realistic runs.
TEST_CASES = [
    # Large-cap, stable companies - expect LOW risk, agent should stop early
    {"company_id": "0001318605", "tag": "Assets", "label": "Tesla - Assets"},
    {"company_id": "0001318605", "tag": "Liabilities", "label": "Tesla - Liabilities"},
    {"company_id": "0001318605", "tag": "NetIncomeLoss", "label": "Tesla - NetIncomeLoss"},
    {"company_id": "0000320193", "tag": "Assets", "label": "Apple - Assets"},
    {"company_id": "0000320193", "tag": "Revenues", "label": "Apple - Revenues"},
    {"company_id": "0000789019", "tag": "Revenues", "label": "Microsoft - Revenues"},
    {"company_id": "0000789019", "tag": "Assets", "label": "Microsoft - Assets"},
    {"company_id": "0001652044", "tag": "Assets", "label": "Alphabet - Assets"},
    {"company_id": "0001018724", "tag": "Revenues", "label": "Amazon - Revenues"},
    {"company_id": "0001045810", "tag": "NetIncomeLoss", "label": "Nvidia - NetIncomeLoss"},

    # Small-cap / previously-flagged companies - more likely to trigger
    # MEDIUM/HIGH risk. CIKs confirmed via real SEC filing text.
    {"company_id": "0001434601", "tag": "Assets", "label": "Transglobal Mgmt Group (fka Marquie Group) - Assets"},
    {"company_id": "0001434601", "tag": "Liabilities", "label": "Transglobal Mgmt Group (fka Marquie Group) - Liabilities"},
    {"company_id": "0000811222", "tag": "Assets", "label": "Cardiff Lexington Corp - Assets"},
    {"company_id": "0000811222", "tag": "Liabilities", "label": "Cardiff Lexington Corp - Liabilities"},
    {"company_id": "0001169138", "tag": "Assets", "label": "GivBux Inc - Assets"},
    {"company_id": "0001169138", "tag": "NetIncomeLoss", "label": "GivBux Inc - NetIncomeLoss"},
]


DIRECT_TOOL_TEST_CASE = {"company_id": "0001318605", "tag": "Assets", "ticker": "TSLA"}


def run_direct_tool_exercise():
    """
    Calls each tool DIRECTLY, independent of whether the agent's own
    decision logic would choose to call it. This is a different, honest
    claim than the agent trace above: "every tool works when called" -
    not "the agent chose to call every tool on these particular inputs."
    Both are true and both matter; conflating them would overclaim.
    """
    cid, tag = DIRECT_TOOL_TEST_CASE["company_id"], DIRECT_TOOL_TEST_CASE["tag"]
    results = {}

    tests = [
        ("get_anomalous_metrics", lambda: tool_get_anomalous_metrics(cid)),
        ("get_sec_filing_context", lambda: tool_get_sec_filing_context(cid, tag)),
        ("compare_to_peers", lambda: tool_compare_to_peers(cid, tag)),
        ("get_historical_trend", lambda: tool_get_historical_trend(cid, tag)),
        ("retrieve_similar_cases", lambda: tool_retrieve_similar_cases("Tesla", tag)),
        ("graph_investigate", lambda: tool_graph_investigate(cid, tag)),
    ]

    for name, fn in tests:
        start = time.time()
        try:
            result = fn()
            elapsed = time.time() - start
            has_error = isinstance(result, dict) and "error" in result
            results[name] = {
                "success": not has_error,
                "latency_s": round(elapsed, 2),
                "note": result.get("error") if has_error else "OK",
            }
        except Exception as e:
            results[name] = {"success": False, "latency_s": round(time.time() - start, 2), "note": str(e)}

    return results


def print_rerank_scores(cid, tag):
    """Shows the actual CrossEncoder rerank scores hidden inside
    retrieve_similar_cases, so reranking is visible, not just trusted."""
    query = f"Tesla {tag}"
    candidates = advanced_retrieval.hybrid_search(collection, query, tag=tag, top_k=10)
    ranked = advanced_retrieval.rerank(query, candidates, top_n=5)

    print("\n" + "=" * 60)
    print("RERANK SCORES (Advanced RAG detail)")
    print("=" * 60)
    print(f"Candidates before rerank: {len(candidates)}")
    for c in ranked:
        print(f"  {c['rerank_score']:.3f}  {c['doc']}")


def print_direct_tool_report(results):
    print("\n" + "=" * 60)
    print("DIRECT TOOL EXERCISE (each tool called independently)")
    print("=" * 60)
    print("| Tool | Works | Latency | Note |")
    print("|---|---|---|---|")
    for name, r in results.items():
        status = "Yes" if r["success"] else "FAILED"
        print(f"| `{name}` | {status} | {r['latency_s']}s | {r['note']} |")


def run_eval():
    results = []
    tool_call_counts = {}

    for case in TEST_CASES:
        print(f"Running: {case['label']} ({case['company_id']}, {case['tag']})...")
        start = time.time()
        try:
            result = analyze_company(case["company_id"], case["tag"])
            elapsed = time.time() - start
            trace = result.get("agent_trace", [])

            for tool_name in trace:
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

            results.append({
                "case": case["label"],
                "success": True,
                "risk_level": result.get("risk_level"),
                "n_tools_called": len(trace),
                "trace": trace,
                "latency_s": round(elapsed, 2),
                "report_length": len(result.get("final_report", "")),
                "cites_real_number": _mentions_a_number(result.get("final_report", "")),
            })
        except Exception as e:
            results.append({
                "case": case["label"],
                "success": False,
                "error": str(e),
                "latency_s": round(time.time() - start, 2),
            })

    return results, tool_call_counts


def _mentions_a_number(text: str) -> bool:
    """Crude 'grounded response' check: does the final report contain an
    actual number (a real score/percentage), not just vague language?"""
    import re
    return bool(re.search(r"\d+\.\d+|\d+%", text))


def print_report(results, tool_call_counts):
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    latencies = [r["latency_s"] for r in results if r["success"]]
    grounded = sum(1 for r in results if r.get("cites_real_number"))

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    for r in results:
        print(f"\n{r['case']}")
        if r["success"]:
            print(f"  Risk: {r['risk_level']} | Tools called: {r['n_tools_called']} | Latency: {r['latency_s']}s")
            print(f"  Trace: {' -> '.join(r['trace'])}")
        else:
            print(f"  FAILED: {r['error']}")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE (paste into your README)")
    print("=" * 60)
    print(f"| Metric | Value |")
    print(f"|---|---|")
    print(f"| Test cases run | {n} |")
    print(f"| Success rate (no crash) | {n_success}/{n} ({100*n_success//n if n else 0}%) |")
    print(f"| Grounded responses (cites a real number) | {grounded}/{n_success} ({100*grounded//n_success if n_success else 0}%) |")
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[min(len(latencies)-1, int(len(latencies)*0.95))]
        print(f"| p50 latency | {p50}s |")
        print(f"| p95 latency | {p95}s |")
    print(f"| Tool call distribution | {json.dumps(tool_call_counts)} |")
    try:
        print(f"| Records in ChromaDB | {collection.count()} |")
    except Exception:
        pass


if __name__ == "__main__":
    results, tool_call_counts = run_eval()
    print_report(results, tool_call_counts)

    direct_results = run_direct_tool_exercise()
    print_direct_tool_report(direct_results)

    print_rerank_scores(DIRECT_TOOL_TEST_CASE["company_id"], DIRECT_TOOL_TEST_CASE["tag"])

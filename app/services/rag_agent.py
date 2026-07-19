# app/services/rag_agent.py
"""
Agentic RAG investigator.

Difference from the old version: the LLM is handed a toolbox
(score_company_metric, retrieve_similar_cases, deep_search, conclude) and
decides for itself which tool to call, in what order, and when it has
enough evidence to conclude — instead of walking a fixed
assess_risk -> retrieve -> investigate graph with a hardcoded input score.

score_company_metric is the tool that actually runs the trained VAE live
against real SEC EDGAR data. That's the fix for the "the API never touches
the model" gap.
"""
import json
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
import pandas as pd

from app.config.settings import settings
from app.services import edgar_client
from app.services import vae_scorer

MAX_AGENT_STEPS = 6

# ---------- retrieval store (unchanged) ----------

chroma_client = chromadb.PersistentClient(path="data/chromadb")
ef = DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="anomalies", embedding_function=ef)

if collection.count() == 0:
    print("Ingesting...")
    df = pd.read_csv("data/anomaly_results.csv").dropna(subset=["company", "tag", "anomaly_score"]).head(500)
    docs, metas, ids = [], [], []
    for i, row in df.iterrows():
        docs.append(f"Company: {row['company']} Metric: {row['tag']} Score: {row['anomaly_score']}")
        metas.append({"company": str(row["company"]), "tag": str(row["tag"])})
        ids.append(str(i))
    collection.add(documents=docs, metadatas=metas, ids=ids)
    print(f"Done - {len(docs)} records")

groq_client = Groq(api_key=settings.groq_api_key)

# ---------- tool implementations ----------

def tool_score_company_metric(company_id: str, tag: str) -> dict:
    """Fetches live SEC EDGAR data and runs it through the real VAE."""
    try:
        live = edgar_client.get_live_sequence(company_id, tag)
    except edgar_client.EdgarLookupError as e:
        return {"error": str(e)}

    result = vae_scorer.score_sequence(live["sequence"], model_path=settings.model_path)
    return {
        "entity_name": live["entity_name"],
        "tag": tag,
        "n_data_points": live["n_points"],
        "most_recent_dates": live["dates"][-3:],
        "reconstruction_error": round(result["reconstruction_error"], 6),
        "risk_level": result["risk_level"],
        "is_anomaly": result["is_anomaly"],
        "percentile_context": result["percentile_context"],
    }


def tool_retrieve_similar_cases(company: str, tag: str, n: int = 3) -> str:
    results = collection.query(query_texts=[f"{company} {tag}"], n_results=n)
    docs = results["documents"][0] if results["documents"] else []
    return "\n".join(docs) if docs else "No similar cases found in the historical corpus."


def tool_deep_search(tag: str) -> str:
    results = collection.query(query_texts=[f"{tag} fraud anomaly high risk"], n_results=5)
    docs = results["documents"][0] if results["documents"] else []
    return "\n".join(docs) if docs else "No broader patterns found."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_company_metric",
            "description": (
                "Fetch a company's real, live SEC EDGAR filing history for one "
                "financial metric and run it through the trained VAE anomaly "
                "detector. Always call this first for any new investigation — "
                "it's the only tool that produces an actual model-derived score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_id": {"type": "string", "description": "SEC CIK number, e.g. '0001318605'"},
                    "tag": {"type": "string", "description": "US-GAAP tag, e.g. 'Assets', 'Revenues', 'NetIncomeLoss'"},
                },
                "required": ["company_id", "tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar_cases",
            "description": "Search the historical corpus for similar company/metric anomaly cases to give context for the score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "tag": {"type": "string"},
                },
                "required": ["company", "tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_search",
            "description": "Broaden the search for known fraud/high-risk patterns tied to this metric, when the initial evidence is ambiguous or high-risk.",
            "parameters": {
                "type": "object",
                "properties": {"tag": {"type": "string"}},
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conclude",
            "description": "Call this once you have enough evidence to finish the investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                    "finding": {"type": "string", "description": "3-sentence audit finding summarizing the evidence and conclusion."},
                },
                "required": ["risk_level", "finding"],
            },
        },
    },
]

DISPATCH = {
    "score_company_metric": lambda args: tool_score_company_metric(args["company_id"], args["tag"]),
    "retrieve_similar_cases": lambda args: tool_retrieve_similar_cases(args["company"], args["tag"], args.get("n", 3)),
    "deep_search": lambda args: tool_deep_search(args["tag"]),
}

SYSTEM_PROMPT = """You are an autonomous financial anomaly investigator.

You have tools to score a company's real filing data with a trained VAE
anomaly detection model, retrieve similar historical cases, and search
for broader risk patterns. Decide which tools to call and in what order.

Rules:
- Always call score_company_metric first — you need a real score before
  reasoning about anything.
- Use retrieve_similar_cases to check whether this pattern has precedent.
- Only use deep_search if the score is MEDIUM or HIGH and you need more
  evidence before concluding.
- Call conclude as soon as you have enough evidence. Don't call tools
  you don't need.
"""


def analyze_company(company_id: str, tag: str, ticker: str = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Investigate company_id={company_id} (ticker={ticker}) on metric '{tag}'."},
    ]

    trace = []  # for transparency/debugging - which tools got called, in what order
    score_data = None

    for step in range(MAX_AGENT_STEPS):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = response.choices[0].message
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            # model replied without calling a tool - nudge it back on track
            messages.append({"role": "user", "content": "Please use conclude to finish, or call a tool."})
            continue

        for call in msg.tool_calls:
            fn_name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "conclude":
                final_report = (
                    f"[{args.get('risk_level', 'UNKNOWN')} RISK] {company_id} | Metric: {tag}\n\n"
                    f"FINDING: {args.get('finding', '')}\n\n"
                    f"AGENT TRACE: {' -> '.join(trace) or 'none'}"
                )
                return {
                    "score": score_data.get("reconstruction_error") if score_data else None,
                    "risk_level": args.get("risk_level", "UNKNOWN"),
                    "final_report": final_report,
                    "raw_score_data": score_data,
                    "agent_trace": trace,
                }

            trace.append(fn_name)
            tool_fn = DISPATCH.get(fn_name)
            result = tool_fn(args) if tool_fn else {"error": f"unknown tool {fn_name}"}

            if fn_name == "score_company_metric" and "error" not in result:
                score_data = result

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn_name,
                "content": json.dumps(result),
            })

    # ran out of steps without a conclude call - fail safe with whatever we have
    return {
        "score": score_data.get("reconstruction_error") if score_data else None,
        "risk_level": score_data.get("risk_level", "UNKNOWN") if score_data else "UNKNOWN",
        "final_report": f"Agent did not conclude within {MAX_AGENT_STEPS} steps. Partial evidence: {json.dumps(score_data) if score_data else 'none'}",
        "raw_score_data": score_data,
        "agent_trace": trace,
    }

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict
import json
from app.config.settings import settings

class AnomalyState(TypedDict):
    company: str
    tag: str
    score: float
    risk_level: str
    similar_cases: str
    explanation: str
    final_report: str
    iterations: int
    needs_deep_investigation: bool

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

def tool_retrieve_similar(company, tag, n=3):
    results = collection.query(query_texts=[company + " " + tag], n_results=n)
    docs = results["documents"][0] if results["documents"] else []
    return "\n".join(docs) if docs else "No similar cases found"

def tool_deep_search(tag):
    results = collection.query(query_texts=[tag + " fraud anomaly high risk"], n_results=5)
    docs = results["documents"][0] if results["documents"] else []
    return "\n".join(docs) if docs else "No patterns found"

def assess_risk_node(state):
    state["risk_level"] = "HIGH" if state["score"] > 0.5 else "MEDIUM" if state["score"] > 0.3 else "LOW"
    return state

def retrieve_node(state):
    """Always runs first — retrieve similar cases."""
    state["similar_cases"] = tool_retrieve_similar(state["company"], state["tag"])
    state["iterations"] = state.get("iterations", 0) + 1
    state["needs_deep_investigation"] = True
    return state

def deep_investigate_node(state):
    """LLM decides: deep search or conclude."""
    prompt = f"""You are an autonomous financial fraud investigator.

Investigation so far:
- Company: {state["company"]}
- Metric: {state["tag"]}
- Score: {state["score"]}
- Risk: {state["risk_level"]}
- Similar cases: {state["similar_cases"]}
- Round: {state["iterations"]}

Decide ONE action:
1. Deep search for more patterns → respond: {{"action": "deep_search"}}
2. Conclude investigation → respond: {{"action": "conclude", "finding": "your 3-sentence audit finding here"}}

Respond ONLY with JSON."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])

        if parsed.get("action") == "deep_search":
            extra = tool_deep_search(state["tag"])
            state["similar_cases"] += "\n\nDeep search results:\n" + extra
            state["needs_deep_investigation"] = True
        else:
            state["explanation"] = parsed.get("finding", "")
            state["needs_deep_investigation"] = False
    except Exception:
        state["needs_deep_investigation"] = False

    state["iterations"] = state.get("iterations", 0) + 1
    return state

def write_report_node(state):
    if not state.get("explanation"):
        prompt = f"Company: {state['company']} Metric: {state['tag']} Risk: {state['risk_level']} Evidence: {state['similar_cases']} Write 3 sentence audit recommendation."
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        state["explanation"] = r.choices[0].message.content

    state["final_report"] = (
        f"[{state['risk_level']} RISK] {state['company']} | "
        f"Metric: {state['tag']} | Score: {state['score']:.4f} | "
        f"Agent iterations: {state.get('iterations', 1)}\n\n"
        f"FINDING: {state['explanation']}\n\n"
        f"EVIDENCE:\n{state['similar_cases']}"
    )
    return state

def should_continue(state):
    if state.get("needs_deep_investigation") and state.get("iterations", 0) < 3:
        return "deep_investigate"
    return "report"

graph = StateGraph(AnomalyState)
graph.add_node("assess_risk", assess_risk_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("deep_investigate", deep_investigate_node)
graph.add_node("report", write_report_node)

graph.set_entry_point("assess_risk")
graph.add_edge("assess_risk", "retrieve")
graph.add_edge("retrieve", "deep_investigate")
graph.add_conditional_edges("deep_investigate", should_continue, {
    "deep_investigate": "deep_investigate",
    "report": "report"
})
graph.add_edge("report", END)
agent = graph.compile()

def analyze_company(company, tag, score):
    return agent.invoke({
        "company": company,
        "tag": tag,
        "score": score,
        "risk_level": "",
        "similar_cases": "",
        "explanation": "",
        "final_report": "",
        "iterations": 0,
        "needs_deep_investigation": True
    })

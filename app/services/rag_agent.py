import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.config.settings import settings

class AnomalyState(TypedDict):
    company: str
    tag: str
    score: float
    risk_level: str
    similar_cases: str
    explanation: str
    final_report: str

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

def assess_risk(state):
    state["risk_level"] = "HIGH" if state["score"] > 0.5 else "MEDIUM" if state["score"] > 0.3 else "LOW"
    return state

def retrieve_context(state):
    results = collection.query(query_texts=[state["company"] + " " + state["tag"]], n_results=3)
    state["similar_cases"] = "\n".join(results["documents"][0] if results["documents"] else [])
    return state

def generate_explanation(state):
    prompt = f"Financial fraud analyst. Company: {state['company']} Metric: {state['tag']} Risk: {state['risk_level']} Similar: {state['similar_cases']} In 2 sentences explain what is suspicious."
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    state["explanation"] = response.choices[0].message.content
    return state

def write_report(state):
    state["final_report"] = f"Company: {state['company']} | Score: {state['score']} | Risk: {state['risk_level']} | {state['explanation']}"
    return state

graph = StateGraph(AnomalyState)
graph.add_node("assess_risk", assess_risk)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("generate_explanation", generate_explanation)
graph.add_node("write_report", write_report)
graph.set_entry_point("assess_risk")
graph.add_edge("assess_risk", "retrieve_context")
graph.add_edge("retrieve_context", "generate_explanation")
graph.add_edge("generate_explanation", "write_report")
graph.add_edge("write_report", END)
agent = graph.compile()

def analyze_company(company, tag, score):
    return agent.invoke({"company": company, "tag": tag, "score": score, "risk_level": "", "similar_cases": "", "explanation": "", "final_report": ""})

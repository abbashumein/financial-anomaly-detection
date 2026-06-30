from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(settings.faiss_index_path, embeddings, allow_dangerous_deserialization=True)
groq_client = Groq(api_key=settings.groq_api_key)

def assess_risk(state):
    score = state["score"]
    state["risk_level"] = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.3 else "LOW"
    return state

def retrieve_context(state):
    similar = db.similarity_search(state["company"] + " " + state["tag"], k=3)
    state["similar_cases"] = "\n".join([d.page_content for d in similar])
    return state

def generate_explanation(state):
    prompt = "You are a financial fraud analyst. Answer ONLY based on provided data.\n"
    prompt += "Company: " + state["company"] + "\n"
    prompt += "Metric: " + state["tag"] + "\n"
    prompt += "Risk: " + state["risk_level"] + "\n"
    prompt += "Similar cases: " + state["similar_cases"] + "\n"
    prompt += "In 2 sentences explain what is suspicious and what an auditor should check."
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    state["explanation"] = response.choices[0].message.content
    return state

def write_report(state):
    state["final_report"] = "Company: " + state["company"] + " | Score: " + str(state["score"]) + " | Risk: " + state["risk_level"] + " | " + state["explanation"]
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

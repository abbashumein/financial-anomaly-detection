# app.py
"""
Streamlit frontend for the Agentic Financial Investigation System.

Calls analyze_company() directly (same process) rather than going through
the FastAPI server over HTTP - this keeps the demo to a single process,
which matters for free-tier hosting (Hugging Face Spaces / Streamlit
Community Cloud both run one process per app). The FastAPI app in
app/api/main.py is still the "real" API for programmatic/external use;
this dashboard is the human-facing view of the same underlying agent.
"""
import streamlit as st

from app.services.rag_agent import analyze_company, DEFAULT_METRIC_BASKET
from app.services.cache import cache_stats

st.set_page_config(page_title="Financial Investigation Agent", page_icon="🔎", layout="centered")

st.title("🔎 Agentic Financial Investigation System")
st.markdown(
    "VAE anomaly detection + an LLM agent that **decides what evidence to gather** "
    "from live SEC EDGAR data, rather than a fixed pipeline."
)

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "- **Model**: VAE trained on 285k+ real financial sequences\n"
        "- **Data**: Live SEC EDGAR `companyfacts` API\n"
        "- **Agent**: Groq-hosted LLaMA, tool-calling loop\n"
        "- **Tools**: score, rank anomalous metrics, retrieve similar cases"
    )
    st.subheader("Cache")
    stats = cache_stats()
    st.metric("Live cached entries", stats["live_entries"])
    st.caption("Repeat lookups for the same company reuse cached SEC data instead of re-fetching.")

st.subheader("Investigate a company")

col1, col2 = st.columns([2, 1])
with col1:
    company_id = st.text_input("SEC CIK number", value="0001318605", help="e.g. 0001318605 (Tesla)")
with col2:
    tag = st.selectbox("Metric to investigate", DEFAULT_METRIC_BASKET, index=0)

ticker = st.text_input("Ticker (optional, for display only)", value="")

if st.button("Run investigation", type="primary"):
    with st.spinner("Agent is investigating - scoring, then deciding what evidence to pull next..."):
        try:
            result = analyze_company(company_id, tag, ticker or None)
        except Exception as e:
            st.error(f"Investigation failed: {e}")
            st.stop()

    risk = result.get("risk_level", "UNKNOWN")
    risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(risk, "gray")

    st.markdown(f"### Risk level: :{risk_color}[{risk}]")

    score = result.get("score")
    if score is not None:
        st.metric("Reconstruction error", f"{score:.6f}")

    st.subheader("Agent's investigation steps")
    trace = result.get("agent_trace", [])
    if trace:
        st.write(" → ".join(f"`{t}`" for t in trace))
    else:
        st.write("No tools were called.")

    st.subheader("Final report")
    st.text(result.get("final_report", "No report generated."))

    with st.expander("Raw score data"):
        st.json(result.get("raw_score_data") or {})

st.divider()
st.caption(
    "This is a portfolio project, not investment advice. Anomaly scores reflect "
    "statistical deviation from training-distribution patterns, not confirmed fraud."
)

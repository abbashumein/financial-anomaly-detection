# Financial Anomaly Detection System

> AI-powered fraud detection on real SEC EDGAR financial filings — VAE + RAG + LangGraph + LLM

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-purple)](https://langchain-ai.github.io/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

Reads 22 million rows of real SEC EDGAR financial filings across 6 quarters (2024Q4 – 2026Q1), trains a Variational Autoencoder to learn what normal financial behavior looks like, then automatically flags companies whose patterns deviate — explaining each anomaly in plain English through a LangGraph multi-step agent backed by RAG and Llama 3.3.

---

## Architecture

```
SEC EDGAR (22M rows, 6 quarters)
         │
         ▼
   Polars ETL Pipeline
         │
         ▼
  VAE (PyTorch) ──── learns "normal" financial patterns
         │
         ▼
  Anomaly Scoring ── reconstruction error > 95th percentile
         │
         ▼
  LangGraph Agent (4 nodes)
  ┌─────────────────────────────────────────┐
  │  assess_risk → retrieve_context         │
  │  → generate_explanation → write_report  │
  └─────────────────────────────────────────┘
         │
         ▼
  RAG (FAISS + HuggingFace embeddings)
         │
         ▼
  LLM (Llama 3.3 via Groq / Gemini)
         │
         ▼
  Guardrails ── blocks ungrounded claims
         │
         ▼
  FastAPI (/analyze, /predictions, /health)
         │
         ▼
  SQLite ── stores every prediction
         │
         ▼
  Streamlit UI + LangSmith monitoring
```

---

## Results

| Metric | Value |
|---|---|
| Dataset | SEC EDGAR, 6 quarters, 22M rows |
| Sequences trained | 285,275 |
| Anomalies detected | 14,264 (5%) |
| AUROC | 0.7226 |
| VAE vs Isolation Forest agreement | 94.7% |

### Real Companies Flagged

| Company | Metric | Signal |
|---|---|---|
| MARQUIE GROUP INC | Assets + Liabilities | Sharp drop at period 16-17 — restructuring or write-off |
| CARDIFF LEXINGTON CORP | Discontinued ops EPS | Spike then collapse — sold division, booked one-time gain |
| GIVBUX INC | NonoperatingIncomeExpense | Active 13 quarters then zero — one-time income source disappeared |

---

## Full AI Stack

| Layer | Tool |
|---|---|
| Data processing | Polars |
| Deep learning model | PyTorch VAE |
| Baseline comparison | scikit-learn Isolation Forest |
| Vector database | FAISS |
| Embeddings | HuggingFace sentence-transformers |
| LLM (primary) | Llama 3.3 via Groq |
| LLM (secondary) | Gemini 2.0 Flash |
| Agent orchestration | LangGraph |
| Guardrails | Prompt-constrained generation |
| Monitoring | LangSmith |
| API | FastAPI + Pydantic |
| Database | SQLite |
| Frontend | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Config | pydantic-settings + .env |

---

## Project Structure

```
financial-anomaly-detection/
├── app/
│   ├── api/
│   │   └── main.py          # FastAPI endpoints
│   ├── config/
│   │   └── settings.py      # pydantic-settings, reads .env
│   ├── database/
│   │   └── db.py            # SQLite — stores all predictions
│   ├── models/
│   │   └── vae.py           # VAE architecture + load function
│   └── services/
│       └── rag_agent.py     # LangGraph agent + RAG pipeline
├── app.py                   # Streamlit UI
├── anomaly_detection.ipynb  # Training notebook (Colab)
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## API Endpoints

### POST /analyze
Run the full VAE → RAG → LangGraph pipeline for a company.

**Request:**
```json
{
  "company_id": "MARQUIE GROUP, INC.",
  "ticker": "MQGI",
  "fiscal_year": 2025,
  "fiscal_quarter": "Q4"
}
```

**Response:**
```json
{
  "prediction_id": 1,
  "company_id": "MARQUIE GROUP, INC.",
  "anomaly_score": 0.6790,
  "is_anomaly": true,
  "risk_level": "HIGH",
  "explanation": "The company exhibits suspicious activity across Assets and Liabilities with scores of 0.6790 and 0.3182. An auditor should investigate asset valuation practices and liability disclosures for potential misstatement."
}
```

### GET /predictions
List stored predictions with optional company filter.

### GET /health
Health check — returns `{"status": "ok"}`.

---

## How to Run

### Local

```bash
git clone https://github.com/abbashumein/financial-anomaly-detection
cd financial-anomaly-detection

pip install -r requirements.txt

cp .env.example .env
# fill in your API keys in .env

# Run FastAPI
uvicorn app.api.main:app --reload

# Run Streamlit
streamlit run app.py
```

### Docker

```bash
docker build -t anomaly-detection .
docker run -p 8000:8000 --env-file .env anomaly-detection
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
GEMINI_API_KEY=your_gemini_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-anomaly-detection
```

Get free keys:
- Groq: https://console.groq.com
- LangSmith: https://smith.langchain.com
- Gemini: https://aistudio.google.com

---

## Data

Download SEC EDGAR financial statements (free, no account required):

https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

Quarters used: 2024Q4, 2025Q1, 2025Q2, 2025Q3, 2025Q4, 2026Q1

---

## Design Decisions

**Why VAE over supervised models?**
No labeled fraud dataset exists for SEC filings. VAE learns the latent space of normal financial behavior unsupervised — anything it cannot reconstruct well is flagged as anomalous.

**Why Groq + Llama over OpenAI?**
Open-source model, no per-token cost at development scale, and faster inference. Gemini kept as fallback provider — the codebase supports swapping providers with one parameter.

**Why FAISS over ChromaDB?**
FAISS runs fully in-process with no server dependency. For 14k document embeddings at this scale, it is faster and simpler. ChromaDB would be the upgrade path for a managed vector store.

---

## Author

Ali Abbas — Data Scientist & AI Engineer

[GitHub](https://github.com/abbashumein) | [LinkedIn](https://linkedin.com/in/abbashumein)

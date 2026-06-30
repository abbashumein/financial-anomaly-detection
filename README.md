# Financial Anomaly Detection System

A production-grade AI system that detects suspicious financial patterns in SEC EDGAR filings using a VAE + RAG + LangGraph pipeline, served via FastAPI.

## What It Does

Ingests 22 million rows of real SEC financial filings across 6 quarters (2024Q4-2026Q1), trains a Variational Autoencoder to learn normal financial behavior, flags anomalous companies, and explains each anomaly in plain English using RAG + LLaMA 3.3.

## Results

| Metric | Value |
|---|---|
| Sequences processed | 285,275 |
| Anomalies detected | 14,264 (5%) |
| AUROC | 0.7226 |
| VAE vs Isolation Forest agreement | 94.7% |

## Real Companies Flagged

- **MARQUIE GROUP INC** - Assets and liabilities dropped sharply (restructuring signal)
- **CARDIFF LEXINGTON CORP** - Discontinued operations spike then collapse
- **GIVBUX INC** - Nonoperating income disappeared after 13 quarters

## Tech Stack

| Layer | Technology |
|---|---|
| Data | SEC EDGAR, Polars |
| Model | Variational Autoencoder (PyTorch) |
| Baseline | Isolation Forest (scikit-learn) |
| RAG | FAISS + HuggingFace sentence-transformers |
| LLM | LLaMA 3.3-70b via Groq |
| Agent | LangGraph (4-node pipeline) |
| Guardrails | Prompt-based hallucination prevention |
| Monitoring | LangSmith |
| API | FastAPI + Uvicorn |
| Storage | SQLite |
| UI | Streamlit |
| Container | Docker |

## Project Structure
financial-anomaly-detection/
├── app/
│   ├── api/main.py
│   ├── config/settings.py
│   ├── database/db.py
│   ├── models/vae.py
│   └── services/rag_agent.py
├── app.py
├── anomaly_detection.ipynb
├── Dockerfile
└── requirements.txt

## API Usage

POST /analyze

```json
{
  "company_id": "0001318605",
  "ticker": "TSLA",
  "fiscal_year": 2024,
  "fiscal_quarter": "Q4"
}
```

Response:

```json
{
  "prediction_id": 1,
  "company_id": "0001318605",
  "anomaly_score": 0.847,
  "is_anomaly": true,
  "risk_level": "HIGH",
  "explanation": "Unusual spike in nonoperating income relative to 6-quarter baseline..."
}
```

## Setup

```bash
git clone https://github.com/abbashumein/financial-anomaly-detection
cd financial-anomaly-detection
pip install -r requirements.txt
streamlit run app.py
uvicorn app.api.main:app --reload
```

## Data

SEC EDGAR: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

## Author

Ali Abbas - AI/ML Engineer
GitHub: https://github.com/abbashumein
LinkedIn: https://linkedin.com/in/ali-abbas-0b6894223

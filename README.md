# Financial Anomaly Detection System

This project is evolving from a research notebook into a production-grade AI system.

Current stage:
- VAE-based anomaly detection implemented
- SEC EDGAR financial data pipeline working
- Basic anomaly scoring completed

Next stage:
- FastAPI inference service
- RAG-based explanation system
- AI agent for multi-step fraud reasoning
- Docker + deployment


## What It Does
Reads 22 million rows of real SEC financial filings across 6 quarters (2024Q4–2026Q1), trains a Variational Autoencoder to learn normal financial behavior, and automatically flags companies with suspicious patterns — explaining each anomaly in plain English using RAG and LLaMA 3.3.

## Results
- 285,275 financial time-series sequences processed
- 14,264 anomalies detected (5%)
- AUROC: 0.7226
- 94.7% agreement between VAE and Isolation Forest

## Real Companies Flagged
- MARQUIE GROUP INC — Assets and Liabilities dropped sharply (restructuring signal)
- CARDIFF LEXINGTON CORP — Discontinued operations spike then collapse
- GIVBUX INC — Nonoperating income disappeared after 13 quarters

## Full AI Stack
- Data: SEC EDGAR (6 quarters, 22M rows)
- Model: Variational Autoencoder (PyTorch)
- Baseline: Isolation Forest (scikit-learn)
- RAG: FAISS + HuggingFace sentence-transformers
- LLM: LLaMA 3.3 via Groq
- Agent: LangGraph (4-step reasoning pipeline)
- Guardrails: Prompt-based hallucination prevention
- Monitoring: LangSmith
- App: Streamlit
- Data Processing: Polars

## How To Run
```bash
git clone https://github.com/abbashumein/financial-anomaly-detection
cd financial-anomaly-detection
pip install -r requirements.txt
streamlit run app.py
```

## Data
Download SEC EDGAR data free from:
https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

## Author
Ali Abbas | Data Scientist & AI Engineer

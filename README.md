# Financial Anomaly Detection System

A production-grade autonomous AI system that detects suspicious financial patterns in SEC EDGAR filings using VAE + ChromaDB RAG + LangGraph autonomous agent, served via FastAPI on Azure.

## Live API
**Base URL:** https://financial-anomaly.whitemushroom-bdf53e45.eastus.azurecontainerapps.io

- Swagger UI: /docs
- Health: /health
- Analyze: POST /analyze
- Predictions: GET /predictions

## Performance Metrics

| Metric | Value |
|---|---|
| VAE sequences trained | 285,275 |
| Anomalies detected | 14,264 (5%) |
| AUROC | 0.7226 |
| VAE vs Isolation Forest agreement | 94.7% |
| ChromaDB records | 500 |
| Agent iterations per query | 3 |
| Deployment | Azure Container Apps |

## Real Companies Flagged

- **MARQUIE GROUP, INC.** - Assets and liabilities dropped sharply (restructuring signal)
- **CARDIFF LEXINGTON CORP** - Discontinued operations spike then collapse
- **GIVBUX INC** - Nonoperating income disappeared after 13 quarters

## Architecture


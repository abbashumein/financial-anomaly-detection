# Financial Anomaly Detection

This is an AI system, not just a model: a tool-calling LLM agent (Groq-hosted, currently openai/gpt-oss-120b) that investigates a company's SEC filings in real time, deciding for itself which of 6 tools to call, in what order, and when it has enough evidence to conclude.

At the core of the agent's toolset is a Variational Autoencoder, trained offline on 285,000+ real financial sequences from SEC EDGAR, which learned what "normal" financial reporting looks like for a given metric. No labeled fraud dataset exists for SEC filings — the model has to learn structure from the data itself, and anything it reconstructs poorly gets flagged as statistically unusual.

The deployed API wires all of this together live, not from a cache. Given a company's SEC CIK and a financial metric (e.g. Assets, NetIncomeLoss), the agent can:

Score the metric with a real VAE forward pass on live SEC EDGAR data
Rank which specific metrics are driving an anomaly (get_anomalous_metrics)
Pull real supporting text from the company's actual 10-K/10-Q filings (get_sec_filing_context)
Compare the company against sector peers using the same VAE (compare_to_peers)
Distinguish a sudden one-time spike from gradual drift (get_historical_trend)
Check for multi-hop connections — shared subsidiaries, related entities — via a small knowledge graph (graph_investigate)

Retrieval itself isn't blind top-k search: candidates are metadata-filtered, combined from semantic + keyword (BM25) search, then reranked with a CrossEncoder before reaching the LLM.

That decision-making is the actual "agentic" part: the agent isn't following a fixed script — it calls a different number and combination of tools per investigation, based on how strong the evidence looks at each step.

## Architecture

```
SEC EDGAR APIs (live, per-request, no API key)
  - companyfacts (structured financial data)
  - full-text search (real filing text + industry/SIC lookup)
         │
         ▼
  In-memory TTL cache (15 min) ── same company across multiple
                                   tags/tools hits SEC once, not N times
         │
         ▼
  Sequence builder ── window-bounded to match training shape,
                       min-max scaled, zero-padded to length 20
         │
         ▼
  VAE (PyTorch) ──── real forward pass → reconstruction error
         │
         ▼
  Risk bucket ── calibrated against real training-set percentiles
                  (p90 = 0.087, p95 = 0.105, from 285,275 sequences)
         │
         ▼
  Tool-Calling Agent (Groq-hosted LLM, configurable model)
  ┌──────────────────────────────────────────────────────────┐
  │  agent decides which tool to call next, and when          │
  │  it has enough evidence to stop:                          │
  │                                                            │
  │  score_company_metric    → always called first             │
  │  get_anomalous_metrics   → which metrics drive the anomaly │
  │  get_sec_filing_context  → real evidence from 10-K/10-Q    │
  │  compare_to_peers        → same-industry comparison         │
  │  get_historical_trend    → sudden spike vs gradual drift    │
  │  retrieve_similar_cases  → historical precedent (see below) │
  │  graph_investigate       → optional multi-hop connections   │
  │  conclude                → ends the investigation           │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
  Advanced RAG ── metadata filtering (same metric tag) →
                  hybrid search (Chroma semantic + BM25 keyword) →
                  CrossEncoder reranking → top-N to the LLM
         │
         ▼
  GraphRAG (networkx, in-memory) ── deterministic edges from the
                  tools above + one cached LLM extraction per unique
                  filing excerpt → multi-hop traversal
         │
         ▼
  FastAPI (/analyze [API-key + rate-limited], /health, /cache-stats)
         │
         ▼
  SQLite ── every prediction persisted for audit history
```

## Offline Model Training Results

| Metric | Value |
|---|---|
| Dataset | SEC EDGAR bulk financial statement data, 6 quarters, 22M rows |
| Sequences trained on | 285,275 |
| Anomalies flagged (p95 threshold) | 14,264 (5%) |
| AUROC (vs. Isolation Forest as a weak proxy label) | 0.7226 |
| VAE / Isolation Forest agreement | 94.7% |

**Caveat on these numbers:** there's no ground-truth fraud label for SEC filings, so AUROC is measured against Isolation Forest's flags as a proxy, not verified fraud — a consistency check between two unsupervised methods, not a precision/recall claim.

**Companies the offline model flagged during training:**

| Company | Metric | Signal |
|---|---|---|
| The Marquie Group, Inc. | Assets, Liabilities | Sharp value shift mid-sequence — restructuring or write-off pattern |
| Cardiff Lexington Corp | Discontinued ops EPS | Spike then collapse — consistent with a one-time divestiture gain |
| GivBux, Inc. | Non-operating income/expense | Active for several quarters then drops to zero — income source disappeared |

*(Note: re-querying live EDGAR for The Marquie Group's CIK now returns the entity name "Transglobal Management Group, Inc." — small-cap companies rename/restructure often; this may be the same legal entity under a new name, unconfirmed.)*

## Live Scoring: Design Decisions and Known Limitations

Being upfront about these rather than glossing over them, because they're the kind of thing a technical interview digs into:

- **Window-bounded live scoring.** Pulling a company's full multi-year EDGAR history initially caused stable companies (Apple, Tesla) to falsely score HIGH, because training sequences were short and heavily zero-padded. Live scoring now bounds to the same window shape the model trained on.
- **The model only sees a company's most recent reported values** for a metric — an anomaly outside that rolling window isn't visible to it. Production would need a wider window or periodic retraining.
- **Per-sequence min-max scaling discards absolute magnitude** — the model sees relative shape, not scale, within one sequence. Confirmed with a synthetic 8x-spike test that scored near the typical range instead of flagging as anomalous.
- **RAG corpus is a 500-record sample**, not the full 285,275-sequence set.
- **Single-metric scoring per call** — the agent investigates one financial tag at a time, not a cross-metric picture.
- **No reranking or query rewriting** in the RAG step — intentionally simple retrieval, not a hidden GraphRAG pipeline.

## Tech Stack

| Layer | Tool |
|---|---|
| Agent orchestration | Hand-rolled tool-calling loop (Groq function-calling API) |
| LLM | Llama 3.3 via Groq |
| Deep learning model | PyTorch VAE |
| Live data source | SEC EDGAR `companyfacts` API (public, free, no key) |
| Vector database | ChromaDB (local persistent store) |
| Embeddings | Sentence-transformer (ChromaDB default, ONNX MiniLM) |
| Data processing (offline) | Polars |
| Baseline comparison (offline) | scikit-learn Isolation Forest |
| API | FastAPI + Pydantic |
| Database | SQLite |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Config | pydantic-settings + `.env` |

## Project Structure

```
financial-anomaly-detection/
├── app/
│   ├── api/main.py            # FastAPI endpoints
│   ├── config/settings.py     # pydantic-settings, reads .env
│   ├── database/db.py         # SQLite — stores all predictions
│   ├── models/vae.py          # VAE architecture + load function
│   └── services/
│       ├── edgar_client.py    # live SEC EDGAR fetch + sequence builder
│       ├── vae_scorer.py      # loads trained weights, runs live scoring
│       └── rag_agent.py       # tool-calling agent + ChromaDB retrieval
├── anomaly_detection.ipynb    # offline training notebook
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## API Reference

### `POST /analyze`

Runs the live VAE-scoring + agentic RAG pipeline for a company/metric pair.

**Request:**
```json
{
  "company_id": "0001318605",
  "tag": "Assets",
  "ticker": "TSLA"
}
```
`company_id` is the company's SEC CIK number. `tag` is any US-GAAP tag the company discloses (e.g. `Assets`, `Revenues`, `NetIncomeLoss`). `ticker` is optional.

**Response:**
```json
{
  "prediction_id": 3,
  "company_id": "0001318605",
  "anomaly_score": 0.05192,
  "is_anomaly": false,
  "risk_level": "LOW",
  "explanation": "[LOW RISK] 0001318605 | Metric: Assets\n\nFINDING: ...\n\nAGENT TRACE: score_company_metric -> retrieve_similar_cases"
}
```

The `explanation` field includes an `AGENT TRACE` showing exactly which tools the agent called and in what order.

### `GET /predictions`
List stored predictions, optional company filter.

### `GET /health`
Health check — returns `{"status": "ok"}`.

## How to Run

### Local
```bash
git clone https://github.com/abbashumein/financial-anomaly-detection
cd financial-anomaly-detection

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY and EDGAR_USER_AGENT

uvicorn app.api.main:app --reload
```

### Docker
```bash
docker build -t anomaly-detection .
docker run -p 8000:8000 --env-file .env anomaly-detection
```

## Environment Variables

```
GROQ_API_KEY=your_groq_key_here        # required — powers the tool-calling agent
EDGAR_USER_AGENT=your-name your-app/1.0 (your_email@example.com)   # required — SEC rejects requests without a descriptive User-Agent
```

Get a free Groq key: https://console.groq.com

## Training Data

Offline training used SEC EDGAR's public bulk financial statement datasets: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

Quarters used: 2024Q4, 2025Q1, 2025Q2, 2025Q3, 2025Q4, 2026Q1

## Design Decisions

**Why a VAE?** No labeled fraud dataset exists for SEC filings, so this had to be unsupervised. The tradeoff: it flags statistically unusual, not fraudulent — a legitimately fast-growing company can score like a genuinely suspicious one. That's why the agent retrieves precedent and pattern-searches instead of treating the raw VAE score as a verdict.

**Why Groq + Llama over OpenAI?** Open-source model, no per-token cost at development scale, fast inference — good fit for a portfolio project with real usage during interviews/demos.

**Why a hand-rolled tool-calling loop instead of LangGraph?** An earlier version used a fixed LangGraph pipeline with a hardcoded input score — the VAE was never actually called at inference time. Rebuilding as a direct tool-calling loop against Groq's function-calling API made the control flow fully visible and let the LLM genuinely decide the investigation path, rather than hiding that decision inside a framework abstraction.

**Why ChromaDB over FAISS?** Runs fully in-process with a local persistent store, no separate index-management step — simpler for a project at this scale.

## Author

Ali Abbas — AI/ML Engineer & Data Specialist

[GitHub](https://github.com/abbashumein) | [LinkedIn](https://linkedin.com/in/ali-abbas-0b6894223/)

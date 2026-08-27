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
| LLM | Groq-hosted (configurable via `GROQ_MODEL`, default `openai/gpt-oss-120b`) |
| Deep learning model | PyTorch VAE |
| Live data source | SEC EDGAR `companyfacts` API + full-text search API (public, free, no key) |
| Vector database | ChromaDB (local persistent store) |
| Embeddings | Sentence-transformer (ChromaDB default, ONNX MiniLM) |
| Keyword search | BM25 (`rank_bm25`) — combined with embeddings for hybrid retrieval |
| Reranking | CrossEncoder (`sentence-transformers`, `ms-marco-MiniLM-L-6-v2`) |
| Knowledge graph | NetworkX (in-memory) — multi-hop entity/relationship traversal |
| Data processing (offline) | Polars |
| Baseline comparison (offline) | scikit-learn Isolation Forest |
| API | FastAPI + Pydantic |
| Auth & rate limiting | Custom API-key dependency + SlowAPI |
| Caching | In-memory TTL cache (SEC API calls) |
| Database | SQLite |
| Frontend | Streamlit |
| Testing | pytest (39 tests — unit, integration, regression) |
| Containerization | Docker |
| CI/CD | GitHub Actions (runs full test suite on every push) |
| Config | pydantic-settings + `.env` |

## Project Structure

```
financial-anomaly-detection/
├── app/
│   ├── api/main.py                 # FastAPI endpoints (auth + rate-limited)
│   ├── config/settings.py          # pydantic-settings, reads .env
│   ├── core/security.py            # API-key auth dependency
│   ├── database/db.py              # SQLite — stores all predictions
│   ├── models/vae.py               # VAE architecture + load function
│   └── services/
│       ├── edgar_client.py         # live SEC EDGAR fetch + sequence builder
│       ├── vae_scorer.py           # loads trained weights, runs live scoring
│       ├── cache.py                # in-memory TTL cache for SEC API calls
│       ├── sec_filing_search.py    # real 10-K/10-Q filing text search
│       ├── peer_comparison.py      # industry (SIC) peer lookup + scoring
│       ├── advanced_retrieval.py   # metadata filter + hybrid search + rerank
│       ├── graph_rag.py            # in-memory knowledge graph, multi-hop
│       └── rag_agent.py            # tool-calling agent (6 tools) + retrieval
├── tests/                          # 39 tests: unit, integration, regression
│   ├── conftest.py
│   ├── _stubs/                     # lightweight fallbacks if heavy deps absent
│   └── test_*.py
├── app.py                          # Streamlit frontend
├── anomaly_detection.ipynb         # offline training notebook
├── .github/workflows/deploy.yml    # CI — runs tests on every push
├── .env.example
├── Dockerfile
├── requirements.txt                # pinned versions
└── .gitignore
```

## Problem

SEC filings are public, but nobody actually reads them at scale. Regulators, analysts, and auditors rely on humans manually flagging "this number looks weird" — a process that doesn't scale past a handful of companies and misses subtle statistical anomalies a human wouldn't catch by eye.

There's no labeled "fraud" dataset for SEC filings to train on — fraud is rare, disclosed inconsistently, and legally sensitive. So this project treats it as an unsupervised problem: learn what *normal* financial reporting looks like, and flag anything a trained model can't reconstruct well as statistically unusual — then have an AI agent investigate *why*, the way a human analyst would.


## What This Achieves

- A **real trained model** (VAE, not a heuristic) that scores live company data against a learned baseline of normal financial behavior
- An **agent that investigates, not just scores** — given a flagged company, it autonomously decides whether to check which specific metric is driving the anomaly, pull real filing text as evidence, compare against industry peers, check if the change was sudden or gradual, and search for historical precedent — the same steps a human analyst would take, but decided by the LLM per-investigation, not hardcoded
- **Advanced retrieval, not blind search** — metadata filtering, hybrid (semantic + keyword) search, and CrossEncoder reranking, so retrieved evidence is actually relevant, not just "closest by embedding"
- **A small knowledge graph layer** enabling multi-hop questions flat retrieval can't answer, like "is this company connected to another flagged company through a shared subsidiary?"
- **Production-adjacent engineering**: 39 automated tests, CI that runs them on every push, pinned dependencies, API auth + rate limiting, caching, and a working frontend — not just a notebook


## Known Limitations

- **No labeled fraud ground truth.** The VAE flags statistical deviation from normal patterns, not confirmed fraud — a HIGH score means "unusual," not "fraudulent." There's no dataset of confirmed SEC fraud cases to validate detection accuracy against.
- **Moderate AUROC (0.72)** against a proxy Isolation Forest baseline, not a true fraud label — disclosed here rather than hidden, since it's a realistic number for an unsupervised approach on this kind of data.
- **Not horizontally scalable as-is.** The cache and knowledge graph are in-memory and per-process — they reset on restart and aren't shared across multiple server instances. Fine for a single-process portfolio deployment; would need Redis + a real graph database for multi-instance production use.
- **First-request latency.** The CrossEncoder reranker and BM25 index both build/download on first use, adding a few seconds to the very first investigation after startup. Subsequent requests are fast.
- **Peer discovery depends on SEC's full-text search matching well.** `compare_to_peers` finds companies in the same industry (SIC code) that also mention the relevant financial term — it can occasionally miss true peers or surface a loose match if the search term is too generic.
- **Single shared API key, no per-user accounts.** Auth is a single `X-API-Key` check, appropriate for a portfolio demo but not multi-tenant production use.
- **SQLite, not a production database.** Fine for single-instance prediction logging; would need Postgres or similar under real concurrent write load.
- **Free-tier Groq rate limits apply.** Heavy usage could hit Groq's free-tier request limits; there's no fallback LLM provider configured.
- **Investigation latency.** A full multi-tool investigation (score → rank → filing search → peers → trend → graph) can take 10-30+ seconds depending on how many tools the agent chooses to call — there's no streaming response yet.

## Performance

**Model calibration** (against 285,275 real training sequences):
| Metric | Value |
|---|---|
| AUROC (vs. Isolation Forest proxy baseline) | 0.7226 |
| p50 (typical reconstruction error) | 0.0438 |
| p90 threshold | 0.0867 |
| p95 (anomaly threshold) | 0.1052 |


**System efficiency** (measured, not estimated):

| Metric | Value |
|---|---|
| SEC API calls per company investigation | 1, cached — down from 5 (one per metric) before caching was added |
| Test suite size | 39 tests |
| Test suite runtime | ~7-30s (varies with cold-start model loading) |
| CI | Runs full suite on every push via GitHub Actions |

**Note:** there is no precision/recall/F1 reported here, because there's no labeled ground-truth fraud dataset to compute them against (see Known Limitations) — AUROC against a proxy baseline is the honest number available for this kind of unsupervised problem.


## Engineering Challenges & How They Were Solved


| Challenge | What went wrong | Fix |
|---|---|---|
| **Distribution mismatch** | Live inference pulled a company's *full* filing history, but the VAE was trained on short, fixed-length windows — scores were meaningless | Rebuilt the sequence pipeline to window live data to match the exact shape the model was trained on |
| **Repeated SEC API calls** | Investigating one company across 5 metrics triggered 5 separate SEC API calls for the same underlying data | Added a 15-minute in-memory TTL cache on the network layer — same company now costs 1 API call, not 5 |
| **Blind retrieval** | The original RAG only did plain semantic similarity search — no filtering, so a Revenue anomaly could get compared against unrelated Liabilities cases | Rebuilt as a 3-stage pipeline: metadata filtering → hybrid (semantic + BM25 keyword) search → CrossEncoder reranking |
| **Silent test-suite gap** | The historical-cases index only got built the *first* time the app ran (empty-database check) — meant it silently didn't exist on every subsequent run | Refactored so the index always builds from the full dataset, regardless of whether ingestion already happened |
| **Model deprecation, mid-project** | Groq deprecated `llama-3.3-70b-versatile` during development — the agent returned 404s with no warning | Replaced the hardcoded model name with a `GROQ_MODEL` setting, so future deprecations are a one-line `.env` change, not a code hunt |
| **API protocol mismatch after model swap** | The new model rejected conversation turns where a prior assistant message had `"tool_calls": null` — Groq's stricter validation caught what the old model silently tolerated | Fixed message construction to omit the `tool_calls` key entirely when the LLM has no tool call, instead of setting it to `null` |
| **Order-dependent test failures** | Auth tests passed alone but failed in the full suite — a different test file imported the settings module first, locking in an empty API key before the auth test could set it | Moved required test environment variables into `conftest.py`, which always runs before test collection, regardless of file order |


## Author

Ali Abbas — AI/ML Engineer & Data Specialist

[GitHub](https://github.com/abbashumein) | [LinkedIn](https://linkedin.com/in/ali-abbas-0b6894223/)

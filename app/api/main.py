# app/api/main.py
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.services.rag_agent import analyze_company
from app.services.cache import cache_stats
from app.database.db import init_db, insert_prediction, get_predictions
from app.core.security import require_api_key

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Financial Anomaly Detection API",
    version="2.0.0",
    description="VAE + Agentic RAG pipeline for SEC EDGAR anomaly investigation",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # portfolio demo: open by design so recruiters can hit it
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()

# ---------- schemas ----------

class AnalyzeRequest(BaseModel):
    company_id: str = Field(..., example="0001318605", description="SEC CIK number")
    tag: str = Field(..., example="Assets", description="US-GAAP tag to investigate, e.g. Assets, Revenues, NetIncomeLoss")
    ticker: Optional[str] = Field(None, example="TSLA")
    fiscal_year: Optional[int] = Field(None, example=2023)
    fiscal_quarter: Optional[str] = Field(None, example="Q4")

class AnalyzeResponse(BaseModel):
    prediction_id: int
    company_id: str
    anomaly_score: float
    is_anomaly: bool
    risk_level: str
    explanation: str

# ---------- routes ----------

@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
async def analyze(request: Request, req: AnalyzeRequest):
    """Run the live VAE-scoring + agentic RAG pipeline for a given company/tag."""
    try:
        result = analyze_company(req.company_id, req.tag, req.ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    score = result.get("score")
    raw = result.get("raw_score_data") or {}
    # Prefer the VAE scorer's own is_anomaly flag (calibrated against the
    # real p95 threshold); fall back to risk_level if scoring failed for
    # some reason so the response never silently lies.
    is_anomaly = raw.get("is_anomaly", result.get("risk_level") == "HIGH")

    record = {
        "company_id":     req.company_id,
        "ticker":         req.ticker,
        "fiscal_year":    req.fiscal_year,
        "fiscal_quarter": req.fiscal_quarter,
        "anomaly_score":  score if score is not None else 0.0,
        "is_anomaly":     int(is_anomaly),
        "risk_level":     result.get("risk_level", "UNKNOWN"),
        "explanation":    result.get("final_report", ""),
        "raw_metrics":    raw,
    }
    pred_id = insert_prediction(record)

    return AnalyzeResponse(
        prediction_id=pred_id,
        company_id=req.company_id,
        anomaly_score=score if score is not None else 0.0,
        is_anomaly=is_anomaly,
        risk_level=result.get("risk_level", "UNKNOWN"),
        explanation=result.get("final_report", ""),
    )

@app.get("/predictions")
async def list_predictions(
    company_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    return get_predictions(company_id=company_id, limit=limit)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/cache-stats")
async def cache_stats_endpoint():
    """How much is currently cached - useful for demoing that caching works."""
    return cache_stats()
# app/services/peer_comparison.py
"""
Finds sector peers for a company and compares anomaly scores against
them - "is this unusual, or is this just how this industry looks?"

Two SEC endpoints, both free/no-key:
  1. data.sec.gov/submissions/CIK{10digit}.json - gives us the target
     company's SIC (industry) code.
  2. efts.sec.gov/LATEST/search-index - the SAME full-text search API
     used in sec_filing_search.py, which also supports filtering by
     `sics=` - so we reuse it to find OTHER companies in that industry,
     instead of building a second, separate "peer lookup" system.

Actual peer SCORING reuses tool_score_company_metric as-is (imported
from rag_agent at call time to avoid a circular import) - no new model,
no new math, same pattern as get_anomalous_metrics.
"""
import requests

from app.config.settings import settings
from app.services.cache import ttl_cache
from app.services.edgar_client import normalize_cik, EdgarLookupError
from app.services.sec_filing_search import SEARCH_URL, TAG_TO_SEARCH_TERM

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _user_agent() -> str:
    ua = (settings.edgar_user_agent or "").strip()
    if not ua:
        raise EdgarLookupError("EDGAR_USER_AGENT is not set - required by SEC on every request.")
    return ua


@ttl_cache(seconds=900)
def get_company_sic(company_id: str) -> dict:
    """Returns {'sic': '3711', 'sic_description': 'Motor Vehicles...', 'entity_name': ...}"""
    cik = normalize_cik(company_id)
    resp = requests.get(
        SUBMISSIONS_URL.format(cik=cik),
        headers={"User-Agent": _user_agent()},
        timeout=15,
    )
    if resp.status_code == 404:
        raise EdgarLookupError(f"No SEC record found for CIK {cik}.")
    resp.raise_for_status()
    data = resp.json()

    sic = data.get("sic")
    if not sic:
        raise EdgarLookupError(f"CIK {cik} has no SIC code on file.")

    return {
        "sic": sic,
        "sic_description": data.get("sicDescription", "Unknown industry"),
        "entity_name": data.get("name", company_id),
    }


@ttl_cache(seconds=900)
def find_peer_ciks(sic: str, exclude_cik: str, search_term: str, limit: int = 3) -> list[dict]:
    """Find other companies (different CIK) in the same SIC industry that
    have recent 10-K/10-Q filings mentioning search_term."""
    params = {
        "q": f'"{search_term}"',
        "forms": "10-K,10-Q",
        "sics": sic,
    }
    resp = requests.get(SEARCH_URL, params=params, headers={"User-Agent": _user_agent()}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("hits", {}).get("hits", [])
    seen_ciks = {exclude_cik}
    peers = []

    for hit in hits:
        source = hit.get("_source", {})
        hit_ciks = source.get("ciks", [])
        if not hit_ciks:
            continue
        cik = hit_ciks[0]
        if cik in seen_ciks:
            continue
        seen_ciks.add(cik)
        peers.append({
            "cik": cik,
            "entity_name": (source.get("display_names") or [cik])[0],
        })
        if len(peers) >= limit:
            break

    return peers

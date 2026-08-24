# app/services/sec_filing_search.py
"""
Searches the actual TEXT of SEC filings (10-K/10-Q narrative sections),
as opposed to edgar_client.py which pulls structured numeric facts.

Uses SEC's real, free, no-key full-text search API:
    https://efts.sec.gov/LATEST/search-index

This is a different SEC system than data.sec.gov/api/xbrl/companyfacts -
that one gives numbers, this one gives you which actual filing documents
mention a topic, plus a link to the raw filing so we can pull the
surrounding narrative text as evidence.

Verified response shape (real query run against the live endpoint):
{
  "hits": {
    "hits": [
      {
        "_id": "<accession-with-dashes>:<filename>.htm",
        "_source": {
          "ciks": ["0001318605"],
          "display_names": ["Tesla, Inc.  (TSLA)  (CIK 0001318605)"],
          "form": "10-K",
          "adsh": "0001318605-24-000010",
          "file_date": "2024-03-29",
          ...
        }
      },
      ...
    ]
  }
}
"""
import re
import requests

from app.config.settings import settings
from app.services.cache import ttl_cache
from app.services.edgar_client import normalize_cik, EdgarLookupError

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{filename}"

# Maps our VAE metric tags to plain-English search terms - the SEC full
# text search is keyword-based, not tag-based, so "NetIncomeLoss" (an
# XBRL tag) needs to become "net income" (words a filing actually uses).
TAG_TO_SEARCH_TERM = {
    "Assets": "total assets",
    "Liabilities": "total liabilities",
    "StockholdersEquity": "stockholders equity",
    "Revenues": "revenue recognition",
    "NetIncomeLoss": "net income",
}


def _user_agent() -> str:
    ua = (settings.edgar_user_agent or "").strip()
    if not ua:
        raise EdgarLookupError("EDGAR_USER_AGENT is not set - required by SEC on every request.")
    return ua


@ttl_cache(seconds=900)
def search_filings(company_id: str, search_term: str, forms=("10-K", "10-Q"), limit: int = 3) -> list[dict]:
    """Find recent filings for this company whose text mentions search_term."""
    cik = normalize_cik(company_id)
    params = {
        "q": f'"{search_term}"',
        "forms": ",".join(forms),
        "ciks": cik,
    }
    resp = requests.get(SEARCH_URL, params=params, headers={"User-Agent": _user_agent()}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("hits", {}).get("hits", [])[:limit]
    results = []
    for hit in hits:
        source = hit.get("_source", {})
        doc_id = hit.get("_id", "")
        if ":" not in doc_id:
            continue
        accession, filename = doc_id.split(":", 1)
        accession_no_dashes = accession.replace("-", "")
        cik_int = str(int(cik))  # archive URLs use the CIK without leading zeros

        results.append({
            "entity_name": (source.get("display_names") or [company_id])[0],
            "form": source.get("form"),
            "file_date": source.get("file_date"),
            "doc_url": ARCHIVE_URL.format(cik_int=cik_int, accession_no_dashes=accession_no_dashes, filename=filename),
        })
    return results


def _strip_html(html: str) -> str:
    """Lightweight tag stripper - good enough for pulling readable
    paragraph text out of a filing; avoids adding a full HTML-parsing
    dependency for what's fundamentally a text-search feature."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_excerpt(doc_url: str, search_term: str, context_chars: int = 400) -> str | None:
    """Fetch the filing document and pull the text window around the
    first mention of search_term. Returns None if the term isn't
    actually found in the fetched text (can happen - full-text search
    matches on the whole submission including exhibits, not just the
    primary document)."""
    resp = requests.get(doc_url, headers={"User-Agent": _user_agent()}, timeout=20)
    resp.raise_for_status()
    text = _strip_html(resp.text)

    idx = text.lower().find(search_term.lower())
    if idx == -1:
        return None

    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(search_term) + context_chars)
    excerpt = text[start:end].strip()
    return f"...{excerpt}..."

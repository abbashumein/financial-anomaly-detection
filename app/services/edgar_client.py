# app/services/edgar_client.py
"""
Live SEC EDGAR client.

Replaces the offline bulk-file pipeline (num.txt/sub.txt/tag.txt from
SEC's quarterly DERA datasets) that the training notebook used, with the
free, no-key, real-time `companyfacts` API. Given a CIK and a US-GAAP tag,
this pulls that company's actual disclosed history and reshapes it into
the exact 20-length, min-max-scaled, zero-padded sequence format the VAE
was trained on (see anomaly_detection.ipynb, Step 4).

SEC requires a descriptive User-Agent on every request (no API key needed).
Set EDGAR_USER_AGENT in your .env, e.g.:
    EDGAR_USER_AGENT="ali-cdss-project/1.0 ([email protected])"
"""
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from app.config.settings import settings

MAX_LEN = 20  # must match VAE(seq_len=20)
BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


class EdgarLookupError(Exception):
    """Raised when a CIK/tag combination can't be resolved to live data."""


def _user_agent() -> str:
    ua = (settings.edgar_user_agent or "").strip()
    if not ua:
        raise EdgarLookupError(
            "EDGAR_USER_AGENT is not set. SEC requires a descriptive "
            "User-Agent (e.g. 'yourapp/1.0 ([email protected])') on every "
            "request or it will reject the call."
        )
    return ua


def normalize_cik(company_id: str) -> str:
    """Accepts '1318605', '0001318605', or 'CIK0001318605' -> '0001318605'."""
    digits = "".join(ch for ch in company_id if ch.isdigit())
    if not digits:
        raise EdgarLookupError(f"'{company_id}' doesn't contain a CIK number.")
    return digits.zfill(10)


def fetch_company_facts(company_id: str) -> dict:
    cik = normalize_cik(company_id)
    url = BASE_URL.format(cik=cik)
    resp = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=15)
    if resp.status_code == 404:
        raise EdgarLookupError(f"No EDGAR filings found for CIK {cik}.")
    resp.raise_for_status()
    return resp.json()


def build_sequence(facts: dict, tag: str, max_len: int = MAX_LEN):
    """
    Mirrors the notebook's Step 4 exactly:
      - pull the tag's USD-unit values, sorted chronologically
      - trim to the most recent `max_len` points
      - min-max scale to [0, 1]
      - zero-pad on the right if shorter than max_len

    Returns (padded_array[float32, shape=(max_len,)], raw_values, dates, unit_used)
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    if tag not in gaap:
        available = sorted(gaap.keys())
        raise EdgarLookupError(
            f"Tag '{tag}' not disclosed by this company. "
            f"{len(available)} tags available, e.g.: {available[:8]}"
        )

    units = gaap[tag].get("units", {})
    unit_key = "USD" if "USD" in units else next(iter(units), None)
    if not unit_key:
        raise EdgarLookupError(f"Tag '{tag}' has no usable unit data.")

    points = units[unit_key]
    # de-dupe same-period restatements: keep the latest filed value per end date
    by_end = {}
    for p in points:
        if "end" in p and "val" in p:
            by_end[p["end"]] = p
    ordered = sorted(by_end.values(), key=lambda p: p["end"])

    if not ordered:
        raise EdgarLookupError(f"Tag '{tag}' has no dated values.")

    recent = ordered[-max_len:]
    raw_values = [float(p["val"]) for p in recent]
    dates = [p["end"] for p in recent]

    arr = np.array(raw_values, dtype=float)
    if arr.max() != arr.min():
        scaled = MinMaxScaler().fit_transform(arr.reshape(-1, 1)).flatten()
    else:
        scaled = np.zeros(len(arr))

    pad_width = max_len - len(scaled)
    padded = np.pad(scaled, (0, pad_width), "constant").astype("float32")

    return padded, raw_values, dates, unit_key


def get_live_sequence(company_id: str, tag: str):
    """One-call convenience wrapper used by the scoring tool."""
    facts = fetch_company_facts(company_id)
    entity_name = facts.get("entityName", company_id)
    padded, raw_values, dates, unit = build_sequence(facts, tag)
    return {
        "entity_name": entity_name,
        "sequence": padded,
        "raw_values": raw_values,
        "dates": dates,
        "unit": unit,
        "n_points": len(raw_values),
    }

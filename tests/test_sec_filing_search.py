"""
Tests for app/services/sec_filing_search.py and the get_sec_filing_context
agent tool. The SEC search response fixture matches the REAL, verified
schema of efts.sec.gov/LATEST/search-index (confirmed against a live
query during development) - not a guessed shape.
"""
import pytest
from app.services import sec_filing_search, cache
from app.services.rag_agent import tool_get_sec_filing_context


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.clear_cache()
    yield
    cache.clear_cache()


FAKE_SEARCH_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_id": "0001318605-24-000010:tsla-20231231.htm",
                "_source": {
                    "ciks": ["0001318605"],
                    "display_names": ["Tesla, Inc.  (TSLA)  (CIK 0001318605)"],
                    "form": "10-K",
                    "adsh": "0001318605-24-000010",
                    "file_date": "2024-01-29",
                },
            }
        ]
    }
}

FAKE_FILING_HTML = """
<html><body>
<p>Unrelated boilerplate risk factor text goes here for a while.</p>
<p>During the fiscal year, our <b>net income</b> increased significantly
due to higher deliveries and improved production efficiency.</p>
</body></html>
"""


class _FakeSearchResp:
    status_code = 200
    def raise_for_status(self): pass
    def json(self): return FAKE_SEARCH_RESPONSE


class _FakeFilingResp:
    status_code = 200
    text = FAKE_FILING_HTML
    def raise_for_status(self): pass


@pytest.fixture
def fake_sec(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if "efts.sec.gov" in url:
            return _FakeSearchResp()
        return _FakeFilingResp()
    monkeypatch.setattr("app.services.sec_filing_search.requests.get", fake_get)


def test_search_filings_builds_correct_archive_url(fake_sec):
    results = sec_filing_search.search_filings("0001318605", "net income")
    assert len(results) == 1
    assert results[0]["entity_name"].startswith("Tesla")
    assert results[0]["doc_url"] == (
        "https://www.sec.gov/Archives/edgar/data/1318605/000131860524000010/tsla-20231231.htm"
    )


def test_fetch_excerpt_strips_html_and_finds_term(fake_sec):
    excerpt = sec_filing_search.fetch_excerpt(
        "https://www.sec.gov/Archives/edgar/data/1318605/000131860524000010/tsla-20231231.htm",
        "net income",
    )
    assert excerpt is not None
    assert "net income" in excerpt.lower()
    assert "<p>" not in excerpt
    assert "<b>" not in excerpt


def test_fetch_excerpt_returns_none_when_term_not_in_document(fake_sec):
    excerpt = sec_filing_search.fetch_excerpt(
        "https://www.sec.gov/Archives/edgar/data/1318605/000131860524000010/tsla-20231231.htm",
        "some phrase that definitely is not in the fake document",
    )
    assert excerpt is None


def test_tool_get_sec_filing_context_end_to_end(fake_sec):
    result = tool_get_sec_filing_context("0001318605", "NetIncomeLoss")
    assert "error" not in result
    assert result["entity_name"].startswith("Tesla")
    assert result["form"] == "10-K"
    assert "net income" in result["excerpt"].lower()


def test_tool_get_sec_filing_context_handles_no_results(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        class Empty:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"hits": {"hits": []}}
        return Empty()
    monkeypatch.setattr("app.services.sec_filing_search.requests.get", fake_get)

    result = tool_get_sec_filing_context("0001318605", "NetIncomeLoss")
    assert "error" in result

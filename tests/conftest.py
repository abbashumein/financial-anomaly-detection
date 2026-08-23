import sys
import os
import pathlib

_STUBS = pathlib.Path(__file__).parent / "_stubs"


def _ensure_importable(module_name: str) -> None:
    """If the real package is installed, do nothing - use it. If it's not
    installed (e.g. a lightweight CI box or this sandbox, where chromadb/
    groq are heavy optional deps), fall back to the tiny stub in
    tests/_stubs/ so importing app.services.rag_agent doesn't crash just
    because of an unrelated dependency we're not even testing."""
    try:
        __import__(module_name)
    except ImportError:
        sys.path.insert(0, str(_STUBS))


for _mod in ("chromadb", "groq", "pydantic_settings"):
    _ensure_importable(_mod)

# SEC requires a descriptive User-Agent even to construct edgar_client calls
# in some code paths - harmless dummy value for tests, real network calls
# are always mocked anyway.
os.environ.setdefault("EDGAR_USER_AGENT", "test-suite/1.0 (test@example.com)")

# app/core/security.py
"""
API-key auth, fail-closed.

- API_KEY set        -> X-API-Key must match (constant-time compare).
- API_KEY not set    -> 503 (server misconfigured), NOT open access.
- Local dev only: set ALLOW_NO_AUTH=true to explicitly skip auth.
"""
import hmac
import os

from fastapi import Header, HTTPException
from app.config.settings import settings


def _no_auth_allowed() -> bool:
    return os.getenv("ALLOW_NO_AUTH", "").strip().lower() == "true"


async def require_api_key(x_api_key: str | None = Header(default=None)):
    if not settings.api_key:
        if _no_auth_allowed():
            return  # explicit local-dev opt-out
        raise HTTPException(status_code=503, detail="Server auth is not configured.")

    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode(), settings.api_key.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header.",
        )
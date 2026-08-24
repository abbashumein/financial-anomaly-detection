# app/core/security.py
"""
Simple API-key auth for a portfolio-scale project.

Why not JWT/OAuth: this API has one "user type" (whoever has the key) -
there's no login flow, no per-user permissions, no session state to
manage. A shared API key via header is the right amount of security for
this scope; JWT would be solving a problem this project doesn't have.

If API_KEY is unset in settings (e.g. local dev with no .env value),
auth is skipped entirely - this makes local development frictionless
while still being real, enforced auth once a key is configured for a
public/demo deployment.
"""
from fastapi import Header, HTTPException
from app.config.settings import settings


async def require_api_key(x_api_key: str | None = Header(default=None)):
    if not settings.api_key:
        return  # no key configured -> auth disabled (local dev mode)

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header.",
        )

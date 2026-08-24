# app/services/cache.py
"""
Minimal in-memory TTL cache.

No Redis, no external service - just a dict with timestamps. This is
intentionally simple: for a portfolio project with no 24/7 hosting
budget, a per-process cache is the right amount of engineering. It
resets when the process restarts, which is fine here (SEC data doesn't
change minute-to-minute anyway).

Usage:
    @ttl_cache(seconds=900)
    def fetch_company_facts(company_id: str) -> dict:
        ...
"""
import time
import functools

_store: dict = {}   # key -> (expires_at_epoch, value)


def ttl_cache(seconds: int):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__module__, fn.__qualname__, args, tuple(sorted(kwargs.items())))
            now = time.time()

            cached = _store.get(key)
            if cached is not None and cached[0] > now:
                return cached[1]

            result = fn(*args, **kwargs)
            _store[key] = (now + seconds, result)
            return result

        return wrapper
    return decorator


def cache_stats() -> dict:
    """For a /health or debug endpoint - how many things are cached right now."""
    now = time.time()
    live = sum(1 for expires_at, _ in _store.values() if expires_at > now)
    return {"total_entries": len(_store), "live_entries": live}


def clear_cache():
    _store.clear()

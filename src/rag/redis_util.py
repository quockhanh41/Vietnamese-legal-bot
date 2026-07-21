"""Shared lazy Redis client factory (M4).

`redis` is imported only when a URL is actually present, so the in-memory
fallback path used by tests / eval / local dev needs no redis package or
running server. Any import or connection failure returns None — callers then
fall back to their in-memory backend."""
from __future__ import annotations

import os
from typing import Optional


def get_redis_client(url: Optional[str] = None):
    """Return a live `redis.Redis` for REDIS_URL (or `url`), else None.

    decode_responses=True → values come back as str (we store JSON text).
    Pings once so a set-but-dead URL degrades to None instead of failing later.
    """
    url = url or os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None

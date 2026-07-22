"""Embedding-similarity response cache (M4.2).

Keyed on the (condensed) query embedding, namespaced by collection so a
collection swap auto-invalidates. Redis 7 has no vector search → client-side
cosine scan over a bounded (FIFO, CACHE_MAX) buffer. Serving-layer only and
fail-open (any error → miss). Stores RAGResponse as a plain dict so the Redis
backend can JSON-serialize it."""
from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

from src.rag.orchestrator import RAGResponse
from src.rag.redis_util import get_redis_client

SIM_THRESHOLD = 0.95
CACHE_MAX = 500


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache(ABC):
    def __init__(self, embed_fn: Callable[[str], List[float]]):
        self._embed = embed_fn

    @abstractmethod
    def _entries(self, collection: str) -> List[Tuple[List[float], Dict]]:
        """(embedding, response_dict) pairs for this collection."""

    @abstractmethod
    def _store(self, collection: str, emb: List[float], resp: Dict) -> None:
        """Append one entry, FIFO-evicting to CACHE_MAX."""

    def get(self, query: str, collection: str) -> Optional[RAGResponse]:
        try:
            emb = self._embed(query)
            best, best_sim = None, 0.0
            for cached_emb, resp in self._entries(collection):
                sim = _cosine(emb, cached_emb)
                if sim > best_sim:
                    best, best_sim = resp, sim
            if best is not None and best_sim >= SIM_THRESHOLD:
                return RAGResponse(**best)
        except Exception:
            return None
        return None

    def set(self, query: str, collection: str, response: RAGResponse) -> None:
        try:
            self._store(collection, self._embed(query), asdict(response))
        except Exception:
            pass

    @staticmethod
    def from_env(embed_fn: Callable[[str], List[float]]) -> Optional["SemanticCache"]:
        if os.getenv("SEMANTIC_CACHE_DISABLED"):
            return None
        client = get_redis_client()
        if client is not None:
            return RedisSemanticCache(embed_fn, client)
        return MemorySemanticCache(embed_fn)


class MemorySemanticCache(SemanticCache):
    def __init__(self, embed_fn):
        super().__init__(embed_fn)
        self._buf: Dict[str, List[Tuple[List[float], Dict]]] = {}

    def _entries(self, collection):
        return self._buf.get(collection, [])

    def _store(self, collection, emb, resp):
        buf = self._buf.setdefault(collection, [])
        buf.append((emb, resp))
        while len(buf) > CACHE_MAX:
            del buf[0]


class RedisSemanticCache(SemanticCache):
    def __init__(self, embed_fn, client):
        super().__init__(embed_fn)
        self._r = client

    @staticmethod
    def _key(collection):
        return f"vlb:cache:{collection}"

    def _entries(self, collection):
        raw = self._r.lrange(self._key(collection), 0, -1)
        out = []
        for x in raw:
            item = json.loads(x)
            out.append((item["emb"], item["resp"]))
        return out

    def _store(self, collection, emb, resp):
        key = self._key(collection)
        self._r.rpush(key, json.dumps({"emb": emb, "resp": resp}, ensure_ascii=False))
        self._r.ltrim(key, -CACHE_MAX, -1)

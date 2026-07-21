"""Conversation memory for multi-turn chat (M4.1).

Serving-layer only — the RAG pipeline never sees session_id. One interface,
two backends: Redis when REDIS_URL is set, else a process-local dict. Both cap
at MAX_TURNS and are fail-open (any error → empty history / silent no-op; a
chat request never fails because memory is unavailable)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Optional

from src.rag.redis_util import get_redis_client

MAX_TURNS = 10
TTL_SECONDS = 3600


class SessionStore(ABC):
    @abstractmethod
    def get_history(self, session_id: Optional[str]) -> List[Dict]:
        """Most-recent-last list of {"question","answer"}; [] if none/error."""

    @abstractmethod
    def append_turn(self, session_id: Optional[str], question: str, answer: str) -> None:
        """Append one turn; no-op if session_id is falsy."""

    @staticmethod
    def from_env() -> "SessionStore":
        client = get_redis_client()
        return RedisSessionStore(client) if client is not None else MemorySessionStore()


class MemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS))

    def get_history(self, session_id):
        if not session_id:
            return []
        return list(self._data[session_id])

    def append_turn(self, session_id, question, answer):
        if not session_id:
            return
        self._data[session_id].append({"question": question, "answer": answer})


class RedisSessionStore(SessionStore):
    def __init__(self, client) -> None:
        self._r = client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"vlb:session:{session_id}"

    def get_history(self, session_id):
        if not session_id:
            return []
        try:
            raw = self._r.lrange(self._key(session_id), -MAX_TURNS, -1)
            return [json.loads(x) for x in raw]
        except Exception:
            return []

    def append_turn(self, session_id, question, answer):
        if not session_id:
            return
        try:
            key = self._key(session_id)
            self._r.rpush(key, json.dumps({"question": question, "answer": answer}, ensure_ascii=False))
            self._r.ltrim(key, -MAX_TURNS, -1)
            self._r.expire(key, TTL_SECONDS)
        except Exception:
            pass

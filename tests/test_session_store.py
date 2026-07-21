from src.rag.session_store import (
    MAX_TURNS,
    MemorySessionStore,
    RedisSessionStore,
    SessionStore,
)


class FakeRedis:
    """Minimal in-memory Redis list emulation (rpush/lrange/ltrim/expire)
    with correct negative-index semantics for our usage."""

    def __init__(self):
        self.store = {}

    @staticmethod
    def _slice(lst, start, stop):
        n = len(lst)
        if start < 0:
            start = max(n + start, 0)
        if stop < 0:
            stop = n + stop
        stop = min(stop, n - 1)
        if start > stop or start >= n:
            return []
        return lst[start:stop + 1]

    def rpush(self, k, v):
        self.store.setdefault(k, []).append(v)

    def lrange(self, k, start, stop):
        return list(self._slice(self.store.get(k, []), start, stop))

    def ltrim(self, k, start, stop):
        self.store[k] = self._slice(self.store.get(k, []), start, stop)

    def expire(self, k, ttl):
        pass


def test_memory_store_appends_and_caps():
    store = MemorySessionStore()
    for i in range(MAX_TURNS + 2):
        store.append_turn("s1", f"q{i}", f"a{i}")
    hist = store.get_history("s1")
    assert len(hist) == MAX_TURNS
    assert hist[0] == {"question": "q2", "answer": "a2"}
    assert hist[-1] == {"question": f"q{MAX_TURNS + 1}", "answer": f"a{MAX_TURNS + 1}"}


def test_memory_store_none_session_is_noop():
    store = MemorySessionStore()
    store.append_turn(None, "q", "a")
    assert store.get_history(None) == []


def test_memory_store_isolates_sessions():
    store = MemorySessionStore()
    store.append_turn("a", "qa", "aa")
    store.append_turn("b", "qb", "ab")
    assert store.get_history("a") == [{"question": "qa", "answer": "aa"}]


def test_redis_store_roundtrip_with_fake_client():
    store = RedisSessionStore(FakeRedis())
    store.append_turn("s1", "q1", "a1")
    store.append_turn("s1", "q2", "a2")
    assert store.get_history("s1") == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]


def test_redis_store_caps_history():
    store = RedisSessionStore(FakeRedis())
    for i in range(MAX_TURNS + 3):
        store.append_turn("s1", f"q{i}", f"a{i}")
    assert len(store.get_history("s1")) == MAX_TURNS
    # Verify underlying store is actually trimmed (ltrim working), not just read-sliced
    assert len(store._r.store[store._key("s1")]) == MAX_TURNS


def test_redis_store_errors_degrade_to_empty():
    class Boom:
        def lrange(self, *a):
            raise RuntimeError("down")
    assert RedisSessionStore(Boom()).get_history("s1") == []


def test_redis_store_append_turn_fail_open_on_rpush():
    class BoomOnWrite:
        def rpush(self, *a):
            raise RuntimeError("down")
    # append_turn should not propagate exception (fail-open)
    store = RedisSessionStore(BoomOnWrite())
    store.append_turn("s1", "q", "a")  # Should not raise


def test_redis_store_append_turn_fail_open_on_ltrim():
    class BoomOnTrim:
        def rpush(self, k, v):
            pass
        def ltrim(self, *a):
            raise RuntimeError("trim failed")
    # append_turn should not propagate exception (fail-open)
    store = RedisSessionStore(BoomOnTrim())
    store.append_turn("s1", "q", "a")  # Should not raise


def test_from_env_uses_memory_without_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(SessionStore.from_env(), MemorySessionStore)

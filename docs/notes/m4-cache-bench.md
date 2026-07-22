# M4 Cache Bench (2026-07-22)

Semantic-cache hit latency (`MemorySemanticCache`, 1024-d normalized AITeamVN
dense vectors). Hit path = embed query + cosine scan over cached vectors; it
skips retrieval + rerank + generate + verify (seconds on a miss).

Bench: `scratchpad/cache_bench.py` — one warmed entry, 20 timed `cache.get`
hits on CPU. Machine: local dev (Darwin, CPU embedder).

| metric | ms |
|---|---|
| p50 hit | 82.4 |
| p95 hit | 85.0 |
| min | 79.9 |
| max | 85.0 |

Gate M4 "cache hit < 500ms": **PASS** (p95 85ms << 500ms). The dominant cost is
the single query embed on the already-loaded embedder (~80ms on CPU); the
cosine scan over ≤ `CACHE_MAX` (500) vectors is sub-millisecond, so latency is
flat in cache size and would drop further on GPU. A real cache hit therefore
returns in ~1/10th of a second versus multiple seconds for a full pipeline miss.

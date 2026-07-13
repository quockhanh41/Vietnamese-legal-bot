# Reranker latency (cpu, repeats=20, luat_law_ctx_v3)

| pool | P50 ms | P95 ms | P99 ms | mean ms | pairs/s |
|---|---|---|---|---|---|
| 10 | 459 | 482 | 645 | 466 | 21.4 |
| 20 | 886 | 894 | 903 | 875 | 22.9 |
| 50 | 2135 | 2180 | 2191 | 2126 | 23.5 |
| 100 | 7540 | 7582 | 7611 | 7536 | 13.3 |

> ⚠️ P95 at largest pool exceeds 3s — recommend capping candidates (e.g. top-10) before rerank at M5 CPU deploy.

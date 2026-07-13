"""
Reranker latency harness (M2.6). Measures BGE-Reranker-v2 P50/P95/P99 over
candidate pools of size {10,20,50,100} using REAL chunk texts sampled from a
collection. Measure-only — informs whether to cap candidates at M5.

Usage: .venv/bin/python scripts/bench_reranker.py --collection luat_law_ctx_v3 --repeats 20
"""
import argparse, json, os, sys, time, statistics
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

POOL_SIZES = [10, 20, 50, 100]
QUERIES = [
    "Tội trộm cắp tài sản bị xử phạt như thế nào?",
    "Thời hạn báo trước khi đơn phương chấm dứt hợp đồng lao động?",
    "Điều kiện thành lập doanh nghiệp tư nhân là gì?",
    "Thủ tục cấp giấy chứng nhận quyền sử dụng đất?",
]

def _sample_chunks(collection, n):
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore(collection_name=collection)
    points, _ = s.client.scroll(collection_name=collection, limit=n,
                                with_payload=["chunk_text"], with_vectors=False)
    return [ (p.payload or {}).get("chunk_text", "")[:512] for p in points if (p.payload or {}).get("chunk_text") ]

def _percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

def main(collection, repeats):
    from FlagEmbedding import FlagReranker
    # Pinned to CPU to match production (src/rag/orchestrator.py:315-319).
    device = "cpu"
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cpu")
    pool = _sample_chunks(collection, max(POOL_SIZES))
    if len(pool) < max(POOL_SIZES):
        print(f"⚠️ only {len(pool)} chunks available; capping pool sizes")
    rows = []
    for size in POOL_SIZES:
        if size > len(pool):
            continue
        passages = pool[:size]
        # warm-up
        reranker.compute_score([[QUERIES[0], passages[0]]], normalize=True)
        lat = []
        for _ in range(repeats):
            q = QUERIES[_ % len(QUERIES)]
            pairs = [[q, p] for p in passages]
            t0 = time.perf_counter()
            reranker.compute_score(pairs, normalize=True)
            lat.append((time.perf_counter() - t0) * 1000)  # ms
        rows.append({"pool": size, "p50": _percentile(lat, 50), "p95": _percentile(lat, 95),
                     "p99": _percentile(lat, 99), "mean": statistics.mean(lat),
                     "pairs_per_s": size / (statistics.mean(lat) / 1000)})
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs("data/eval_reports", exist_ok=True)
    meta = {"device": device, "repeats": repeats, "collection": collection, "rows": rows}
    with open(f"data/eval_reports/reranker_bench_{ts}.json", "w") as f:
        json.dump(meta, f, indent=2)
    md = [f"# Reranker latency ({device}, repeats={repeats}, {collection})", "",
          "| pool | P50 ms | P95 ms | P99 ms | mean ms | pairs/s |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pool']} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} | "
                  f"{r['mean']:.0f} | {r['pairs_per_s']:.1f} |")
    if rows and rows[-1]["p95"] > 3000:
        md += ["", "> ⚠️ P95 at largest pool exceeds 3s — recommend capping candidates "
               "(e.g. top-10) before rerank at M5 CPU deploy."]
    report = "\n".join(md)
    with open(f"data/eval_reports/reranker_bench_{ts}.md", "w") as f:
        f.write(report + "\n")
    print(report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="luat_law_ctx_v3")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    main(a.collection, a.repeats)

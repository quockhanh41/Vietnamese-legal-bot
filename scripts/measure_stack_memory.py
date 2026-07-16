"""Gate #3 — peak RAM + latency of the AITeamVN mixed stack on CPU (HF Spaces
deploy parity). Loads MixedHybridEmbedder (AITeamVN dense + BGE-M3 sparse) +
AITeamVN reranker, runs an end-to-end query (embed -> hybrid search -> rerank),
and reports peak RSS vs the 16GB budget. Local RSS approximates HF Spaces —
read trend + margin, not an absolute.

Usage: .venv/bin/python scripts/measure_stack_memory.py \
    --collection luat_law_ctx_v4_aiteamvn --repeats 20
"""
import argparse, json, os, statistics, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

REPORT_DIR = "data/eval_reports"
QUERY = "Điều kiện chuyển mục đích sử dụng đất nông nghiệp sang đất ở?"


def _percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def classify_headroom(peak_rss_mb, budget_mb=16384):
    ratio = peak_rss_mb / budget_mb
    if ratio < 0.70:
        return "fits-headroom"
    if ratio < 0.90:
        return "tight"
    return "doesn't-fit"


def main(collection, repeats, budget_mb):
    import psutil
    from src.data.mixed_embedder import MixedHybridEmbedder
    from src.db.qdrant_client import LegalVectorStore
    from FlagEmbedding import FlagReranker

    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6

    embedder = MixedHybridEmbedder(device="cpu")
    reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=False, device="cpu")
    store = LegalVectorStore(collection_name=collection)

    lat = []
    peak = proc.memory_info().rss / 1e6
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = embedder.encode([QUERY], return_dense=True, return_sparse=True)
        cands = store.search_hybrid(out["dense_vecs"][0].tolist(),
                                    out["lexical_weights"][0], top_k=10)
        reranker.compute_score([[QUERY, c.chunk_text] for c in cands], normalize=True)
        lat.append((time.perf_counter() - t0) * 1000)
        peak = max(peak, proc.memory_info().rss / 1e6)

    report = {
        "device": "cpu",
        "collection": collection,
        "repeats": repeats,
        "budget_mb": budget_mb,
        "rss_baseline_mb": round(rss0, 1),
        "rss_peak_mb": round(peak, 1),
        "rss_stack_mb": round(peak - rss0, 1),
        "headroom_ratio": round(peak / budget_mb, 3),
        "verdict": classify_headroom(peak, budget_mb),
        "latency_ms_p50": round(_percentile(lat, 50), 1),
        "latency_ms_p95": round(_percentile(lat, 95), 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{ts}_stack_memory.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ report → {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="luat_law_ctx_v4_aiteamvn")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--budget-mb", type=int, default=16384)
    a = ap.parse_args()
    main(a.collection, a.repeats, a.budget_mb)

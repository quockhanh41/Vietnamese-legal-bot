"""
M3.1 external benchmark runner — Zalo Legal Text Retrieval 2021.

Measures the production retrieval code paths on the benchmark corpus
(collection built by scripts/ingest_zalo_corpus.py) at three ablation levels:
  A  dense-only        LegalVectorStore.search_dense
  B  hybrid RRF        LegalVectorStore.search_hybrid
  C  hybrid + rerank   search_hybrid(top_k=--pool) → BGE-reranker-v2-m3 (CPU)

Usage:
  Smoke (20 dev queries, no rerank):
    .venv/bin/python scripts/bench_external.py --levels a,b --split dev --limit 20
  Full A/B (all 3,196 queries):
    .venv/bin/python scripts/bench_external.py --levels a,b --split full
  Level C on held-out test split (~320 queries, ~12 min CPU rerank):
    .venv/bin/python scripts/bench_external.py --levels c --split test

Metrics: recall@{1,5,10}, MRR@10, nDCG@10 on exact (law_id, article_id).
Report: data/eval_reports/<ts>_zalo_bench.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from scripts.download_zalo import QA_FILE, load_zalo_questions
from src.eval.external_bench import load_splits
from src.eval.bench_metrics import recall_at_k, mrr_at_k, ndcg_at_k

REPORT_DIR = "data/eval_reports"
RECALL_KS = (1, 5, 10)


def _percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _select_questions(split, limit):
    questions = load_zalo_questions(QA_FILE)
    if split != "full":
        ids = set(load_splits()[split])
        questions = [q for q in questions if q["question_id"] in ids]
    if limit:
        questions = questions[:limit]
    return questions


def _embed_queries(questions, device, batch_size=64):
    from src.data.embedder import BGEEmbedder
    embedder = BGEEmbedder(device=device)
    dense_all, sparse_all = [], []
    texts = [q["question"] for q in questions]
    for i in range(0, len(texts), batch_size):
        out = embedder.encode(texts[i:i + batch_size], batch_size=batch_size,
                              return_dense=True, return_sparse=True)
        dense_all.extend([d.tolist() for d in out["dense_vecs"]])
        sparse_all.extend(out["lexical_weights"])
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)} queries")
    return dense_all, sparse_all


def _ids(results):
    """SearchResult → ranked (law_id, article_id) per Task 4 payload contract."""
    return [(r.document_number, r.article) for r in results]


def run_level(level, questions, dense, sparse, store, pool, reranker=None):
    per_query, latencies = [], []
    for i, q in enumerate(questions):
        relevant = {(ra["law_id"], ra["article_id"]) for ra in q["relevant_articles"]}
        t0 = time.perf_counter()
        if level == "a":
            retrieved = _ids(store.search_dense(dense[i], top_k=max(RECALL_KS)))
        elif level == "b":
            retrieved = _ids(store.search_hybrid(dense[i], sparse[i],
                                                 top_k=max(RECALL_KS)))
        else:  # c
            candidates = store.search_hybrid(dense[i], sparse[i], top_k=pool)
            scores = reranker.compute_score(
                [[q["question"], r.chunk_text] for r in candidates], normalize=True)
            if not isinstance(scores, list):
                scores = [scores]
            ranked = [r for _, r in sorted(zip(scores, candidates),
                                           key=lambda t: t[0], reverse=True)]
            retrieved = _ids(ranked[:max(RECALL_KS)])
        latencies.append((time.perf_counter() - t0) * 1000)
        row = {f"recall@{k}": recall_at_k(retrieved, relevant, k) for k in RECALL_KS}
        row["mrr@10"] = mrr_at_k(retrieved, relevant, k=10)
        row["ndcg@10"] = ndcg_at_k(retrieved, relevant, k=10)
        per_query.append(row)
        if (i + 1) % 200 == 0:
            print(f"  level {level.upper()}: {i + 1}/{len(questions)}")

    n = len(per_query)
    summary = {m: sum(r[m] for r in per_query) / n for m in per_query[0]}
    summary["queries"] = n
    summary["latency_ms_p50"] = round(_percentile(latencies, 50), 1)
    summary["latency_ms_p95"] = round(_percentile(latencies, 95), 1)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="a,b",
                    help="comma-separated subset of a,b,c")
    ap.add_argument("--split", default="full",
                    choices=["full", "train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (smoke)")
    ap.add_argument("--collection", default="zalo_legal_2021")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--pool", type=int, default=50, help="rerank candidate pool (level c)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="query-embedding device")
    args = ap.parse_args()
    levels = [l.strip() for l in args.levels.split(",")]

    from src.db.qdrant_client import LegalVectorStore
    store = LegalVectorStore(url=args.qdrant_url, api_key="",
                             collection_name=args.collection)

    questions = _select_questions(args.split, args.limit)
    print(f"{len(questions)} queries (split={args.split}) → levels {levels}")
    dense, sparse = _embed_queries(questions, args.device)

    reranker = None
    if "c" in levels:
        from FlagEmbedding import FlagReranker
        # CPU pinned to match production (see scripts/bench_reranker.py).
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cpu")

    results = {}
    for level in levels:
        print(f"── level {level.upper()} ──")
        results[level] = run_level(level, questions, dense, sparse, store,
                                   args.pool, reranker)
        print(json.dumps(results[level], indent=2))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "benchmark": "zalo-legal-text-retrieval-2021",
        "collection": args.collection,
        "split": args.split,
        "limit": args.limit or None,
        "pool": args.pool if "c" in levels else None,
        "levels": results,
        "timestamp": ts,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{ts}_zalo_bench.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ report → {path}")


if __name__ == "__main__":
    main()

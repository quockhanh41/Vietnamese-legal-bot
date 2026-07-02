"""
Chunking strategy evaluation orchestrator.

Two-round evaluation:
  Round 1: Parameter sweep with retrieval-only metrics (0 LLM calls)
           → Recall@k, MRR, Hit Rate, chunk stats
  Round 2: Full RAGAS on top strategies (LLM judge needed)

Usage:
    python scripts/eval_chunking.py --round 1 --docs 20 --queries data/eval/test_queries.json
    python scripts/eval_chunking.py --round 2 --docs 100 --queries data/eval/test_queries.json
"""

import json
import sys
import os
import time
import hashlib
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import pyarrow.dataset as ds
import numpy as np

from src.data.chunker import create_chunker
from src.data.embedder import BGEEmbedder
from src.db.qdrant_client import LegalVectorStore


# --- Parameter Sweeps ---

ROUND1_PARAMS = {
    "fixed_size": [
        {"chunk_size": 512, "overlap": 50},
        {"chunk_size": 512, "overlap": 100},
        {"chunk_size": 1024, "overlap": 50},
        {"chunk_size": 1024, "overlap": 100},
        {"chunk_size": 2048, "overlap": 50},
        {"chunk_size": 2048, "overlap": 100},
    ],
    "sentence": [
        {"max_chars": 1000, "min_sentences": 2},
        {"max_chars": 1500, "min_sentences": 3},
        {"max_chars": 2000, "min_sentences": 2},
    ],
    "hierarchical": [
        {"max_chunk_size": 1000, "min_chunk_size": 200},
        {"max_chunk_size": 1500, "min_chunk_size": 200},
        {"max_chunk_size": 2000, "min_chunk_size": 200},
    ],
    "semantic": [
        {"threshold": 0.5, "min_size": 300},
        {"threshold": 0.6, "min_size": 300},
        {"threshold": 0.7, "min_size": 300},
    ],
    "article_level": [
        {"min_article_length": 100},
    ],
}

STRATEGY_NAMES = {
    "fixed_size": "S1: Fixed-size",
    "sentence": "S2: Sentence-based",
    "hierarchical": "S3: Hierarchical",
    "semantic": "S4: Semantic",
    "article_level": "S5: Article-level",
}

K_VALUES = [3, 5, 10]


def _make_collection_name(strategy: str, params: dict) -> str:
    key = f"{strategy}_{json.dumps(params, sort_keys=True)}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"eval_{strategy}_{h}"


def _load_documents(data_path: str, num_docs: int, seed: int = 42,
                   required_ids: set = None) -> List[dict]:
    """Load a random sample of documents, guaranteeing required_ids are included."""
    dataset = ds.dataset(data_path, format="parquet")

    LEGAL_TYPES = {"Luật", "Bộ luật", "Nghị định", "Thông tư", "Nghị quyết", "Quyết định"}
    required_ids = required_ids or set()

    # Phase 1: Load required docs by ID (scan full dataset)
    required_docs = {}
    if required_ids:
        for batch in dataset.to_batches(batch_size=200):
            batch_df = batch.to_pandas()
            for row_dict in batch_df.to_dict("records"):
                doc_id = row_dict.get("id")
                if doc_id in required_ids and doc_id not in required_docs:
                    content = row_dict.get("content", "")
                    if content and len(content) >= 500:
                        if row_dict.get("legal_type") in LEGAL_TYPES:
                            required_docs[doc_id] = row_dict
            if len(required_docs) >= len(required_ids):
                break
        print(f"  Pre-loaded {len(required_docs)}/{len(required_ids)} required query docs")

    # Phase 2: Fill remaining slots with random docs
    docs = list(required_docs.values())
    remaining = num_docs - len(docs)
    if remaining > 0:
        for batch in dataset.to_batches(batch_size=200):
            batch_df = batch.to_pandas()
            rows = batch_df.to_dict("records")
            np.random.seed(seed)
            np.random.shuffle(rows)
            for row_dict in rows:
                if len(docs) >= num_docs:
                    break
                # Skip if already included from required set
                if row_dict.get("id") in required_ids:
                    continue
                content = row_dict.get("content", "")
                if not content or len(content) < 500:
                    continue
                if row_dict.get("legal_type") not in LEGAL_TYPES:
                    continue
                docs.append(row_dict)
            if len(docs) >= num_docs:
                break

    return docs


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clear_gpu_memory():
    """Clear GPU memory caches to prevent OOM across multiple eval combos."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    import gc
    gc.collect()


def _compute_retrieval_metrics(
    search_results: List,
    query: dict,
    k_values: List[int],
) -> Dict:
    """
    Compute retrieval-only metrics for a single query.

    Uses source_doc_id + source_article from query metadata to determine
    if the correct document/article was retrieved.

    Returns: {recall@k, mrr, hit@k} dict
    """
    source_doc_id = query.get("source_doc_id")
    source_article = query.get("source_article", "")
    ground_truth_texts = query.get("ground_truth_contexts", [])
    ground_truth = " ".join(ground_truth_texts) if ground_truth_texts else ""

    metrics = {}

    # Find rank of first relevant result
    first_rank = None
    for rank, result in enumerate(search_results, 1):
        # Match by document_id OR article
        doc_match = (source_doc_id and result.document_id == source_doc_id)
        article_match = (source_article and result.article and
                         source_article[:30] in result.article)

        if doc_match or article_match:
            first_rank = rank
            break

    # If no exact match, try text overlap as fallback
    if first_rank is None and ground_truth:
        best_overlap = 0
        best_rank = None
        gt_words = set(ground_truth.split())
        for rank, result in enumerate(search_results, 1):
            result_words = set(result.chunk_text.split())
            if not gt_words or not result_words:
                continue
            overlap = len(gt_words & result_words) / len(gt_words)
            if overlap > best_overlap and overlap > 0.1:
                best_overlap = overlap
                best_rank = rank
        first_rank = best_rank

    # Compute Recall@k and Hit@k
    for k in k_values:
        is_hit = first_rank is not None and first_rank <= k
        metrics[f"recall@{k}"] = 1.0 if is_hit else 0.0
        metrics[f"hit@{k}"] = 1.0 if is_hit else 0.0

    # MRR (Mean Reciprocal Rank)
    metrics["mrr"] = 1.0 / first_rank if first_rank else 0.0

    return metrics


def run_evaluation(
    data_path: str,
    queries_path: str,
    round_num: int,
    num_docs: int,
    seed: int = 42,
    device: str = "cpu",
):
    """Run the full evaluation pipeline."""
    # Load test queries
    with open(queries_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    print(f"Loaded {len(test_queries)} test queries")

    # Extract required doc IDs from queries (must be in the indexed set)
    required_ids = set(q["source_doc_id"] for q in test_queries if q.get("source_doc_id"))

    # Load documents
    print(f"\nLoading {num_docs} documents (including {len(required_ids)} query docs)...")
    docs = _load_documents(data_path, num_docs, seed, required_ids=required_ids)
    print(f"Loaded {len(docs)} valid documents")

    # Init embedder (shared across all combos)
    print("\nLoading BGE-M3 embedder (shared instance, ~2GB)...")
    import torch
    torch.set_num_threads(2)  # Limit CPU threads to avoid memory spikes
    embedder = BGEEmbedder(device=device)
    EMBED_BATCH = 16  # Smaller batch to reduce peak memory
    print("Embedder loaded ✓")

    # Determine combos to run
    if round_num == 1:
        combos_to_run = ROUND1_PARAMS
        output_file = "data/eval/round1_results.json"
    else:
        round1_path = "data/eval/round1_results.json"
        if not os.path.exists(round1_path):
            print(f"ERROR: Round 1 results not found at {round1_path}")
            return
        with open(round1_path, "r") as f:
            round1_data = json.load(f)
        combos_to_run = {}
        for strategy in ROUND1_PARAMS:
            strategy_results = [r for r in round1_data["results"] if r["strategy"] == strategy]
            if strategy_results:
                best = max(strategy_results, key=lambda r: r.get("recall@5", 0))
                combos_to_run[strategy] = [best["params"]]
            else:
                combos_to_run[strategy] = ROUND1_PARAMS[strategy][:1]
        output_file = "data/eval/round2_results.json"

    # Load existing results for incremental resume
    all_results = []
    completed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing = json.load(f)
            for r in existing.get("results", []):
                key = f"{r['strategy']}_{json.dumps(r['params'], sort_keys=True)}"
                completed_keys.add(key)
                all_results.append(r)
        if completed_keys:
            print(f"Resuming: {len(completed_keys)} combos already completed, skipping...")

    total_new = sum(
        1 for s, plist in combos_to_run.items()
        for p in plist
        if f"{s}_{json.dumps(p, sort_keys=True)}" not in completed_keys
    )
    print(f"\n{'='*60}")
    print(f"Round {round_num}: {total_new} remaining combos ({len(completed_keys)} done)")
    print(f"Documents: {len(docs)}, Test queries: {len(test_queries)}")
    if round_num == 1:
        print(f"Metrics: Retrieval-only (Recall@k, MRR) — 0 LLM calls needed")
    else:
        print(f"Metrics: Full RAGAS (faithfulness, context_precision, etc.)")
    print(f"{'='*60}\n")

    for strategy, params_list in combos_to_run.items():
        for params in params_list:
            key = f"{strategy}_{json.dumps(params, sort_keys=True)}"
            if key in completed_keys:
                continue

            label = f"{STRATEGY_NAMES.get(strategy, strategy)} ({params})"
            print(f"\n{'─'*60}")
            print(f"Evaluating: {label}")
            print(f"{'─'*60}")

            try:
                if round_num == 1:
                    result = _eval_retrieval_only(
                        strategy=strategy,
                        params=params,
                        docs=docs,
                        test_queries=test_queries,
                        embedder=embedder,
                    )
                else:
                    result = _eval_full_ragas(
                        strategy=strategy,
                        params=params,
                        docs=docs,
                        test_queries=test_queries,
                        embedder=embedder,
                    )
                result["label"] = label
                all_results.append(result)

                # Print per-combo summary
                print(f"  Chunks: {result['num_chunks']} (avg {result['avg_chunk_size']:.0f} chars)")
                print(f"  Emb time: {result['embedding_time_s']:.1f}s")
                for k in [3, 5, 10]:
                    key_k = f"recall@{k}"
                    if key_k in result:
                        print(f"  Recall@{k}: {result[key_k]:.3f}")

                # Incremental save
                outcome = {
                    "round": round_num,
                    "num_docs": len(docs),
                    "num_queries": len(test_queries),
                    "metrics_type": "retrieval-only" if round_num == 1 else "ragas",
                    "results": all_results,
                }
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(outcome, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
                # Clear GPU memory on failure to prevent cascading crashes
                _clear_gpu_memory()
                continue

            # Clear GPU memory between combos to prevent memory accumulation
            _clear_gpu_memory()

    # Save results
    outcome = {
        "round": round_num,
        "num_docs": len(docs),
        "num_queries": len(test_queries),
        "metrics_type": "retrieval-only" if round_num == 1 else "ragas",
        "results": all_results,
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outcome, f, ensure_ascii=False, indent=2)

    # Print summary
    _print_summary(all_results, round_num)
    print(f"\nResults saved to: {output_file}")


def _eval_retrieval_only(
    strategy: str,
    params: dict,
    docs: List[dict],
    test_queries: List[dict],
    embedder: BGEEmbedder,
) -> dict:
    """Evaluate with retrieval-only metrics (0 LLM calls)."""
    collection_name = _make_collection_name(strategy, params)

    # 1. Chunk
    if strategy == "semantic":
        chunker = create_chunker(strategy, embedder=embedder, **params)
    else:
        chunker = create_chunker(strategy, **params)

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    if not all_chunks:
        return _empty_result(strategy, params)

    chunk_texts = [c.chunk_text[:4000] for c in all_chunks]

    # 2. Embed
    t0 = time.time()
    dense_vecs, sparse_vecs = embedder.encode_documents(chunk_texts, batch_size=16)
    embedding_time = time.time() - t0
    _clear_gpu_memory()  # Free GPU memory after large embedding batch

    # 3. Index into Qdrant
    store = LegalVectorStore(collection_name=collection_name)
    store.create_collection(recreate=True)
    metadata_list = [c.to_dict() for c in all_chunks]
    store.upsert_chunks(
        chunk_texts=chunk_texts,
        dense_vectors=[v.tolist() for v in dense_vecs],
        sparse_vectors=sparse_vecs,
        metadata_list=metadata_list,
    )

    # 4. Evaluate retrieval for each query
    query_metrics = []
    search_times = []

    for q in test_queries:
        question = q["question"]

        t0 = time.time()
        query_vec = embedder.encode_queries([question])[0]
        search_results = store.search_dense(query_vec.tolist(), top_k=max(K_VALUES))
        search_times.append(time.time() - t0)

        m = _compute_retrieval_metrics(search_results, q, K_VALUES)
        query_metrics.append(m)

    # 5. Aggregate metrics
    result = {
        "strategy": strategy,
        "params": params,
        "num_chunks": len(all_chunks),
        "avg_chunk_size": round(_safe_mean([len(t) for t in chunk_texts]), 0),
        "embedding_time_s": round(embedding_time, 2),
        "avg_search_latency_ms": round(_safe_mean(search_times) * 1000, 1),
    }

    for k in K_VALUES:
        result[f"recall@{k}"] = round(_safe_mean([m[f"recall@{k}"] for m in query_metrics]), 4)
        result[f"hit@{k}"] = round(_safe_mean([m[f"hit@{k}"] for m in query_metrics]), 4)
    result["mrr"] = round(_safe_mean([m["mrr"] for m in query_metrics]), 4)

    # 6. Clean up
    try:
        store.client.delete_collection(collection_name)
    except Exception:
        pass

    return result


def _eval_full_ragas(
    strategy: str,
    params: dict,
    docs: List[dict],
    test_queries: List[dict],
    embedder: BGEEmbedder,
) -> dict:
    """
    Full RAGAS evaluation with LLM judge.
    Only used in Round 2 with top strategies (few combos).
    """
    from src.llm.gateway import LLMGateway
    from src.llm.prompts import SYSTEM_PROMPT, GENERATION_PROMPT
    from src.eval.ragas_runner import RagasEvaluator

    collection_name = _make_collection_name(strategy, params)

    # Same chunk + embed + index flow as retrieval-only
    if strategy == "semantic":
        chunker = create_chunker(strategy, embedder=embedder, **params)
    else:
        chunker = create_chunker(strategy, **params)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))
    if not all_chunks:
        return _empty_result(strategy, params)

    chunk_texts = [c.chunk_text[:4000] for c in all_chunks]
    t0 = time.time()
    dense_vecs, sparse_vecs = embedder.encode_documents(chunk_texts, batch_size=16)
    embedding_time = time.time() - t0

    store = LegalVectorStore(collection_name=collection_name)
    store.create_collection(recreate=True)
    store.upsert_chunks(
        chunk_texts=chunk_texts,
        dense_vectors=[v.tolist() for v in dense_vecs],
        sparse_vectors=sparse_vecs,
        metadata_list=[c.to_dict() for c in all_chunks],
    )

    # Full RAGAS evaluation
    llm = LLMGateway()
    evaluator = RagasEvaluator(llm)

    search_times = []
    ragas_samples = []

    for q in test_queries:
        question = q["question"]

        t0 = time.time()
        query_vec = embedber.encode_queries([question])[0]
        search_results = store.search_dense(query_vec.tolist(), top_k=5)
        search_times.append(time.time() - t0)

        retrieved_contexts = [r.chunk_text for r in search_results]
        context_str = "\n\n---\n\n".join(
            f"[{i+1}] {r.document_title} — {r.article or ''}\n{r.chunk_text[:500]}"
            for i, r in enumerate(search_results)
        )
        prompt = GENERATION_PROMPT.format(context=context_str, question=question)

        raw = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT,
                           temperature=0.1, json_mode=True)
        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer", raw)
        except json.JSONDecodeError:
            answer = raw

        ragas_samples.append({
            "question": question,
            "answer": answer,
            "contexts": retrieved_contexts,
            "ground_truth": q.get("ground_truth_answer", ""),
        })

    # Compute RAGAS
    avg_scores = {}
    try:
        all_scores = []
        for sample in ragas_samples:
            try:
                s = evaluator.evaluate_sample(**sample)
                all_scores.append(s)
            except Exception:
                continue
        if all_scores:
            for key in all_scores[0].keys():
                avg_scores[key] = _safe_mean([s.get(key, 0.0) for s in all_scores])
    except Exception as e:
        print(f"    RAGAS eval warning: {e}")

    # Clean up
    try:
        store.client.delete_collection(collection_name)
    except Exception:
        pass

    return {
        "strategy": strategy,
        "params": params,
        "num_chunks": len(all_chunks),
        "avg_chunk_size": round(_safe_mean([len(t) for t in chunk_texts]), 0),
        "embedding_time_s": round(embedding_time, 2),
        "avg_search_latency_ms": round(_safe_mean(search_times) * 1000, 1),
        "faithfulness": round(avg_scores.get("faithfulness", 0.0), 4),
        "context_precision": round(avg_scores.get("context_precision", 0.0), 4),
        "context_recall": round(avg_scores.get("context_recall", 0.0), 4),
        "answer_relevancy": round(avg_scores.get("answer_relevancy", 0.0), 4),
    }


def _empty_result(strategy: str, params: dict) -> dict:
    return {
        "strategy": strategy,
        "params": params,
        "num_chunks": 0,
        "avg_chunk_size": 0,
        "embedding_time_s": 0.0,
        "avg_search_latency_ms": 0.0,
    }


def _print_summary(results: List[dict], round_num: int):
    if not results:
        print("\nNo results to display.")
        return

    print(f"\n{'='*90}")
    print(f"SUMMARY — Round {round_num} Chunking Strategy Comparison")
    print(f"{'='*90}")

    # Sort by recall@5 (primary metric for retrieval quality)
    sort_key = "recall@5" if round_num == 1 else "faithfulness"
    sorted_results = sorted(results, key=lambda r: r.get(sort_key, 0), reverse=True)

    if round_num == 1:
        header = f"{'Strategy':<45} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'#Chunks':>8} {'Emb(s)':>7}"
    else:
        header = f"{'Strategy':<45} {'Faith':>7} {'CtxPrec':>8} {'CtxRec':>7} {'AnsRel':>7} {'#Chunks':>8}"

    print(header)
    print("-" * 90)

    for r in sorted_results:
        label = r.get("label", f"{r['strategy']} {r['params']}")[:44]
        if round_num == 1:
            print(
                f"{label:<45} "
                f"{r.get('recall@3', 0):6.3f} "
                f"{r.get('recall@5', 0):6.3f} "
                f"{r.get('recall@10', 0):6.3f} "
                f"{r.get('mrr', 0):6.3f} "
                f"{r.get('num_chunks', 0):8d} "
                f"{r.get('embedding_time_s', 0):7.1f}"
            )
        else:
            print(
                f"{label:<45} "
                f"{r.get('faithfulness', 0):7.3f} "
                f"{r.get('context_precision', 0):8.3f} "
                f"{r.get('context_recall', 0):7.3f} "
                f"{r.get('answer_relevancy', 0):7.3f} "
                f"{r.get('num_chunks', 0):8d}"
            )

    # Best strategy
    best = sorted_results[0]
    print(f"\n🏆 Best strategy: {best['label']}")
    if round_num == 1:
        print(f"   Recall@5={best.get('recall@5', 0):.3f}, MRR={best.get('mrr', 0):.3f}")


@click.command()
@click.option("--round", "round_num", type=int, required=True, help="Round 1 or 2")
@click.option("--docs", "num_docs", type=int, default=20, help="Number of documents")
@click.option("--queries", "queries_path", default="data/eval/test_queries.json")
@click.option("--data-path", default="data/raw/legal_docs_merged.parquet")
@click.option("--seed", type=int, default=42)
@click.option("--device", default="cpu")
def main(round_num, num_docs, queries_path, data_path, seed, device):
    if not os.path.exists(queries_path):
        print(f"ERROR: Test queries file not found: {queries_path}")
        print("Generate it first: python scripts/generate_eval_queries.py --docs 20")
        return
    run_evaluation(data_path, queries_path, round_num, num_docs, seed, device)


if __name__ == "__main__":
    main()

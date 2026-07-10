"""
Diagnostic script — per-signal confidence breakdown for specific eval questions.

Runs the REAL AdaptiveRAGPipeline (real Qdrant + real Groq LLM) on a fixed list
of question IDs from data/eval/eval_set_criminal_v1.json, and dumps:
  - complexity/route chosen
  - retrieved chunks (doc/article/score) vs expected_articles ground truth
  - rerank scores (if any)
  - the generated answer
  - NLI claims + entailment labels (if the verifier ran semantic verification)
  - the full 5-signal confidence breakdown + weighted score

Written for the 2026-07-07 false-refusal diagnosis
(docs/notes/crim-false-refusal-diagnosis-2026-07-07.md) — NOT part of the
regular eval suite, safe to keep around as a debugging tool.

Usage:
    .venv/bin/python scripts/inspect_confidence.py crim-109 crim-201 crim-202 crim-302
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.data.constants import DEFAULT_COLLECTION_NAME
from src.data.embedder import BGEEmbedder
from src.db.qdrant_client import LegalVectorStore
from src.llm.gateway import LLMGateway
from src.rag.orchestrator import AdaptiveRAGPipeline

EVAL_SET = "data/eval/eval_set_criminal_v1.json"


def load_question(qid: str):
    with open(EVAL_SET, encoding="utf-8") as f:
        data = json.load(f)
    for q in data["questions"]:
        if q["id"] == qid:
            return q
    raise KeyError(qid)


def main():
    qids = sys.argv[1:] or ["crim-109", "crim-201", "crim-202", "crim-302"]

    vector_store = LegalVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
    print("Loading BGE-M3 embedder...")
    embedder = BGEEmbedder()
    llm = LLMGateway()
    pipeline = AdaptiveRAGPipeline(vector_store=vector_store, embedder=embedder, llm_gateway=llm)

    results_out = {}

    for qid in qids:
        q = load_question(qid)
        print(f"\n{'='*80}\n{qid}: {q['question']}\n{'='*80}")

        # Monkeypatch _post_process to capture intermediate objects without
        # duplicating pipeline logic.
        captured = {}
        orig_post_process = pipeline._post_process

        def spy_post_process(question, answer, results, context, rerank_scores=None, _orig=orig_post_process):
            conf_result, ver_result, final_answer = _orig(
                question, answer, results, context, rerank_scores=rerank_scores
            )
            captured["results"] = results
            captured["rerank_scores"] = rerank_scores
            captured["raw_answer"] = answer
            captured["context"] = context
            captured["conf_result"] = conf_result
            captured["ver_result"] = ver_result
            return conf_result, ver_result, final_answer

        pipeline._post_process = spy_post_process
        try:
            resp = pipeline.query(q["question"])
        finally:
            pipeline._post_process = orig_post_process

        conf = captured.get("conf_result")
        ver = captured.get("ver_result")
        results = captured.get("results", [])

        # Ground truth vs retrieved
        expected = q.get("expected_articles", [])
        retrieved = [
            {
                "document_id": r.document_id,
                "document_title": r.document_title,
                "article": r.article,
                "score": round(r.score, 4) if r.score is not None else None,
            }
            for r in results
        ]

        print(f"\nPipeline used: {resp.pipeline_used}")
        print(f"Expected articles: {expected}")
        print(f"Retrieved ({len(retrieved)}):")
        for r in retrieved[:12]:
            print(f"  - doc={r['document_id']} article={r['article']} score={r['score']} title={r['document_title'][:50]}")

        print(f"\nRerank scores: {captured.get('rerank_scores')}")

        print(f"\n--- Generated answer (raw) ---\n{captured.get('raw_answer')}")

        print(f"\n--- Final response (post refusal gate) ---\nrefused={resp.refused}\nanswer={resp.answer[:300]}")

        if ver is not None:
            print(f"\n--- Verifier ---")
            print(f"citation_validity: {ver.citation_validity}")
            print(f"invalid_citations: {ver.invalid_citations}")
            print(f"claims ({len(ver.claims)}):")
            for c, lbl in zip(ver.claims, ver.entailment_labels):
                print(f"  [{lbl}] {c[:120]}")
            print(f"entailment_ratio={ver.entailment_ratio} contradiction_ratio={ver.contradiction_ratio}")
            print(f"grounding_ratio={ver.grounding_ratio} grounded_claims={ver.grounded_claims}")
            print(f"invalid_block_refs={ver.invalid_block_refs} ungrounded_article_citations={ver.ungrounded_article_citations}")
            print(f"warnings: {ver.warnings}")

        if conf is not None:
            print(f"\n--- Confidence breakdown ---")
            print(f"  retriever_score       = {conf.retriever_score:.4f}  (weight 0.20)")
            print(f"  reranker_score        = {conf.reranker_score:.4f}  (weight 0.20, available={captured.get('rerank_scores') is not None})")
            print(f"  llm_self_eval_score   = {conf.llm_self_eval_score:.4f}  (weight 0.25)")
            print(f"  context_overlap_score = {conf.context_overlap_score:.4f}  (weight 0.15)")
            print(f"  entailment_score      = {conf.entailment_score:.4f}  (weight 0.10, available={ver.entailment_labels != [] if ver else False})")
            print(f"  grounding_score       = {conf.grounding_score:.4f}  (weight 0.10, available={bool(ver.grounded_claims) if ver else False})")
            print(f"  => weighted score = {conf.score:.4f}  refused={conf.refused} (threshold={pipeline._confidence_scorer.refusal_threshold})")

        results_out[qid] = {
            "question": q["question"],
            "pipeline_used": resp.pipeline_used,
            "expected_articles": expected,
            "retrieved": retrieved,
            "rerank_scores": captured.get("rerank_scores"),
            "raw_answer": captured.get("raw_answer"),
            "final_answer": resp.answer,
            "refused": resp.refused,
            "verifier": {
                "citation_validity": ver.citation_validity if ver else None,
                "invalid_citations": ver.invalid_citations if ver else None,
                "claims": ver.claims if ver else None,
                "entailment_labels": ver.entailment_labels if ver else None,
                "entailment_ratio": ver.entailment_ratio if ver else None,
                "contradiction_ratio": ver.contradiction_ratio if ver else None,
                "grounding_ratio": ver.grounding_ratio if ver else None,
                "grounded_claims": ver.grounded_claims if ver else None,
                "invalid_block_refs": ver.invalid_block_refs if ver else None,
                "ungrounded_article_citations": ver.ungrounded_article_citations if ver else None,
                "warnings": ver.warnings if ver else None,
            } if ver else None,
            "confidence": {
                "retriever_score": conf.retriever_score,
                "reranker_score": conf.reranker_score,
                "llm_self_eval_score": conf.llm_self_eval_score,
                "context_overlap_score": conf.context_overlap_score,
                "entailment_score": conf.entailment_score,
                "grounding_score": conf.grounding_score,
                "score": conf.score,
                "refused": conf.refused,
            } if conf else None,
        }

    out_path = "data/eval_reports/inspect_confidence_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)
    print(f"\n\nSaved full dump to {out_path}")


if __name__ == "__main__":
    main()

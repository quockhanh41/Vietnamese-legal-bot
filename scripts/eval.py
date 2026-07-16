"""
Eval runner — Tuần 2 "Đo được" (docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md §4).

Two INDEPENDENT tiers:

  Tier 1 — Retrieval eval (default, NO LLM calls)
      Embeds each question, searches the vector store (dense AND hybrid, so the
      plan's "hybrid > dense" claim can actually be measured), and checks whether
      each expected {doc_id, article} appears in the top-5 / top-10 results.
      Reports recall@5, recall@10, MRR — both STRICT (payload field match) and
      LENIENT (article number found in chunk_text) — the gap between the two
      quantifies the known chunker metadata-misalignment issue.

  Tier 2 — Answer/refusal eval (--answers flag, USES LLM — costs money/rate limits)
      Runs AdaptiveRAGPipeline.query() (non-streaming path — includes confidence
      scoring + hallucination verification) per question. Measures refusal
      accuracy, false-refusal rate, citation accuracy, and dumps the confidence
      score distribution split by expected behavior (calibration data for the
      0.70 refusal threshold).

  TODO(ragas): RAGAS (faithfulness/answer-relevancy against ground_truth) is
  intentionally NOT implemented in this pass — optional deps + a separate LLM
  budget. Wire it up as a third tier once Tier 1/2 numbers establish a baseline.

Usage:
    python scripts/eval.py                                   # tier 1, criminal set, dense+hybrid
    python scripts/eval.py --set criminal --limit 3           # smoke test
    python scripts/eval.py --set general                      # tier 1 (no expected_articles -> 0 questions, reports that)
    python scripts/eval.py --answers --limit 5 --sleep 3       # tier 2 (LLM calls!)
    python scripts/eval.py --set general --answers             # tier 2 refusal-rate check on general set
"""

import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
from dotenv import load_dotenv

load_dotenv()

from src.data.constants import DEFAULT_COLLECTION_NAME
from src.data.embedder import BGEEmbedder
from src.data.mixed_embedder import build_embedder
from src.db.qdrant_client import LegalVectorStore, SearchResult
from src.eval.metrics import (
    citation_accuracy,
    false_refusal_rate,
    mrr,
    recall_at_k,
    refusal_accuracy,
)
from src.llm.gateway import LLMGateway
from src.rag.citation_lookup import CitationLookup, retrieve_routed
from src.rag.orchestrator import AdaptiveRAGPipeline

CRIMINAL_SET = "data/eval/eval_set_criminal_v1.json"
GENERAL_SET = "data/eval/eval_set_general_v1.json"
SECTORS_SET = "data/eval/eval_set_sectors_v2.json"  # M2.7 — v1 + Thông tư questions
CORPUS_PARQUET = "data/raw/corpus_600docs.parquet"  # M2.7 — doc_id -> sector, for leakage@k
REPORT_DIR = "data/eval_reports"

SET_PATHS = {
    "criminal": CRIMINAL_SET,
    "general": GENERAL_SET,
    "sectors": SECTORS_SET,
}
# Sets whose questions carry a single `expected_behavior` (grounded sets) vs
# the general set's refuse-now/answer-later pair of fields.
GROUNDED_SETS = {"criminal", "sectors"}

RETRIEVAL_K = (1, 3, 5, 10)
FETCH_K = max(RETRIEVAL_K)


# ── Loading ───────────────────────────────────────────────────────────────

def load_questions(set_name: str, limit: Optional[int]) -> List[Dict]:
    path = SET_PATHS[set_name]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if limit:
        questions = questions[:limit]
    return questions


def _result_to_dict(r: SearchResult) -> Dict:
    return {
        "document_id": r.document_id,
        "article": r.article,
        "chunk_text": r.chunk_text,
        "document_title": r.document_title,
        "score": r.score,
    }


def load_doc_sector(parquet_path: str = CORPUS_PARQUET) -> Dict[int, str]:
    """doc_id -> sector, loaded from corpus_600docs.parquet. Caller guards
    on file existence — returns {} if the parquet is missing."""
    if not os.path.exists(parquet_path):
        return {}
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["id", "sector"])
    return {int(row.id): row.sector for row in df.itertuples()}


def cross_sector_leakage(hit_doc_ids: List[int], query_sector: str, doc_sector: Dict[int, str], k: int) -> float:
    """Fraction of the top-k hits whose doc's sector != query_sector.
    Hits whose doc_id is not in doc_sector (out-of-corpus) are ignored."""
    known = [doc_sector[d] for d in hit_doc_ids[:k] if d in doc_sector]
    if not known:
        return 0.0
    foreign = sum(1 for s in known if s != query_sector)
    return foreign / len(known)


# ── Tier 1: Retrieval eval ────────────────────────────────────────────────

def run_tier1(questions: List[Dict], vector_store: LegalVectorStore, embedder: BGEEmbedder) -> Dict:
    """Retrieval-only eval: no LLM calls. Returns {"dense": {...}, "hybrid": {...}, "routed": {...}}.

    "routed" is the citation-lookup route (src/rag/citation_lookup.py): for
    citation-style questions ("Điều 173 BLHS 2015 quy định về tội gì?") it
    uses direct metadata lookup + dense fill instead of vector search; for
    every other question it falls back to hybrid search (same as the
    "hybrid" mode) — see retrieve_routed() docstring.
    """
    eligible = [
        q for q in questions
        if q.get("expected_behavior") == "answer" and q.get("expected_articles")
    ]
    skipped = len(questions) - len(eligible)

    doc_sector = load_doc_sector()  # {} if corpus_600docs.parquet is missing

    rows_by_mode = {"dense": [], "hybrid": [], "routed": []}
    citation_lookup = CitationLookup(vector_store)

    for q in eligible:
        output = embedder.encode([q["question"]], return_dense=True, return_sparse=True)
        dense_vec = output["dense_vecs"][0]
        sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}

        dense_hits = vector_store.search_dense(dense_vec.tolist(), top_k=FETCH_K)
        hybrid_hits = vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=FETCH_K)
        routed_hits, route, _parsed = retrieve_routed(
            q["question"], vector_store, embedder, citation_lookup, fill_top_k=FETCH_K,
        )

        for mode, hits in (("dense", dense_hits), ("hybrid", hybrid_hits), ("routed", routed_hits)):
            hit_dicts = [_result_to_dict(r) for r in hits]
            expected = q["expected_articles"]
            row = {
                "id": q["id"],
                "type": q["type"],
                "difficulty": q["difficulty"],
                "legal_sector": q.get("legal_sector"),
                "expected_articles": expected,
                "top_hits": [
                    {"document_id": h["document_id"], "article": h["article"], "score": round(h["score"], 4)}
                    for h in hit_dicts[:5]
                ],
            }
            if mode == "routed":
                row["route"] = route
                if doc_sector and q.get("legal_sector"):
                    row["leakage@5"] = cross_sector_leakage(
                        [h["document_id"] for h in hit_dicts], q["legal_sector"], doc_sector, 5,
                    )
            for k in RETRIEVAL_K:
                row[f"recall@{k}_strict"] = recall_at_k(hit_dicts, expected, k, lenient=False)
                row[f"recall@{k}_lenient"] = recall_at_k(hit_dicts, expected, k, lenient=True)
            row["rr_strict"] = mrr(hit_dicts, expected, lenient=False)
            row["rr_lenient"] = mrr(hit_dicts, expected, lenient=True)
            rows_by_mode[mode].append(row)

    return {
        "n_questions_total": len(questions),
        "n_questions_eligible": len(eligible),
        "n_questions_skipped": skipped,
        "modes": {mode: _aggregate_tier1(rows) for mode, rows in rows_by_mode.items()},
        "rows": rows_by_mode,
    }


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.mean(values), 4) if values else None


def _aggregate_tier1(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0}

    agg: Dict = {"n": len(rows)}
    for k in RETRIEVAL_K:
        agg[f"recall@{k}_strict"] = _mean([r[f"recall@{k}_strict"] for r in rows])
        agg[f"recall@{k}_lenient"] = _mean([r[f"recall@{k}_lenient"] for r in rows])
    agg["mrr_strict"] = _mean([r["rr_strict"] for r in rows])
    agg["mrr_lenient"] = _mean([r["rr_lenient"] for r in rows])

    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    agg["by_type"] = {
        t: {
            "n": len(trows),
            **{f"recall@{k}_strict": _mean([r[f"recall@{k}_strict"] for r in trows]) for k in RETRIEVAL_K},
            **{f"recall@{k}_lenient": _mean([r[f"recall@{k}_lenient"] for r in trows]) for k in RETRIEVAL_K},
            "mrr_strict": _mean([r["rr_strict"] for r in trows]),
            "mrr_lenient": _mean([r["rr_lenient"] for r in trows]),
        }
        for t, trows in sorted(by_type.items())
    }

    # Per-sector breakdown — Gate M1 requires per-sector recall reported
    # separately (docs/Plan_Giai_Doan_Cuoi_2026-07-10.md). Only the sectors
    # set carries legal_sector; criminal/general rows have None and are
    # skipped, leaving by_sector empty (not printed) for those sets.
    by_sector: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        sec = r.get("legal_sector")
        if sec:
            by_sector[sec].append(r)
    agg["by_sector"] = {
        s: {
            "n": len(srows),
            **{f"recall@{k}_strict": _mean([r[f"recall@{k}_strict"] for r in srows]) for k in RETRIEVAL_K},
            **{f"recall@{k}_lenient": _mean([r[f"recall@{k}_lenient"] for r in srows]) for k in RETRIEVAL_K},
            "mrr_strict": _mean([r["rr_strict"] for r in srows]),
            "mrr_lenient": _mean([r["rr_lenient"] for r in srows]),
            "leakage@5": _mean([r["leakage@5"] for r in srows if "leakage@5" in r]),
        }
        for s, srows in sorted(by_sector.items())
    }
    return agg


# ── Tier 2: Answer/refusal eval ───────────────────────────────────────────

def row_citation_accuracy(
    expected_behavior: Optional[str],
    refused: bool,
    answer_text: str,
    expected_articles: List[Dict],
) -> Optional[float]:
    """Citation accuracy for one Tier-2 row, or None when not measurable.

    None when the question isn't an answer-test OR the bot refused: a
    refusal carries no citations to grade and is already priced into
    false_refusal_rate — scoring it 0 here double-counts the failure and
    conflates two metrics (M1 gate discovery, 2026-07-12: 10 of 16
    zero-citation sectors rows were runtime refusals, dragging the true
    ~0.83 down to a reported 0.67).
    """
    if expected_behavior != "answer" or refused:
        return None
    return citation_accuracy(answer_text, expected_articles)

def run_tier2(
    questions: List[Dict],
    pipeline: AdaptiveRAGPipeline,
    set_name: str,
    sleep_s: float,
) -> Dict:
    """Answer/refusal eval — calls the LLM through AdaptiveRAGPipeline.query()."""
    behavior_field = "expected_behavior" if set_name in GROUNDED_SETS else "expected_behavior_current"

    rows = []
    for i, q in enumerate(questions):
        expected_behavior = q.get(behavior_field)
        expected_articles = q.get("expected_articles", [])

        t0 = time.time()
        error = None
        try:
            resp = pipeline.query(q["question"])
        except Exception as e:
            resp = None
            error = str(e)
        elapsed_ms = (time.time() - t0) * 1000

        row = {
            "id": q["id"],
            "type": q["type"],
            "expected_behavior": expected_behavior,
            "error": error,
        }
        if resp is not None:
            row.update({
                "refused": resp.refused,
                "confidence_score": round(resp.confidence_score, 4),
                "confidence_label": resp.confidence,
                "pipeline_used": resp.pipeline_used,
                "latency_ms": round(resp.latency_ms or elapsed_ms, 1),
                "citation_accuracy": row_citation_accuracy(
                    expected_behavior, resp.refused, resp.answer, expected_articles
                ),
            })
        else:
            row.update({
                "refused": None,
                "confidence_score": None,
                "confidence_label": None,
                "pipeline_used": None,
                "latency_ms": round(elapsed_ms, 1),
                "citation_accuracy": None,
            })
        rows.append(row)
        click.echo(f"  [{i+1}/{len(questions)}] {q['id']} -> "
                   f"refused={row['refused']} conf={row['confidence_score']} error={error}")

        if i < len(questions) - 1 and sleep_s > 0:
            time.sleep(sleep_s)

    return {"aggregate": aggregate_tier2(rows), "rows": rows}


def aggregate_tier2(rows: List[Dict]) -> Dict:
    """Compute the Tier-2 aggregate block from per-question rows.

    Pure — extracted from run_tier2 so the --retry merge path recomputes
    aggregates over merged rows with identical logic.
    """
    ok_rows = [r for r in rows if r["error"] is None]
    conf_by_behavior = defaultdict(list)
    for r in ok_rows:
        if r["confidence_score"] is not None:
            conf_by_behavior[r["expected_behavior"]].append(r["confidence_score"])

    cite_scores = [r["citation_accuracy"] for r in ok_rows if r["citation_accuracy"] is not None]

    return {
        "n": len(rows),
        "n_errors": len(rows) - len(ok_rows),
        "refusal_accuracy": refusal_accuracy(ok_rows),
        "false_refusal_rate": false_refusal_rate(ok_rows),
        "mean_citation_accuracy": _mean(cite_scores),
        "confidence_distribution": {
            behavior: {
                "n": len(scores),
                "mean": _mean(scores),
                "min": round(min(scores), 4) if scores else None,
                "max": round(max(scores), 4) if scores else None,
                "median": round(statistics.median(scores), 4) if scores else None,
            }
            for behavior, scores in sorted(conf_by_behavior.items())
        },
    }


# ── Tier 2 retry/merge (Groq free-tier TPD dies mid-run — re-run ONLY the
# errored questions instead of re-burning every successful question's tokens) ─

def failed_question_ids(questions: List[Dict], prev_rows: List[Dict]) -> List[str]:
    """IDs needing a re-run: rows with error != None, plus set questions
    that have no row at all (previous run crashed early). Set order."""
    ok_ids = {r["id"] for r in prev_rows if r.get("error") is None}
    return [q["id"] for q in questions if q["id"] not in ok_ids]


def merge_tier2_rows(
    questions: List[Dict], prev_rows: List[Dict], new_rows: List[Dict]
) -> List[Dict]:
    """Merge retry rows into a previous run's rows, in set-question order.

    A new row always wins for its question id (even if it errored again —
    the fresh error is the current truth); otherwise the previous row is
    carried over. Questions with no row in either run are skipped.
    """
    prev_by_id = {r["id"]: r for r in prev_rows}
    new_by_id = {r["id"]: r for r in new_rows}
    merged = []
    for q in questions:
        row = new_by_id.get(q["id"]) or prev_by_id.get(q["id"])
        if row is not None:
            merged.append(row)
    return merged


# ── Report writing + printing ─────────────────────────────────────────────

def write_report(report: Dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(REPORT_DIR, f"{ts}_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def print_tier1_summary(tier1: Dict, set_name: str):
    click.echo(f"\n## Tier 1 — Retrieval eval ({set_name})\n")
    click.echo(f"Câu hỏi: {tier1['n_questions_total']} tổng, "
                f"{tier1['n_questions_eligible']} đủ điều kiện "
                f"(answer + có expected_articles), {tier1['n_questions_skipped']} bỏ qua.\n")

    if tier1["n_questions_eligible"] == 0:
        click.echo("⚠️ Không có câu hỏi nào đủ điều kiện đo retrieval "
                    "(cần expected_behavior=answer và expected_articles non-empty).")
        return

    for mode in ("dense", "hybrid", "routed"):
        agg = tier1["modes"][mode]
        click.echo(f"### Mode: {mode} (n={agg['n']})\n")
        click.echo("| metric | strict | lenient |")
        click.echo("|---|---|---|")
        for k in RETRIEVAL_K:
            click.echo(f"| recall@{k} | {agg[f'recall@{k}_strict']} | {agg[f'recall@{k}_lenient']} |")
        click.echo(f"| MRR | {agg['mrr_strict']} | {agg['mrr_lenient']} |")
        click.echo("")
        click.echo("| type | n | recall@5 (strict) | recall@10 (strict) | MRR (strict) |")
        click.echo("|---|---|---|---|---|")
        for t, tagg in agg["by_type"].items():
            click.echo(f"| {t} | {tagg['n']} | {tagg['recall@5_strict']} | "
                        f"{tagg['recall@10_strict']} | {tagg['mrr_strict']} |")
        click.echo("")

        # Per-sector recall — Gate M1 artifact (only present for the sectors set).
        # leakage@5 (cross-sector leakage, frequency-bias diagnostic — M2.7)
        # is only populated for the routed mode; None elsewhere.
        by_sector = agg.get("by_sector") or {}
        if by_sector:
            click.echo("| lĩnh vực | n | recall@5 (strict) | recall@10 (strict) | MRR (strict) | leakage@5 |")
            click.echo("|---|---|---|---|---|---|")
            for s, sagg in by_sector.items():
                click.echo(f"| {s} | {sagg['n']} | {sagg['recall@5_strict']} | "
                            f"{sagg['recall@10_strict']} | {sagg['mrr_strict']} | {sagg['leakage@5']} |")
            click.echo("")

    d5 = tier1["modes"]["dense"]["recall@5_strict"] or 0
    h5 = tier1["modes"]["hybrid"]["recall@5_strict"] or 0
    r5 = tier1["modes"]["routed"]["recall@5_strict"] or 0
    verdict = "hybrid > dense ✅" if h5 > d5 else ("hybrid == dense" if h5 == d5 else "hybrid < dense ⚠️")
    click.echo(f"So sánh recall@5 (strict): dense={d5} vs hybrid={h5} -> {verdict}")

    routed_verdict = "routed > hybrid ✅" if r5 > h5 else ("routed == hybrid" if r5 == h5 else "routed < hybrid ⚠️")
    click.echo(f"So sánh recall@5 (strict): routed={r5} vs hybrid={h5} -> {routed_verdict}")

    # citation-lookup is the type the routed mode targets specifically —
    # break it out so the "did the route fix the known failure" claim is
    # checked directly, not just averaged in with every other type.
    cl_hybrid = tier1["modes"]["hybrid"]["by_type"].get("citation-lookup")
    cl_routed = tier1["modes"]["routed"]["by_type"].get("citation-lookup")
    if cl_hybrid and cl_routed:
        click.echo(
            f"citation-lookup recall@5 (strict): hybrid={cl_hybrid['recall@5_strict']} "
            f"-> routed={cl_routed['recall@5_strict']} "
            f"(MRR {cl_hybrid['mrr_strict']} -> {cl_routed['mrr_strict']})"
        )
    sh_hybrid = tier1["modes"]["hybrid"]["by_type"].get("single-hop")
    sh_routed = tier1["modes"]["routed"]["by_type"].get("single-hop")
    if sh_hybrid and sh_routed:
        click.echo(
            f"single-hop recall@5 (strict, must NOT regress): hybrid={sh_hybrid['recall@5_strict']} "
            f"-> routed={sh_routed['recall@5_strict']}"
        )


def print_tier2_summary(tier2: Dict, set_name: str):
    agg = tier2["aggregate"]
    click.echo(f"\n## Tier 2 — Answer/refusal eval ({set_name})\n")
    click.echo(f"Câu hỏi: {agg['n']} (lỗi: {agg['n_errors']})\n")
    click.echo(f"- Refusal accuracy (đúng khi phải từ chối): {agg['refusal_accuracy']}")
    click.echo(f"- False-refusal rate (từ chối nhầm câu đáng lẽ trả lời được): {agg['false_refusal_rate']}")
    click.echo(f"- Citation accuracy trung bình (câu đáng lẽ trả lời): {agg['mean_citation_accuracy']}\n")
    click.echo("### Phân bố confidence_score theo expected_behavior (calibration data cho ngưỡng 0.70)\n")
    click.echo("| expected_behavior | n | mean | median | min | max |")
    click.echo("|---|---|---|---|---|---|")
    for behavior, d in agg["confidence_distribution"].items():
        click.echo(f"| {behavior} | {d['n']} | {d['mean']} | {d['median']} | {d['min']} | {d['max']} |")
    click.echo("\nTODO(ragas): RAGAS faithfulness/answer-relevancy chưa chạy trong pass này "
               "(deps optional + cần ngân sách LLM riêng) — xem docstring đầu file.")


def run_retry(retry_report: str, sleep_s: float):
    """--retry path: re-run only the errored/missing questions of a previous
    tier2_answers report and write a merged report (config.retry_of chains
    back to the source, so provenance survives multiple mop-up passes)."""
    with open(retry_report, encoding="utf-8") as f:
        prev = json.load(f)
    prev_config = prev.get("config", {})
    if prev_config.get("tier") != "tier2_answers" or "tier2" not in prev:
        raise click.ClickException(f"{retry_report} không phải report tier2_answers.")

    set_name = prev_config["set"]
    collection = prev_config["collection"]
    prev_rows = prev["tier2"]["rows"]

    all_questions = load_questions(set_name, limit=None)
    retry_ids = failed_question_ids(all_questions, prev_rows)
    click.echo(f"🔁 Retry từ {retry_report}: set={set_name}, collection={collection}, "
               f"{len(retry_ids)}/{len(all_questions)} câu cần chạy lại.")
    if not retry_ids:
        click.echo("✅ Không còn câu lỗi — không có gì để chạy.")
        return

    retry_questions = [q for q in all_questions if q["id"] in set(retry_ids)]

    vector_store = LegalVectorStore(collection_name=collection)
    click.echo("⏳ Loading BGE-M3 embedder (~2GB RAM)...")
    embedder = BGEEmbedder()
    llm = LLMGateway()
    pipeline = AdaptiveRAGPipeline(vector_store=vector_store, embedder=embedder, llm_gateway=llm)

    tier2_new = run_tier2(retry_questions, pipeline, set_name, sleep_s)

    merged_rows = merge_tier2_rows(all_questions, prev_rows, tier2_new["rows"])
    tier2 = {"aggregate": aggregate_tier2(merged_rows), "rows": merged_rows}

    config = {
        "set": set_name,
        "limit": None,
        "collection": collection,
        "tier": "tier2_answers",
        "sleep_s": sleep_s,
        "retry_of": retry_report,
        "n_retried": len(retry_questions),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report = {"config": config, "tier2": tier2}
    print_tier2_summary(tier2, set_name)
    path = write_report(report)
    click.echo(f"\n💾 Merged report saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--set", "set_name", type=click.Choice(["criminal", "general", "sectors"]), default="criminal",
              help="Bộ eval để chạy.")
@click.option("--limit", type=int, default=None, help="Chỉ chạy N câu hỏi đầu tiên (smoke test).")
@click.option("--answers", is_flag=True, default=False,
              help="Chạy Tier 2 (answer/refusal, dùng LLM — tốn tiền/rate limit) thay vì Tier 1.")
@click.option("--sleep", "sleep_s", type=float, default=2.0,
              help="Giây nghỉ giữa các câu hỏi ở Tier 2 (tránh rate limit Groq).")
@click.option("--collection", default=DEFAULT_COLLECTION_NAME, help="Tên Qdrant collection.")
@click.option("--embedder", "embedder_type", type=click.Choice(["bge-m3", "mixed"]),
              default="bge-m3", help="bge-m3 (default) hoặc mixed (AITeamVN dense + BGE-M3 sparse).")
@click.option("--embed-model", default=None,
              help="Model dense cho --embedder mixed (mặc định AITeamVN/Vietnamese_Embedding).")
@click.option("--reranker", "reranker_model", default=None,
              help="Reranker model cho Tier-2 (mặc định BAAI/bge-reranker-v2-m3). "
                   "Ví dụ: AITeamVN/Vietnamese_Reranker.")
@click.option("--retry", "retry_report", type=click.Path(exists=True), default=None,
              help="Đường dẫn report tier2_answers cũ: chạy lại CHỈ các câu lỗi "
                   "(Groq TPD chết giữa chừng), merge vào rows cũ, ghi report mới. "
                   "set/collection lấy từ config của report — bỏ qua --set/--collection.")
def main(set_name: str, limit: Optional[int], answers: bool, sleep_s: float,
         collection: str, embedder_type: str, embed_model: Optional[str],
         reranker_model: Optional[str], retry_report: Optional[str]):
    if retry_report:
        run_retry(retry_report, sleep_s)
        return

    questions = load_questions(set_name, limit)
    click.echo(f"📊 Eval set: {set_name} ({len(questions)} câu hỏi"
               f"{' — limit áp dụng' if limit else ''})")

    vector_store = LegalVectorStore(collection_name=collection)
    click.echo("⏳ Loading BGE-M3 embedder (~2GB RAM)...")
    embedder = build_embedder(embedder_type, embed_model, device="cpu")

    config = {
        "set": set_name,
        "limit": limit,
        "collection": collection,
        "tier": "tier2_answers" if answers else "tier1_retrieval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report = {"config": config}

    if answers:
        config["sleep_s"] = sleep_s
        llm = LLMGateway()
        rr_kwargs = {"reranker_model": reranker_model} if reranker_model else {}
        pipeline = AdaptiveRAGPipeline(vector_store=vector_store, embedder=embedder,
                                       llm_gateway=llm, **rr_kwargs)
        tier2 = run_tier2(questions, pipeline, set_name, sleep_s)
        report["tier2"] = tier2
        print_tier2_summary(tier2, set_name)
    else:
        tier1 = run_tier1(questions, vector_store, embedder)
        report["tier1"] = tier1
        print_tier1_summary(tier1, set_name)

    path = write_report(report)
    click.echo(f"\n💾 Report saved: {path}")


if __name__ == "__main__":
    main()

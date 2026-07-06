"""
Validate eval sets against the ingested corpus.

Checks:
  1. Schema: required fields, unique IDs, valid enum values.
  2. Grounding (criminal set): every expected_articles entry must exist in the
     chunk data (data/raw/luat_law_chunks.json) — both doc_id and article number.
  3. Stats: question counts by type/difficulty/behavior.

Usage:
    python scripts/validate_eval_set.py                     # validate both sets
    python scripts/validate_eval_set.py --chunks-file path  # custom chunk dump
"""

import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click

CRIMINAL_SET = "data/eval/eval_set_criminal_v1.json"
GENERAL_SET = "data/eval/eval_set_general_v1.json"
CHUNKS_FILE = "data/raw/luat_law_chunks.json"

VALID_TYPES = {
    "citation-lookup", "single-hop", "multi-hop", "comparison",
    "amendment", "out-of-scope",
}
VALID_DIFFICULTY = {"easy", "medium", "hard"}


def artnum(article: str) -> str | None:
    """Extract the article number from 'Điều 123' / 'Điều 123. Tội ...'"""
    m = re.search(r"\d+[a-z]?", article or "")
    return m.group() if m else None


def build_corpus_index(chunks_path: str) -> dict[int, set[str]]:
    """Map doc_id -> set of article numbers present in chunk metadata OR chunk text."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    index: dict[int, set[str]] = {}
    for c in chunks:
        did = c.get("document_id")
        if did is None:
            continue
        arts = index.setdefault(did, set())
        n = artnum(c.get("article", ""))
        if n:
            arts.add(n)
        # Articles can also appear inside chunk text (amendment laws quote them)
        for m in re.finditer(r"Điều\s+(\d+[a-z]?)\s*[\.:]", c.get("chunk_text", "")):
            arts.add(m.group(1))
    return index


def validate_schema(questions: list[dict], id_prefix: str, behavior_field: str) -> list[str]:
    errors = []
    seen_ids = set()
    for q in questions:
        qid = q.get("id", "<missing>")
        if qid in seen_ids:
            errors.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        if not qid.startswith(id_prefix):
            errors.append(f"{qid}: id must start with '{id_prefix}'")
        if not q.get("question", "").strip():
            errors.append(f"{qid}: empty question")
        if q.get("type") not in VALID_TYPES:
            errors.append(f"{qid}: invalid type '{q.get('type')}'")
        if q.get("difficulty") not in VALID_DIFFICULTY:
            errors.append(f"{qid}: invalid difficulty '{q.get('difficulty')}'")

        behavior = q.get(behavior_field)
        if behavior not in ("answer", "refuse"):
            errors.append(f"{qid}: {behavior_field} must be answer|refuse, got '{behavior}'")
        if behavior == "answer" and not q.get("ground_truth", "").strip():
            errors.append(f"{qid}: expected_behavior=answer but ground_truth is empty")
        if behavior == "refuse" and q.get("expected_articles"):
            errors.append(f"{qid}: expected_behavior=refuse but expected_articles is non-empty")
    return errors


def validate_grounding(questions: list[dict], corpus: dict[int, set[str]]) -> list[str]:
    """Check every expected_articles entry exists in the ingested corpus."""
    errors = []
    for q in questions:
        for ea in q.get("expected_articles", []):
            did = ea.get("doc_id")
            n = artnum(ea.get("article", ""))
            if did not in corpus:
                errors.append(f"{q['id']}: doc_id {did} not in corpus")
            elif n not in corpus[did]:
                errors.append(f"{q['id']}: '{ea.get('article')}' (số {n}) not found in doc {did}")
    return errors


def print_stats(name: str, questions: list[dict], behavior_field: str):
    print(f"\n📊 {name}: {len(questions)} câu hỏi")
    for field in ("type", "difficulty", behavior_field, "legal_sector"):
        counts = Counter(q.get(field, "?") for q in questions)
        pretty = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"   {field:28s} {pretty}")


@click.command()
@click.option("--chunks-file", default=CHUNKS_FILE, help="Chunk dump for grounding check")
def main(chunks_file: str):
    all_errors = []

    # ── Criminal set: schema + grounding ──
    with open(CRIMINAL_SET, encoding="utf-8") as f:
        crim = json.load(f)["questions"]
    errors = validate_schema(crim, "crim-", "expected_behavior")

    if os.path.exists(chunks_file):
        corpus = build_corpus_index(chunks_file)
        errors += validate_grounding(crim, corpus)
        docs = ", ".join(str(d) for d in sorted(corpus))
        print(f"✅ Corpus index: {len(corpus)} docs ({docs})")
    else:
        print(f"⚠️ {chunks_file} not found — skipping grounding check")

    print_stats("eval_set_criminal_v1", crim, "expected_behavior")
    all_errors += errors

    # ── General set: schema only (corpus not ingested yet) ──
    with open(GENERAL_SET, encoding="utf-8") as f:
        gen = json.load(f)["questions"]
    errors = validate_schema(gen, "gen-", "expected_behavior_current")
    unverified = sum(1 for q in gen if not q.get("ground_truth_verified", False))
    print_stats("eval_set_general_v1", gen, "expected_behavior_current")
    print(f"   ground_truth chưa verify với corpus: {unverified}/{len(gen)} "
          f"(bình thường — verify khi ingest required_laws)")
    all_errors += errors

    # ── Result ──
    print()
    if all_errors:
        print(f"❌ {len(all_errors)} lỗi:")
        for e in all_errors:
            print(f"   - {e}")
        sys.exit(1)
    print("✅ Cả 2 eval set hợp lệ.")


if __name__ == "__main__":
    main()

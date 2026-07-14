"""
Validate eval sets against the ingested corpus.

Checks:
  1. Schema: required fields, unique IDs, valid enum values.
  2. Grounding (criminal set): every expected_articles entry must exist in the
     chunk data (data/raw/luat_law_chunks.json) — both doc_id and article number.
  3. Grounding (sectors set): every expected_articles entry must exist as an
     "Điều N." header inside the corresponding document's content column in
     data/raw/corpus_600docs.parquet.
  4. Stats: question counts by type/difficulty/behavior.

Usage:
    python scripts/validate_eval_set.py                     # validate all 3 sets
    python scripts/validate_eval_set.py --chunks-file path   # custom chunk dump
    python scripts/validate_eval_set.py --corpus-parquet data/raw/corpus_600docs.parquet
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
SECTORS_SET = "data/eval/eval_set_sectors_v2.json"  # M2.7 — v1 + Thông tư questions
CHUNKS_FILE = "data/raw/luat_law_chunks.json"
CORPUS_PARQUET = "data/raw/corpus_600docs.parquet"  # M2.7 — matches SECTORS_SET v2

VALID_TYPES = {
    "citation-lookup", "single-hop", "multi-hop", "comparison",
    "amendment", "out-of-scope",
}
VALID_DIFFICULTY = {"easy", "medium", "hard"}

# Header regex shared by the parquet-grounding check: an "Điều N." line start,
# tolerating a leading curly/angled quote and a letter suffix on the number
# (e.g. "25a", "48b" — articles inserted by amendment laws).
ARTICLE_HEADER_RE = re.compile(r'(?:^|\n)\s*[“"«]?\s*Điều\s+(\d+[a-z]?)\s*[\.:]')


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


def build_corpus_index_parquet(parquet_path: str) -> dict[int, set[str]]:
    """Map doc_id -> set of article numbers found as 'Điều N.' headers at the
    start of a line in that document's content (data/raw/corpus_600docs.parquet).

    Tolerates a leading curly/angled quote (amendment laws quote the new
    article text, e.g. '"Điều 25a. ...') and a letter suffix on the article
    number (articles inserted by amendment laws, e.g. Điều 25a, Điều 48b).
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    index: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        did = row["id"]
        content = row.get("content") or ""
        arts = {m.group(1) for m in ARTICLE_HEADER_RE.finditer(content)}
        if arts:
            index[did] = arts
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


def validate_grounding(questions: list[dict], corpus: dict[int, set[str]], label: str = "") -> list[str]:
    """Check every expected_articles entry exists in the given corpus index."""
    errors = []
    prefix = f"[{label}] " if label else ""
    for q in questions:
        for ea in q.get("expected_articles", []):
            did = ea.get("doc_id")
            n = artnum(ea.get("article", ""))
            if did not in corpus:
                errors.append(f"{prefix}{q['id']}: doc_id {did} not in corpus")
            elif n not in corpus[did]:
                errors.append(f"{prefix}{q['id']}: '{ea.get('article')}' (số {n}) not found in doc {did}")
    return errors


def print_stats(name: str, questions: list[dict], behavior_field: str):
    print(f"\n📊 {name}: {len(questions)} câu hỏi")
    for field in ("type", "difficulty", behavior_field, "legal_sector"):
        counts = Counter(q.get(field, "?") for q in questions)
        pretty = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"   {field:28s} {pretty}")


@click.command()
@click.option("--chunks-file", default=CHUNKS_FILE, help="Chunk dump for grounding check (criminal set)")
@click.option("--corpus-parquet", default=CORPUS_PARQUET, help="corpus_600docs.parquet for grounding check (sectors set)")
def main(chunks_file: str, corpus_parquet: str):
    all_errors = []

    chunk_corpus: dict[int, set[str]] = {}
    if os.path.exists(chunks_file):
        chunk_corpus = build_corpus_index(chunks_file)
        docs = ", ".join(str(d) for d in sorted(chunk_corpus))
        print(f"✅ Chunk corpus index ({chunks_file}): {len(chunk_corpus)} docs ({docs})")
    else:
        print(f"⚠️ {chunks_file} not found — skipping chunk-based grounding check")

    parquet_corpus: dict[int, set[str]] = {}
    if os.path.exists(corpus_parquet):
        parquet_corpus = build_corpus_index_parquet(corpus_parquet)
        print(f"✅ Parquet corpus index ({corpus_parquet}): {len(parquet_corpus)} docs với header Điều")
    else:
        print(f"⚠️ {corpus_parquet} not found — skipping parquet-based grounding check")

    # ── Criminal set: schema + grounding (chunks-file, falling back to parquet) ──
    with open(CRIMINAL_SET, encoding="utf-8") as f:
        crim = json.load(f)["questions"]
    errors = validate_schema(crim, "crim-", "expected_behavior")
    if chunk_corpus:
        errors += validate_grounding(crim, chunk_corpus, label="criminal/chunks")
    elif parquet_corpus:
        errors += validate_grounding(crim, parquet_corpus, label="criminal/parquet")
    print_stats("eval_set_criminal_v1", crim, "expected_behavior")
    all_errors += errors

    # ── General set: schema only (expected_articles not used in this schema) ──
    with open(GENERAL_SET, encoding="utf-8") as f:
        gen = json.load(f)["questions"]
    errors = validate_schema(gen, "gen-", "expected_behavior_current")
    if parquet_corpus:
        errors += validate_grounding(gen, parquet_corpus, label="general/parquet")
    unverified = sum(1 for q in gen if not q.get("ground_truth_verified", False))
    print_stats("eval_set_general_v1", gen, "expected_behavior_current")
    print(f"   ground_truth chưa verify với corpus: {unverified}/{len(gen)} "
          f"(bình thường cho câu refuse/chưa review lại sau khi flip — xem field flipped_at)")
    all_errors += errors

    # ── Sectors set: schema + mandatory parquet grounding ──
    if os.path.exists(SECTORS_SET):
        with open(SECTORS_SET, encoding="utf-8") as f:
            sectors = json.load(f)["questions"]
        prefixes = {"ds-", "ld-", "dn-", "dd-", "hg-", "hs-"}
        errors = []
        seen_ids = set()
        for q in sectors:
            qid = q.get("id", "<missing>")
            if qid in seen_ids:
                errors.append(f"{qid}: duplicate id")
            seen_ids.add(qid)
            if not any(qid.startswith(p) for p in prefixes):
                errors.append(f"{qid}: id must start with one of {sorted(prefixes)}")
            if not q.get("question", "").strip():
                errors.append(f"{qid}: empty question")
            if q.get("type") not in VALID_TYPES:
                errors.append(f"{qid}: invalid type '{q.get('type')}'")
            if q.get("difficulty") not in VALID_DIFFICULTY:
                errors.append(f"{qid}: invalid difficulty '{q.get('difficulty')}'")
            if q.get("expected_behavior") != "answer":
                errors.append(f"{qid}: expected_behavior must be 'answer' in eval_set_sectors_v2")
            if not q.get("ground_truth", "").strip():
                errors.append(f"{qid}: ground_truth is empty")
            if not q.get("expected_articles"):
                errors.append(f"{qid}: expected_articles is empty")
        if parquet_corpus:
            errors += validate_grounding(sectors, parquet_corpus, label="sectors/parquet")
        elif not os.path.exists(corpus_parquet):
            errors.append("eval_set_sectors_v2: --corpus-parquet not found — cannot verify grounding")
        print_stats("eval_set_sectors_v2", sectors, "expected_behavior")
        all_errors += errors
    else:
        print(f"⚠️ {SECTORS_SET} not found — skipping")

    # ── Result ──
    print()
    if all_errors:
        print(f"❌ {len(all_errors)} lỗi:")
        for e in all_errors:
            print(f"   - {e}")
        sys.exit(1)
    print("✅ Cả 3 eval set hợp lệ.")


if __name__ == "__main__":
    main()

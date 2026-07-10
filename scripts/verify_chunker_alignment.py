"""
Local, offline gate for the HierarchicalChunker metadata-alignment fix
(M0.4b, roadmap §6.1 #3). No Qdrant / embedding / network access — reads
raw document content straight from the local parquet snapshot and runs the
(fixed) chunkers in-process.

Background: Fable measured the production `luat_law_ctx` collection
directly (1,670 points, all sourced from the same 6 documents used here)
and found 126 points (7.5%) whose `article` payload label did not match
the Điều actually present in `chunk_text` ("WRONG_LABEL"). Root cause and
fix are documented in docs/notes/chunker-misalignment-diagnosis-2026-07-10.md.
This script reproduces that measurement locally against the FIXED chunker
so the fix can be gated without touching Qdrant Cloud.

Checks performed, per document and in aggregate:
  1. Label-alignment classification (mirrors Fable's production audit):
       HEADER_OK        chunk_text opens with the header matching `article`
       MID_ARTICLE       chunk_text has no Điều header at all (by design —
                          Khoản-level chunk under an Điều's metadata)
       WRONG_LABEL       chunk_text contains a DIFFERENT Điều's header
                          than the `article` label says (the bug)
       NO_ARTICLE_LABEL  `article` is empty (harmless preamble chunks)
     Gate: WRONG_LABEL == 0 everywhere.
  2. Article-count preservation (Bug 3): distinct Điều headers found in the
     raw document content vs. distinct Điều labels found in the chunker's
     output — must match; anything in the first set but missing from the
     second means an Điều silently vanished from the index.
  3. Parent-text homogeneity (Bug 4): every chunk sharing the same
     (document, article) must carry an IDENTICAL parent_chunk_text, and
     that shared text must start with the Điều's own header.

Usage:
    .venv/bin/python scripts/verify_chunker_alignment.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.data.chunker  # noqa: F401  (import order workaround, see tests/)
from src.data.chunking.hierarchical import HierarchicalChunker
from src.data.chunking.article_level import ArticleLevelChunker


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "test_6docs.parquet")

# Same amendment-law detection ingest.py uses to pick the chunker per doc.
AMENDMENT_RE = re.compile(r'(?:sửa\s+đổi|bổ\s+sung)', re.IGNORECASE)

# Exact regex specified for this gate: header anchored to line start,
# optional single-letter suffix tolerated (e.g. "Điều 105a.").
HEADER_RE = re.compile(r'(?:^|\n)\s*Điều\s+(\d+[a-z]?)\s*[\.:]', re.IGNORECASE)

# Quote-aware variant, mirroring ArticleLevelChunker's M1.2b contract:
# amendment laws open the first quoted article of a block with a curly
# quote (“Điều 43a. …) which the chunker treats as a boundary, keeping the
# quote in chunk_text but stripping it from the `article` label. Docs
# chunked by ArticleLevelChunker are therefore audited with this regex;
# HierarchicalChunker docs keep the strict HEADER_RE (its pattern does not
# treat quoted lines as boundaries, and original — non-amendment — laws
# contain no such headers).
HEADER_RE_QUOTED = re.compile(
    r'(?:^|\n)\s*[“"«]?\s*Điều\s+(\d+[a-z]?)\s*[\.:]', re.IGNORECASE
)

# Same shape, used to pull the article number out of a chunk's `article`
# metadata label (e.g. "Điều 21. Người bào chữa" -> "21").
LABEL_NUM_RE = re.compile(r'Điều\s+(\d+[a-z]?)', re.IGNORECASE)


def load_documents(data_path=DATA_PATH):
    """Load raw documents from a local parquet snapshot (default: the
    6-doc criminal-law snapshot this gate was built around). Returns
    (rows, error) — error is a human-readable string if loading failed;
    rows is a list of dicts either way (may be shorter/empty on failure)."""
    if not os.path.exists(data_path):
        return [], f"raw data file not found: {data_path}"

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        return [], f"pyarrow not available: {e}"

    table = pq.read_table(data_path)
    df = table.to_pandas()
    rows = [row.to_dict() for _, row in df.iterrows()]

    if data_path == DATA_PATH and len(rows) < 6:
        return rows, (
            f"expected 6 documents, found {len(rows)} in {data_path} — "
            f"report below covers only what was found."
        )
    return rows, None


def article_num_from_label(label):
    if not label:
        return None
    m = LABEL_NUM_RE.search(label)
    return m.group(1).lower() if m else None


def classify_chunk(article_label, chunk_text, header_re=HEADER_RE):
    """Mirrors the production audit's 4-way classification."""
    if not article_label:
        return "NO_ARTICLE_LABEL"

    label_num = article_num_from_label(article_label)
    headers = [h.lower() for h in header_re.findall(chunk_text)]

    if any(h != label_num for h in headers):
        return "WRONG_LABEL"
    if headers and header_re.match(chunk_text):
        # First header in the text occurs at position 0 -> chunk opens
        # with its own Điều's header.
        return "HEADER_OK"
    return "MID_ARTICLE"


def distinct_articles_in_content(content, header_re=HEADER_RE):
    return {h.lower() for h in header_re.findall(content)}


def distinct_articles_in_chunks(chunks):
    return {
        article_num_from_label(c.article)
        for c in chunks
        if c.article is not None and article_num_from_label(c.article) is not None
    }


def check_parent_homogeneity(chunks, header_re=HEADER_RE):
    """Group chunks by article label; every group must share one identical
    parent_chunk_text that starts with that Điều's own header. Returns a
    list of violation strings (empty if all good)."""
    groups = {}
    for c in chunks:
        if not c.article:
            # Falsy (None or "") means NO_ARTICLE_LABEL — e.g. preamble
            # chunks that precede the first Điều. There's no Điều to be
            # homogeneous with; consistent with classify_chunk's
            # `if not article_label` check above.
            continue
        groups.setdefault(c.article, []).append(c)

    violations = []
    for label, group in groups.items():
        parents = {c.parent_chunk_text for c in group}
        if len(parents) != 1:
            violations.append(
                f"article={label!r}: {len(parents)} distinct parent_chunk_text "
                f"values across {len(group)} chunks (expected 1)"
            )
            continue
        parent = next(iter(parents))
        num = article_num_from_label(label)
        m = header_re.match(parent or "")
        if not m or m.group(1).lower() != num:
            violations.append(
                f"article={label!r}: parent_chunk_text does not start with "
                f"its own Điều header — parent[:60]={parent[:60]!r}"
            )
    return violations


def run(data_path=DATA_PATH, quiet=False):
    rows, load_error = load_documents(data_path)
    if load_error:
        print(f"WARNING: {load_error}\n")
    if not rows:
        print("No documents loaded — cannot run the alignment gate.")
        return 1

    hierarchical = HierarchicalChunker(min_chunk_size=200, max_chunk_size=1000)
    article_level = ArticleLevelChunker(min_article_length=100)

    overall_counts = {"HEADER_OK": 0, "MID_ARTICLE": 0, "WRONG_LABEL": 0, "NO_ARTICLE_LABEL": 0}
    overall_total_chunks = 0
    overall_wrong_label_examples = []

    # This task's scope (and gate) is HierarchicalChunker's Bug 1/3/4 fixes.
    # ArticleLevelChunker is explicitly out of scope (per task instructions,
    # not to be modified here) — its results are still measured for
    # visibility below, but failures there are reported as a separate note
    # and do NOT fail this gate.
    hier_article_count_ok = True
    hier_parent_violations = []
    non_gating_notes = []

    print("=" * 88)
    print("HierarchicalChunker metadata-alignment gate (local, offline)")
    print("=" * 88)

    for row in rows:
        title = row.get("title", "<untitled>")
        content = row.get("content", "") or ""
        is_amend = bool(AMENDMENT_RE.search(title))
        chunker_name = "ArticleLevelChunker" if is_amend else "HierarchicalChunker"
        chunker = article_level if is_amend else hierarchical

        chunks = chunker.chunk_document(row)
        overall_total_chunks += len(chunks)

        # Amendment docs are audited quote-aware, mirroring the chunker
        # contract they were split with (see HEADER_RE_QUOTED above).
        doc_header_re = HEADER_RE_QUOTED if is_amend else HEADER_RE

        counts = {"HEADER_OK": 0, "MID_ARTICLE": 0, "WRONG_LABEL": 0, "NO_ARTICLE_LABEL": 0}
        wrong_label_examples = []
        for c in chunks:
            cls = classify_chunk(c.article, c.chunk_text, header_re=doc_header_re)
            counts[cls] += 1
            overall_counts[cls] += 1
            if cls == "WRONG_LABEL" and len(wrong_label_examples) < 3:
                wrong_label_examples.append(
                    f"    label={c.article!r} text[:70]={c.chunk_text[:70]!r}"
                )
        overall_wrong_label_examples.extend(wrong_label_examples)

        content_articles = distinct_articles_in_content(content, header_re=doc_header_re)
        chunk_articles = distinct_articles_in_chunks(chunks)
        missing = content_articles - chunk_articles
        extra = chunk_articles - content_articles
        article_count_ok = not missing

        parent_violations = check_parent_homogeneity(chunks, header_re=doc_header_re)

        if chunker_name == "HierarchicalChunker":
            if not article_count_ok:
                hier_article_count_ok = False
            hier_parent_violations.extend(f"[{title}] {v}" for v in parent_violations)
        else:
            if not article_count_ok:
                non_gating_notes.append(
                    f"[{title}] ({chunker_name}, out of scope for this gate) "
                    f"dropped Điều {sorted(missing)} — same 'short article "
                    f"below min-length threshold' pattern as Bug 3, but in "
                    f"ArticleLevelChunker.flush_article (min_article_length), "
                    f"not touched by this fix per task scope."
                )
            if parent_violations:
                non_gating_notes.extend(
                    f"[{title}] ({chunker_name}, out of scope) {v}" for v in parent_violations
                )

        # In quiet mode (large corpora) only documents with a finding get
        # a per-doc block; clean docs are folded into the TOTAL section.
        has_finding = bool(
            counts["WRONG_LABEL"] or missing or parent_violations
        )
        if quiet and not has_finding:
            continue

        print(f"\n--- {title}  ({chunker_name}) ---")
        print(f"  chunks: {len(chunks)}")
        print(
            f"  HEADER_OK={counts['HEADER_OK']}  MID_ARTICLE={counts['MID_ARTICLE']}  "
            f"WRONG_LABEL={counts['WRONG_LABEL']}  NO_ARTICLE_LABEL={counts['NO_ARTICLE_LABEL']}"
        )
        if wrong_label_examples:
            print("  WRONG_LABEL examples:")
            for ex in wrong_label_examples:
                print(ex)
        print(
            f"  distinct Điều in content: {len(content_articles)}  |  "
            f"distinct Điều in chunk labels: {len(chunk_articles)}  |  "
            f"{'OK' if article_count_ok else 'MISMATCH'}"
        )
        if missing:
            print(f"    MISSING from output (dropped Điều!): {sorted(missing)}")
        if extra:
            # Not necessarily a bug (e.g. a header regex false-positive
            # inside a chunk, or ArticleLevelChunker's whole-doc fallback
            # label) but surfaced for visibility.
            print(f"    extra labels not found as a header in raw content: {sorted(extra)}")
        print(
            f"  parent_chunk_text homogeneity: "
            f"{'OK' if not parent_violations else f'{len(parent_violations)} VIOLATIONS'}"
        )
        for v in parent_violations:
            print(f"    {v}")

    print("\n" + "=" * 88)
    print("TOTAL (all documents)")
    print("=" * 88)
    print(f"  total chunks: {overall_total_chunks}")
    print(
        f"  HEADER_OK={overall_counts['HEADER_OK']}  "
        f"MID_ARTICLE={overall_counts['MID_ARTICLE']}  "
        f"WRONG_LABEL={overall_counts['WRONG_LABEL']}  "
        f"NO_ARTICLE_LABEL={overall_counts['NO_ARTICLE_LABEL']}"
    )
    print("  (production baseline before fix, full luat_law_ctx collection, "
          "1670 points: HEADER_OK=193 MID_ARTICLE=1346 WRONG_LABEL=126 "
          "NO_ARTICLE_LABEL=5)")

    # Gate is scoped to this task: WRONG_LABEL==0 is checked across ALL
    # chunks (both chunkers — it happens to hold for ArticleLevelChunker
    # too, which the diagnosis already found clean on this axis); the
    # "no Điều dropped" / "parent homogeneous" checks are scoped to
    # HierarchicalChunker specifically, since that's the chunker this task
    # fixes (Bug 1/3/4). ArticleLevelChunker findings are reported below as
    # non-gating notes.
    gate_pass = (
        overall_counts["WRONG_LABEL"] == 0
        and hier_article_count_ok
        and not hier_parent_violations
    )

    print("\n" + "-" * 88)
    print(f"GATE: WRONG_LABEL == 0 (all docs)            -> {'PASS' if overall_counts['WRONG_LABEL'] == 0 else 'FAIL'}")
    print(f"GATE: no Điều dropped (HierarchicalChunker)  -> {'PASS' if hier_article_count_ok else 'FAIL'}")
    print(f"GATE: parent_chunk_text homogeneous (Hier.)  -> {'PASS' if not hier_parent_violations else 'FAIL'}")
    print("-" * 88)
    print(f"OVERALL (this task's scope): {'PASS' if gate_pass else 'FAIL'}")

    if non_gating_notes:
        print("\n" + "-" * 88)
        print("NON-GATING NOTES (ArticleLevelChunker — out of scope for this fix,")
        print("reported per task instructions: 'lỗi liên đới' found by this script's")
        print("new Bug-3-style checks, not modified here):")
        print("-" * 88)
        for note in non_gating_notes:
            print(f"  - {note}")

    if load_error:
        print(f"\nNOTE: {load_error}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default=DATA_PATH,
        help="Parquet snapshot to gate (default: the 6-doc criminal set; "
             "M1 dùng data/raw/corpus_100docs.parquet)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Chỉ in per-doc block cho document có finding (dùng cho corpus lớn)",
    )
    args = parser.parse_args()
    sys.exit(run(data_path=args.data_path, quiet=args.quiet))

"""
Download + validate the Zalo AI Challenge 2021 Legal Text Retrieval dataset
(M3.1 — external benchmark; spec docs/superpowers/specs/2026-07-15-m3-external-benchmark-design.md).

Primary source: HF mirror (verified faithful reconstruction — corpus 61,425 rows
and queries 3,196 rows match the official Zalo 2021 figures exactly):
    .venv/bin/python scripts/download_zalo.py --hf
Caveat: the mirror's qrels carry exactly 1 relevant article per query (the
original Zalo qrels had ~1% multi-relevant questions) — flagged for the writeup.
Fallback: Kaggle mirror (needs `kaggle` CLI configured):
    .venv/bin/python scripts/download_zalo.py --kaggle hariwh0/zaloai2021-legal-text-retrieval
Fallback: any manually downloaded zip of the original files:
    .venv/bin/python scripts/download_zalo.py --zip ~/Downloads/zac2021-ltr-data.zip
Validate-only (files already in place):
    .venv/bin/python scripts/download_zalo.py --validate-only

Whatever the source, the result must land as
    data/benchmarks/zalo/legal_corpus.json
    data/benchmarks/zalo/train_question_answer.json
and pass count validation (61,425 articles / 3,271 laws / 3,196 questions)
— guards against truncated/repackaged mirrors. After validation the script
freezes the 80/10/10 question split via src.eval.external_bench.make_splits
(Task 2) unless splits.json already exists (never overwritten).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ZALO_DIR = "data/benchmarks/zalo"
CORPUS_FILE = os.path.join(ZALO_DIR, "legal_corpus.json")
QA_FILE = os.path.join(ZALO_DIR, "train_question_answer.json")
# laws corrected 2026-07-15 (8586 was a conflation with the Kien et al. corpus,
# arXiv:2209.14494; 3271 measured on the verified HF mirror)
EXPECTED = {"laws": 3271, "articles": 61425, "questions": 3196}

HF_DEFAULT_SLUG = "minhnguyent546/zalo-ai-legal-text-retrieval-2021"
HF_DIR = os.path.join(ZALO_DIR, "hf")
HF_FILES = [
    "corpus/train-00000-of-00001.parquet",
    "queries/train-00000-of-00001.parquet",
    "qrels/train-00000-of-00001.parquet",
    "qrels/test-00000-of-00001.parquet",
]


class ZaloValidationError(Exception):
    pass


def load_zalo_corpus(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_zalo_questions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"] if isinstance(data, dict) else data


def validate_zalo(corpus, questions, expected_laws, expected_articles, expected_questions):
    """Count validation + referential integrity of qrels. Raises on any mismatch."""
    n_laws = len(corpus)
    n_articles = sum(len(law["articles"]) for law in corpus)
    n_questions = len(questions)
    actual = {"laws": n_laws, "articles": n_articles, "questions": n_questions}
    expected = {"laws": expected_laws, "articles": expected_articles,
                "questions": expected_questions}
    if actual != expected:
        raise ZaloValidationError(f"Count mismatch: expected {expected}, got {actual}")

    known = {(law["law_id"], art["article_id"])
             for law in corpus for art in law["articles"]}
    for q in questions:
        for ra in q["relevant_articles"]:
            key = (ra["law_id"], ra["article_id"])
            if key not in known:
                raise ZaloValidationError(
                    f"Question {q.get('question_id')} references unknown article {key}")
    return actual


def corpus_rows_to_laws(rows):
    """rows: iterable of {"id": "law#art", "title": str, "text": str} →
    raw-format corpus: [{"law_id", "articles": [{"article_id","title","text"}]}].
    Split id with rsplit("#", 1). Group by law_id preserving first-appearance order;
    article order preserved within each law."""
    articles_by_law = {}
    law_order = []
    for row in rows:
        law_id, article_id = row["id"].rsplit("#", 1)
        if law_id not in articles_by_law:
            articles_by_law[law_id] = []
            law_order.append(law_id)
        articles_by_law[law_id].append(
            {"article_id": article_id, "title": row["title"], "text": row["text"]})
    return [{"law_id": law_id, "articles": articles_by_law[law_id]} for law_id in law_order]


def build_questions(query_rows, qrel_rows):
    """query_rows: iterable of {"query_id", "question"};
    qrel_rows: iterable of {"corpus_id": "law#art", "query_id"} →
    [{"question_id", "question", "relevant_articles": [{"law_id","article_id"}]}]
    in query_rows order. A query may have MULTIPLE qrels → multiple relevant_articles
    (preserve qrel order). A query with zero qrels gets an empty list (do not drop it)."""
    relevant_by_query = {}
    for qrel in qrel_rows:
        law_id, article_id = qrel["corpus_id"].rsplit("#", 1)
        relevant_by_query.setdefault(qrel["query_id"], []).append(
            {"law_id": law_id, "article_id": article_id})
    return [
        {"question_id": q["query_id"], "question": q["question"],
         "relevant_articles": relevant_by_query.get(q["query_id"], [])}
        for q in query_rows
    ]


def _fetch_hf(slug):
    base_url = f"https://huggingface.co/datasets/{slug}/resolve/main/"
    for rel_path in HF_FILES:
        dest = os.path.join(HF_DIR, rel_path)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(base_url + rel_path, dest)


def _reconstruct_from_hf(slug):
    import pandas as pd

    _fetch_hf(slug)
    corpus_rows = pd.read_parquet(
        os.path.join(HF_DIR, "corpus/train-00000-of-00001.parquet")).to_dict("records")
    query_rows = pd.read_parquet(
        os.path.join(HF_DIR, "queries/train-00000-of-00001.parquet")).to_dict("records")
    qrel_rows = (
        pd.read_parquet(os.path.join(HF_DIR, "qrels/train-00000-of-00001.parquet")).to_dict("records")
        + pd.read_parquet(os.path.join(HF_DIR, "qrels/test-00000-of-00001.parquet")).to_dict("records")
    )

    laws = corpus_rows_to_laws(corpus_rows)
    questions = build_questions(query_rows, qrel_rows)

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(laws, f, ensure_ascii=False)
    with open(QA_FILE, "w", encoding="utf-8") as f:
        json.dump({"items": questions}, f, ensure_ascii=False)


def _fetch_kaggle(dataset_slug):
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug,
         "-p", ZALO_DIR, "--unzip"],
        check=True,
    )


def _extract_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ZALO_DIR)


def _locate_and_place():
    """Mirrors nest files differently — find the two canonical files anywhere
    under ZALO_DIR and move them to the canonical paths."""
    for canonical, basename in [(CORPUS_FILE, "legal_corpus.json"),
                                (QA_FILE, "train_question_answer.json")]:
        if os.path.exists(canonical):
            continue
        hits = glob.glob(os.path.join(ZALO_DIR, "**", basename), recursive=True)
        if not hits:
            raise ZaloValidationError(
                f"{basename} not found under {ZALO_DIR} after download/extract")
        shutil.move(hits[0], canonical)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", nargs="?", const=HF_DEFAULT_SLUG, metavar="SLUG",
                     help="HF dataset slug to download + reconstruct from "
                          f"(default: {HF_DEFAULT_SLUG})")
    src.add_argument("--kaggle", metavar="SLUG",
                     help="Kaggle dataset slug to download via kaggle CLI")
    src.add_argument("--zip", metavar="PATH", help="Path to a manually downloaded zip")
    src.add_argument("--validate-only", action="store_true",
                     help="Files already at canonical paths; just validate + split")
    args = ap.parse_args()

    os.makedirs(ZALO_DIR, exist_ok=True)
    if args.hf:
        _reconstruct_from_hf(args.hf)
    elif args.kaggle:
        _fetch_kaggle(args.kaggle)
        _locate_and_place()
    elif args.zip:
        _extract_zip(args.zip)
        _locate_and_place()

    corpus = load_zalo_corpus(CORPUS_FILE)
    questions = load_zalo_questions(QA_FILE)
    stats = validate_zalo(corpus, questions,
                          expected_laws=EXPECTED["laws"],
                          expected_articles=EXPECTED["articles"],
                          expected_questions=EXPECTED["questions"])
    print(f"✅ Zalo dataset validated: {stats}")

    # Freeze split (Task 2 module). Refuses to overwrite — the split is a
    # measurement contract shared with M3.2 fine-tuning.
    from src.eval.external_bench import make_splits, SPLITS_FILE
    if os.path.exists(SPLITS_FILE):
        print(f"splits.json already exists — left untouched ({SPLITS_FILE})")
    else:
        splits = make_splits([q["question_id"] for q in questions])
        with open(SPLITS_FILE, "w", encoding="utf-8") as f:
            json.dump(splits, f, ensure_ascii=False, indent=1)
        print(f"✅ Wrote {SPLITS_FILE}: "
              f"{ {k: len(v) for k, v in splits.items() if isinstance(v, list)} }")


if __name__ == "__main__":
    main()

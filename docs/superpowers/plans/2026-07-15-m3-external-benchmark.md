# M3.1 External Benchmark (Zalo/ALQAC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure our real retrieval stack (BGE-M3 dense / hybrid RRF / + BGE-reranker) on the public Zalo Legal Text Retrieval 2021 benchmark — recall@{1,5,10}, MRR@10, nDCG@10 — with a 3-level ablation and a frozen train/dev/test split that M3.2 fine-tuning will reuse.

**Architecture:** Zalo's 61K-article corpus is indexed article-level (their ground-truth unit) into a LOCAL docker Qdrant collection `zalo_legal_2021`, reusing `LegalVectorStore` for collection/upsert/search so the measured code paths are the production ones. Embedding runs offline (Kaggle GPU or local MPS) into parquet shards; a runner script embeds queries, runs the 3 ablation levels, and writes a JSON report next to the internal eval reports.

**Tech Stack:** Python, FlagEmbedding (BGEM3FlagModel via `src/data/embedder.BGEEmbedder`, FlagReranker `BAAI/bge-reranker-v2-m3`), qdrant-client (`src/db/qdrant_client.LegalVectorStore`), pandas/pyarrow, pytest.

## Global Constraints

- **$0**: free-tier only; local docker Qdrant (`http://localhost:6333`), no cloud spend.
- **Production untouched**: no changes to `DEFAULT_COLLECTION_NAME`, `REFUSAL_THRESHOLD` (0.700), routing, or the Qdrant Cloud collections.
- **Expected dataset counts (hard validation)**: legal corpus **61,425 articles / 8,586 laws**; train Q-A **3,196 questions**. Mismatch = hard fail with actual counts printed.
- **Split seed 20260715**, ratios 80/10/10, persisted to `data/benchmarks/zalo/splits.json` (committed). Never regenerated over an existing file.
- **Retrieval-only**: no LLM calls anywhere in this milestone item.
- `make test` (pytest tests/ -v) stays green; new tests are pure (no network, no models, no Qdrant).
- Spec: `docs/superpowers/specs/2026-07-15-m3-external-benchmark-design.md`.

---

### Task 1: Zalo dataset download + validation

**Files:**
- Create: `scripts/download_zalo.py`
- Create: `tests/test_zalo_data.py`
- Modify: `.gitignore` (data dir rules)

**Interfaces:**
- Produces: `load_zalo_corpus(path) -> List[Dict]` (list of laws: `{"law_id": str, "articles": [{"article_id","title","text"}]}`), `load_zalo_questions(path) -> List[Dict]` (`{"question_id","question","relevant_articles":[{"law_id","article_id"}]}`), `validate_zalo(corpus, questions) -> Dict` in `scripts/download_zalo.py` — Tasks 2/4/5 import the loaders.
- Data on disk: `data/benchmarks/zalo/legal_corpus.json`, `data/benchmarks/zalo/train_question_answer.json` (gitignored).

- [ ] **Step 1: Add gitignore rules** (data files are large; only `splits.json` gets committed)

Append to `.gitignore` after the existing `data/` block (lines 13-16):

```
data/benchmarks/*
!data/benchmarks/zalo
data/benchmarks/zalo/*
!data/benchmarks/zalo/splits.json
```

- [ ] **Step 2: Write the failing tests** — `tests/test_zalo_data.py`:

```python
"""
Unit tests for scripts/download_zalo.py — loaders + count validation.
Pure: tiny in-memory fixtures, no network, no real dataset.

Run: .venv/bin/python -m pytest tests/test_zalo_data.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_zalo import (
    load_zalo_corpus,
    load_zalo_questions,
    validate_zalo,
    ZaloValidationError,
)

CORPUS = [
    {"law_id": "01/2020/qh14", "articles": [
        {"article_id": "1", "title": "Điều 1", "text": "Nội dung điều 1."},
        {"article_id": "2", "title": "Điều 2", "text": "Nội dung điều 2."},
    ]},
    {"law_id": "02/2021/nd-cp", "articles": [
        {"article_id": "5", "title": "Điều 5", "text": "Nội dung điều 5."},
    ]},
]

QUESTIONS_ITEMS_WRAPPED = {"items": [
    {"question_id": "q1", "question": "Câu hỏi 1?",
     "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "2"}]},
]}

QUESTIONS_FLAT = [
    {"question_id": "q1", "question": "Câu hỏi 1?",
     "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "2"}]},
]


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_load_corpus(tmp_path):
    corpus = load_zalo_corpus(_write(tmp_path, "c.json", CORPUS))
    assert len(corpus) == 2
    assert corpus[0]["law_id"] == "01/2020/qh14"
    assert corpus[0]["articles"][1]["article_id"] == "2"


def test_load_questions_items_wrapped(tmp_path):
    qs = load_zalo_questions(_write(tmp_path, "q.json", QUESTIONS_ITEMS_WRAPPED))
    assert len(qs) == 1
    assert qs[0]["relevant_articles"] == [{"law_id": "01/2020/qh14", "article_id": "2"}]


def test_load_questions_flat_list(tmp_path):
    qs = load_zalo_questions(_write(tmp_path, "q.json", QUESTIONS_FLAT))
    assert len(qs) == 1


def test_validate_counts_ok():
    stats = validate_zalo(CORPUS, QUESTIONS_FLAT,
                          expected_laws=2, expected_articles=3, expected_questions=1)
    assert stats == {"laws": 2, "articles": 3, "questions": 1}


def test_validate_counts_mismatch_raises():
    with pytest.raises(ZaloValidationError) as e:
        validate_zalo(CORPUS, QUESTIONS_FLAT,
                      expected_laws=99, expected_articles=3, expected_questions=1)
    assert "99" in str(e.value) and "2" in str(e.value)  # expected vs actual in message


def test_validate_rejects_question_with_unknown_article():
    bad_q = [{"question_id": "qX", "question": "?",
              "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "404"}]}]
    with pytest.raises(ZaloValidationError):
        validate_zalo(CORPUS, bad_q,
                      expected_laws=2, expected_articles=3, expected_questions=1)
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_zalo_data.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.download_zalo'`

- [ ] **Step 4: Implement `scripts/download_zalo.py`**

```python
"""
Download + validate the Zalo AI Challenge 2021 Legal Text Retrieval dataset
(M3.1 — external benchmark; spec docs/superpowers/specs/2026-07-15-m3-external-benchmark-design.md).

Primary source: Kaggle mirror (needs `kaggle` CLI configured):
    .venv/bin/python scripts/download_zalo.py --kaggle hariwh0/zaloai2021-legal-text-retrieval
Fallback: any manually downloaded zip of the original files:
    .venv/bin/python scripts/download_zalo.py --zip ~/Downloads/zac2021-ltr-data.zip
Validate-only (files already in place):
    .venv/bin/python scripts/download_zalo.py --validate-only

Whatever the source, the result must land as
    data/benchmarks/zalo/legal_corpus.json
    data/benchmarks/zalo/train_question_answer.json
and pass count validation (61,425 articles / 8,586 laws / 3,196 questions)
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
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ZALO_DIR = "data/benchmarks/zalo"
CORPUS_FILE = os.path.join(ZALO_DIR, "legal_corpus.json")
QA_FILE = os.path.join(ZALO_DIR, "train_question_answer.json")
EXPECTED = {"laws": 8586, "articles": 61425, "questions": 3196}


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
    src.add_argument("--kaggle", metavar="SLUG",
                     help="Kaggle dataset slug to download via kaggle CLI")
    src.add_argument("--zip", metavar="PATH", help="Path to a manually downloaded zip")
    src.add_argument("--validate-only", action="store_true",
                     help="Files already at canonical paths; just validate + split")
    args = ap.parse_args()

    os.makedirs(ZALO_DIR, exist_ok=True)
    if args.kaggle:
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
```

Note: Step 4 imports `make_splits` inside `main()` only — the module-level functions Task 1's tests cover have no dependency on Task 2, so tests pass before Task 2 exists; `main()` is only runnable after Task 2 (execution order below respects that).

- [ ] **Step 5: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_zalo_data.py -q`
Expected: 6 passed

- [ ] **Step 6: Run the full suite**

Run: `make test`
Expected: all pass (349+ tests, plus the 6 new)

- [ ] **Step 7: Commit**

```bash
git add .gitignore scripts/download_zalo.py tests/test_zalo_data.py
git commit -m "feat(M3.1): Zalo benchmark downloader with count + qrel validation"
```

---

### Task 2: Frozen 80/10/10 split module

**Files:**
- Create: `src/eval/external_bench.py`
- Create: `tests/test_external_bench.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure).
- Produces: `make_splits(question_ids: List[str], seed: int = 20260715, ratios=(0.8, 0.1, 0.1)) -> Dict` returning `{"seed": int, "ratios": [f,f,f], "train": [id...], "dev": [id...], "test": [id...]}`; constant `SPLITS_FILE = "data/benchmarks/zalo/splits.json"`; `load_splits(path=SPLITS_FILE) -> Dict`. Task 1's `main()` and Task 5's runner import these.

- [ ] **Step 1: Write the failing tests** — `tests/test_external_bench.py`:

```python
"""
Unit tests for src/eval/external_bench.py — frozen split for the Zalo benchmark.
Pure functions; determinism is the whole point.

Run: .venv/bin/python -m pytest tests/test_external_bench.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.external_bench import make_splits

IDS = [f"q{i}" for i in range(100)]


def test_split_sizes_80_10_10():
    s = make_splits(IDS)
    assert len(s["train"]) == 80
    assert len(s["dev"]) == 10
    assert len(s["test"]) == 10


def test_split_disjoint_and_complete():
    s = make_splits(IDS)
    union = set(s["train"]) | set(s["dev"]) | set(s["test"])
    assert union == set(IDS)
    assert len(s["train"]) + len(s["dev"]) + len(s["test"]) == len(IDS)


def test_split_deterministic_across_calls_and_input_order():
    a = make_splits(IDS)
    b = make_splits(list(reversed(IDS)))  # input order must not matter
    assert a["train"] == b["train"]
    assert a["dev"] == b["dev"]
    assert a["test"] == b["test"]


def test_split_records_seed_and_ratios():
    s = make_splits(IDS)
    assert s["seed"] == 20260715
    assert s["ratios"] == [0.8, 0.1, 0.1]


def test_split_different_seed_differs():
    assert make_splits(IDS, seed=1)["train"] != make_splits(IDS, seed=2)["train"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_external_bench.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.external_bench'`

- [ ] **Step 3: Implement `src/eval/external_bench.py`**

```python
"""
Frozen train/dev/test split for external benchmarks (M3.1).

The split is a measurement contract: M3.1 reports zero-shot numbers on the
full query set AND on the held-out test split; M3.2 fine-tuning may train
on `train`, tune on `dev`, and compares against M3.1 ONLY on `test`.
Regenerating the split invalidates that comparison — which is why
scripts/download_zalo.py refuses to overwrite an existing splits.json.
"""
import json
import random
from typing import Dict, List

SPLITS_FILE = "data/benchmarks/zalo/splits.json"
DEFAULT_SEED = 20260715


def make_splits(
    question_ids: List[str],
    seed: int = DEFAULT_SEED,
    ratios=(0.8, 0.1, 0.1),
) -> Dict:
    """Deterministic shuffle-then-slice. Input order must not matter, so ids
    are sorted before shuffling with the seeded RNG."""
    ids = sorted(question_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    return {
        "seed": seed,
        "ratios": list(ratios),
        "train": ids[:n_train],
        "dev": ids[n_train:n_train + n_dev],
        "test": ids[n_train + n_dev:],
    }


def load_splits(path: str = SPLITS_FILE) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_external_bench.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/external_bench.py tests/test_external_bench.py
git commit -m "feat(M3.1): frozen 80/10/10 benchmark split (seed 20260715)"
```

- [ ] **Step 6: EXECUTION — download the real dataset** (needs Tasks 1+2 code; kaggle CLI or manual zip)

```bash
.venv/bin/python scripts/download_zalo.py --kaggle hariwh0/zaloai2021-legal-text-retrieval
```
If the kaggle CLI isn't configured or that mirror's contents don't validate, download the original archive manually (Kaggle web UI for the same slug, or the links in github.com/CuongNN218/zalo_ltr_2021) and run with `--zip <path>` instead. **Do not relax the expected counts to make a mirror fit** — find a faithful mirror.

Expected output: `✅ Zalo dataset validated: {'laws': 8586, 'articles': 61425, 'questions': 3196}` then `✅ Wrote data/benchmarks/zalo/splits.json: {'train': 2556, 'dev': 319, 'test': 321}`.

```bash
git add data/benchmarks/zalo/splits.json
git commit -m "data(M3.1): freeze Zalo benchmark split (2556/319/321, seed 20260715)"
```

---

### Task 3: ID-based retrieval metrics (recall@k, MRR@10, nDCG@10)

**Files:**
- Create: `src/eval/bench_metrics.py`
- Create: `tests/test_bench_metrics.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: in `src/eval/bench_metrics.py`, all operating on exact ID tuples — `retrieved: List[Tuple[str, str]]` ((law_id, article_id), ranked) and `relevant: Set[Tuple[str, str]]`:
  - `recall_at_k(retrieved, relevant, k: int) -> float`
  - `mrr_at_k(retrieved, relevant, k: int = 10) -> float`
  - `ndcg_at_k(retrieved, relevant, k: int = 10) -> float` (binary relevance)

  Deliberately separate from `src/eval/metrics.py` (internal eval): that module matches via legal-domain strict/lenient `any_match` semantics; the benchmark is exact-ID binary relevance. Task 5 imports these.

- [ ] **Step 1: Write the failing tests** — `tests/test_bench_metrics.py`:

```python
"""
Unit tests for src/eval/bench_metrics.py — exact-ID retrieval metrics for
external benchmarks. Hand-computed expectations documented inline.

Run: .venv/bin/python -m pytest tests/test_bench_metrics.py -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.bench_metrics import recall_at_k, mrr_at_k, ndcg_at_k

REL = {("L1", "A1"), ("L2", "A2")}
# ranks:      1            2            3
RETRIEVED = [("L9", "A9"), ("L1", "A1"), ("L2", "A2")]


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, REL, k=1) == 0.0
    assert recall_at_k(RETRIEVED, REL, k=2) == 0.5   # found L1:A1 only
    assert recall_at_k(RETRIEVED, REL, k=3) == 1.0


def test_recall_empty_relevant_is_vacuous():
    assert recall_at_k(RETRIEVED, set(), k=3) == 1.0


def test_mrr_first_hit_at_rank_2():
    assert mrr_at_k(RETRIEVED, REL, k=10) == 0.5


def test_mrr_no_hit_within_k():
    assert mrr_at_k(RETRIEVED, REL, k=1) == 0.0


def test_ndcg_hand_computed():
    # DCG  = 1/log2(2+1) + 1/log2(3+1)          (hits at ranks 2 and 3)
    # IDCG = 1/log2(1+1) + 1/log2(2+1)          (2 relevant, ideal ranks 1,2)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, REL, k=10) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_1():
    assert ndcg_at_k([("L1", "A1"), ("L2", "A2")], REL, k=10) == pytest.approx(1.0)


def test_ndcg_no_hits_is_0():
    assert ndcg_at_k([("L9", "A9")], REL, k=10) == 0.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bench_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.bench_metrics'`

- [ ] **Step 3: Implement `src/eval/bench_metrics.py`**

```python
"""
Exact-ID retrieval metrics for external benchmarks (M3.1 — Zalo/ALQAC).

Operates on (law_id, article_id) tuples with binary relevance. Kept separate
from src/eval/metrics.py, whose recall/MRR use the internal eval's
legal-domain strict/lenient matching (any_match) — different semantics.
"""
import math
from typing import List, Set, Tuple

IdPair = Tuple[str, str]


def recall_at_k(retrieved: List[IdPair], relevant: Set[IdPair], k: int) -> float:
    """Fraction of relevant ids found in the top-k. Vacuously 1.0 when
    relevant is empty (mirrors src/eval/metrics.recall_at_k)."""
    if not relevant:
        return 1.0
    found = sum(1 for r in relevant if r in set(retrieved[:k]))
    return found / len(relevant)


def mrr_at_k(retrieved: List[IdPair], relevant: Set[IdPair], k: int = 10) -> float:
    """Reciprocal rank (1-indexed) of the first relevant id within top-k."""
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[IdPair], relevant: Set[IdPair], k: int = 10) -> float:
    """Binary-relevance nDCG@k: DCG over hit ranks / ideal DCG."""
    if not relevant:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bench_metrics.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/bench_metrics.py tests/test_bench_metrics.py
git commit -m "feat(M3.1): exact-ID benchmark metrics (recall@k, MRR@10, nDCG@10)"
```

---

### Task 4: Corpus embed (offline shards) + local Qdrant index

**Files:**
- Create: `scripts/ingest_zalo_corpus.py`
- Create: `tests/test_zalo_ingest.py`

**Interfaces:**
- Consumes: `load_zalo_corpus` from `scripts/download_zalo.py` (Task 1); `BGEEmbedder.encode_documents(texts, batch_size) -> (dense_np, lexical_weights)` from `src/data/embedder.py`; `LegalVectorStore(url=..., collection_name=...)`, `.create_collection()`, `.upsert_chunks(chunk_texts, dense_vectors, sparse_vectors, metadata_list)`, `.get_collection_info()` from `src/db/qdrant_client.py`.
- Produces: `flatten_corpus(corpus) -> List[Dict]` where each row is `{"embed_text": str, "metadata": {...}}` (unit-tested); parquet shards `data/benchmarks/zalo/vectors_NNNN.parquet`; populated local collection `zalo_legal_2021`.
- **Payload contract (Task 5 depends on it):** each point's metadata maps benchmark ids onto EXISTING SearchResult fields so `search_dense`/`search_hybrid` return them without src changes: `document_number = law_id`, `article = article_id`, `document_title = title`, plus `document_id = law_id` and `chunk_index = article_id` (only so `compute_point_id` builds the deterministic `f"{law_id}:{article_id}"` uuid5 key — re-upserts overwrite, never duplicate).

- [ ] **Step 1: Write the failing test** — `tests/test_zalo_ingest.py`:

```python
"""
Unit tests for scripts/ingest_zalo_corpus.py — corpus flattening only
(embedding/upsert paths hit models/Qdrant and are exercised by running the
script, not by unit tests).

Run: .venv/bin/python -m pytest tests/test_zalo_ingest.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ingest_zalo_corpus import flatten_corpus

CORPUS = [
    {"law_id": "01/2020/qh14", "articles": [
        {"article_id": "2", "title": "Điều 2. Phạm vi", "text": "Nội dung."},
    ]},
]


def test_flatten_embed_text_is_title_plus_text():
    rows = flatten_corpus(CORPUS)
    assert len(rows) == 1
    assert rows[0]["embed_text"] == "Điều 2. Phạm vi\nNội dung."


def test_flatten_metadata_payload_contract():
    md = flatten_corpus(CORPUS)[0]["metadata"]
    # Benchmark ids ride on existing SearchResult fields (Task 5 reads
    # result.document_number / result.article):
    assert md["document_number"] == "01/2020/qh14"
    assert md["article"] == "2"
    assert md["document_title"] == "Điều 2. Phạm vi"
    # Deterministic point identity for compute_point_id:
    assert md["document_id"] == "01/2020/qh14"
    assert md["chunk_index"] == "2"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_zalo_ingest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.ingest_zalo_corpus'`

- [ ] **Step 3: Implement `scripts/ingest_zalo_corpus.py`**

```python
"""
Embed + index the Zalo 2021 legal corpus into a LOCAL Qdrant collection
(M3.1). Two modes so the GPU-heavy part can run on Kaggle while the local
docker Qdrant stays reachable only from this machine:

1) Embed to parquet shards (run on Kaggle GPU, or locally on MPS overnight):
   .venv/bin/python scripts/ingest_zalo_corpus.py embed \
       --corpus data/benchmarks/zalo/legal_corpus.json \
       --out-dir data/benchmarks/zalo/vectors \
       --device cuda --batch-size 128 --shard-size 2000
   Resume-safe: existing shard files are skipped, so rerunning after a
   session drop only embeds missing shards. On Kaggle: clone repo, `pip
   install -q -r requirements-ingest.txt`, attach legal_corpus.json as a
   dataset input, download the out-dir as output afterwards (same pattern
   as docs/notes/kaggle-ingest-guide.md).

2) Upsert shards into local Qdrant (run locally, docker compose up qdrant):
   .venv/bin/python scripts/ingest_zalo_corpus.py upsert \
       --shards "data/benchmarks/zalo/vectors/vectors_*.parquet" \
       --collection zalo_legal_2021 --qdrant-url http://localhost:6333

Production Qdrant Cloud is never touched: --qdrant-url is explicit and
defaults to localhost.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_zalo import load_zalo_corpus

EXPECTED_POINTS = 61425


def flatten_corpus(corpus):
    """One row per article. embed_text = title + newline + text (the title
    carries the article heading — cheap context, analogous to the contextual
    prefix used in production ingest). Metadata per the Task 4 payload
    contract (see plan)."""
    rows = []
    for law in corpus:
        for art in law["articles"]:
            rows.append({
                "embed_text": f"{art['title']}\n{art['text']}",
                "metadata": {
                    "document_number": law["law_id"],
                    "article": art["article_id"],
                    "document_title": art["title"],
                    "document_id": law["law_id"],
                    "chunk_index": art["article_id"],
                },
            })
    return rows


def cmd_embed(args):
    import pandas as pd
    from src.data.embedder import BGEEmbedder

    rows = flatten_corpus(load_zalo_corpus(args.corpus))
    print(f"{len(rows)} articles to embed")
    os.makedirs(args.out_dir, exist_ok=True)
    embedder = BGEEmbedder(device=args.device)

    n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    for shard_idx in range(n_shards):
        shard_path = os.path.join(args.out_dir, f"vectors_{shard_idx:04d}.parquet")
        if os.path.exists(shard_path):
            print(f"shard {shard_idx:04d}: exists, skipping")
            continue
        shard = rows[shard_idx * args.shard_size:(shard_idx + 1) * args.shard_size]
        dense, lexical = embedder.encode_documents(
            [r["embed_text"] for r in shard], batch_size=args.batch_size)
        pd.DataFrame({
            "embed_text": [r["embed_text"] for r in shard],
            "metadata": [json.dumps(r["metadata"], ensure_ascii=False) for r in shard],
            "dense": [d.tolist() for d in dense],
            # lexical_weights keys pass through json (str keys) — qdrant-client
            # coerces numeric-string indices to int, same as the prod path
            # where ingest.py hands lexical_weights straight to upsert_chunks.
            "sparse": [json.dumps({str(k): float(v) for k, v in lw.items()})
                       for lw in lexical],
        }).to_parquet(shard_path, index=False)
        print(f"shard {shard_idx:04d}: wrote {len(shard)} rows")
    print("✅ embed done")


def cmd_upsert(args):
    import pandas as pd
    from src.db.qdrant_client import LegalVectorStore

    store = LegalVectorStore(url=args.qdrant_url, api_key="",
                             collection_name=args.collection)
    store.create_collection()  # no quantization: local docker, RAM is not scarce

    shard_paths = sorted(glob.glob(args.shards))
    if not shard_paths:
        sys.exit(f"No shards match {args.shards}")
    total = 0
    for path in shard_paths:
        df = pd.read_parquet(path)
        store.upsert_chunks(
            chunk_texts=df["embed_text"].tolist(),
            dense_vectors=[list(v) for v in df["dense"]],
            sparse_vectors=[json.loads(s) for s in df["sparse"]],
            metadata_list=[json.loads(m) for m in df["metadata"]],
        )
        total += len(df)
        print(f"{os.path.basename(path)}: upserted {len(df)} (running total {total})")

    info = store.get_collection_info()
    print(f"Collection '{args.collection}': {info['points_count']:,} points")
    if info["points_count"] != EXPECTED_POINTS:
        sys.exit(f"❌ expected {EXPECTED_POINTS:,} points, got {info['points_count']:,}")
    print("✅ point count matches corpus")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--corpus", default="data/benchmarks/zalo/legal_corpus.json")
    e.add_argument("--out-dir", default="data/benchmarks/zalo/vectors")
    e.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--shard-size", type=int, default=2000)
    e.set_defaults(func=cmd_embed)

    u = sub.add_parser("upsert")
    u.add_argument("--shards", default="data/benchmarks/zalo/vectors/vectors_*.parquet")
    u.add_argument("--collection", default="zalo_legal_2021")
    u.add_argument("--qdrant-url", default="http://localhost:6333")
    u.set_defaults(func=cmd_upsert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_zalo_ingest.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_zalo_corpus.py tests/test_zalo_ingest.py
git commit -m "feat(M3.1): Zalo corpus embed-to-shards + local Qdrant upsert"
```

- [ ] **Step 6: EXECUTION CHECKPOINT — embed + index for real** (human-in-the-loop; not a subagent step)

1. Probe local throughput first: `embed` with `--device mps --shard-size 500 --batch-size 16` and time the first shard. ≥ ~3 articles/s (≲6h total) → let it run locally in the background. Slower → run the same command on Kaggle T4 (`--device cuda --batch-size 128`), download shards.
2. `docker compose up -d qdrant`, then run the `upsert` command.
3. Gate: script prints `✅ point count matches corpus` (61,425).

---

### Task 5: Benchmark runner + 3-level ablation

**Files:**
- Create: `scripts/bench_external.py`

**Interfaces:**
- Consumes: `load_zalo_questions` (Task 1); `load_splits` (Task 2); `recall_at_k`, `mrr_at_k`, `ndcg_at_k` (Task 3); `LegalVectorStore.search_dense(query_vector, top_k)` / `.search_hybrid(dense_vector, sparse_vector, top_k)` returning `SearchResult` with `.document_number`/`.article` (payload contract, Task 4); `BGEEmbedder.encode(texts, return_dense=True, return_sparse=True)`; `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cpu").compute_score([[q, passage], ...], normalize=True)` (same model + device pinning as production, cf. `scripts/bench_reranker.py`).
- Produces: report JSON `data/eval_reports/<UTC-ts>_zalo_bench.json`.

- [ ] **Step 1: Implement `scripts/bench_external.py`** (no unit test — thin orchestration over already-tested parts; correctness is checked by the smoke run in Step 2)

```python
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
```

- [ ] **Step 2: Smoke run** (requires Task 4 checkpoint done)

Run: `.venv/bin/python scripts/bench_external.py --levels a,b --split dev --limit 20`
Expected: completes without error; recall@10 for both levels **> 0** (any 0.0 across 20 queries means the payload-contract plumbing is broken — debug before proceeding, check `_ids()` output against qrels).

- [ ] **Step 3: Full runs**

```bash
.venv/bin/python scripts/bench_external.py --levels a,b --split full
.venv/bin/python scripts/bench_external.py --levels a,b,c --split test
```
Expected: two reports in `data/eval_reports/`. Sanity band: dense recall@10 roughly 0.7–0.9 (BGE-M3-class zero-shot on Zalo); far outside → debug harness before believing numbers.

- [ ] **Step 4: Run suite + commit** (reports are data — commit them like existing eval reports)

```bash
make test
git add scripts/bench_external.py data/eval_reports/*_zalo_bench.json
git commit -m "feat(M3.1): external benchmark runner + Zalo ablation reports"
```

---

### Task 6: ALQAC — timeboxed acquisition attempt (≤2h)

**Files:**
- Possibly create: `data/benchmarks/alqac/` (gitignored by Task 1 rule)
- No code files unless data is obtained.

**Interfaces:**
- Consumes: nothing. Produces: either ALQAC data ready for the Task 4/5 pipeline, or a documented blocker for Task 7's writeup.

- [ ] **Step 1: Attempt acquisition (timebox 2h total)**

Check in order; stop at first success:
1. https://alqac.github.io/ — data access instructions for ALQAC 2021–2023 (Task 1 = document retrieval, same `(law_id, article_id)` qrel shape as Zalo; ALQAC 2021 was built on the Zalo 2021 corpus).
2. GitHub search: `ALQAC 2021 dataset`, `ALQAC 2022 dataset`, `ALQAC 2023 dataset` — organizers' and participants' repos sometimes carry the released trainsets.
3. If everything is registration-/form-gated with no immediate grant: **stop at the timebox.**

- [ ] **Step 2a (data obtained):** rerun the pipeline — adapt `load_zalo_*` loaders if the year's schema differs (field names have varied slightly between years; keep any adapter inside `scripts/download_zalo.py` as `load_alqac_*` functions with the same return shapes), then Task 4 embed/upsert into collection `alqac_<year>`, then Task 5 runner with `--collection alqac_<year>` (splits: ALQAC train sets are small — report full-set zero-shot only, no split file).

- [ ] **Step 2b (blocked):** write the attempted sources + blocking reason into a short note for Task 7's writeup (which sources checked, dates, what gate). This satisfies the spec's "ALQAC hoặc blocker có ghi chép".

---

### Task 7: Writeup + close 3.1

**Files:**
- Create: `docs/notes/external-benchmark-zalo-alqac.md`
- Modify: `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md` (§7 M3 row note)

**Interfaces:** consumes the report JSONs from Task 5 (and 6 if ALQAC ran).

- [ ] **Step 1: Write `docs/notes/external-benchmark-zalo-alqac.md`** with (all numbers from the actual report files — no estimates):
  - Ablation table: rows = levels A/B/C, columns = recall@1/5/10, MRR@10, nDCG@10, latency P50/P95, #queries, split.
  - Split definition (seed 20260715, 80/10/10, pointer to `splits.json`) and why it's frozen (M3.2 contract).
  - Comparison against published Zalo-2021 numbers — cite specific sources found at write time (competition writeups/papers, e.g. participant reports on GitHub; state clearly that leaderboard test-set numbers are NOT directly comparable since the private test qrels are unavailable — our numbers are zero-shot on the public train set, no training on it).
  - ALQAC section: results, or the Task 6 blocker note.
  - Limitations: article-level indexing ≠ production chunking — measured scope is the embedding+fusion+rerank stack; corpus is 2021-frozen law, not our 600-doc corpus.
  - How it was run (exact commands, hardware, runtime).

- [ ] **Step 2: Update plan doc** — append to §7 M3 row: `3.1 ✅ <date>: Zalo ablation (dense/hybrid/+rerank) recall@10 = <A>/<B>/<C>, report <ts>_zalo_bench.json, writeup docs/notes/external-benchmark-zalo-alqac.md; ALQAC <result-or-blocker>.`

- [ ] **Step 3: Verify close gate** (spec §3): ablation table has all 3 levels on Zalo; ALQAC result or documented blocker; `make test` green; production defaults untouched (`git diff` shows no src/rag, src/api changes).

- [ ] **Step 4: Commit**

```bash
git add docs/notes/external-benchmark-zalo-alqac.md docs/Plan_Giai_Doan_Cuoi_2026-07-10.md
git commit -m "docs(M3.1): Zalo/ALQAC external benchmark writeup + close 3.1"
```

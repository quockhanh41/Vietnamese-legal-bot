# M3.2 Fine-tune Bi-encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune BGE-M3 on the 2,556 frozen Zalo train pairs (+7 mined hard negatives each), decide win/lose against the M3.1 zero-shot baseline via a pre-committed margin rule, run the internal Tier-1 comparison only on a Zalo win, and write up the experiment either way.

**Architecture:** Data prep and hard-negative mining run locally against the existing `zalo_legal_2021` Qdrant collection (M3.1). Training uses the official FlagEmbedding unified recipe on Kaggle GPU (dense+sparse+self-distill → checkpoint stays `BGEM3FlagModel`-loadable). A dense-only in-memory proxy on the dev split picks one checkpoint on Kaggle; only that checkpoint comes home. The Zalo verdict re-embeds the benchmark corpus into `zalo_legal_2021_ftv1` and runs the existing `bench_external.py` harness with a new `--model` flag. The conditional internal comparison reruns the existing `scripts/ingest.py` (which already has `--model-path`) into `luat_law_ctx_v4_ftv1` and compares Tier-1 routed recall@5.

**Tech Stack:** FlagEmbedding (BGEM3FlagModel + `FlagEmbedding.finetune.embedder.encoder_only.m3`), Qdrant (local docker + Cloud), Kaggle GPU (T4/P100), pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-07-15-m3-2-finetune-biencoder-design.md` (approved 2026-07-15)

## Global Constraints

- **Frozen split contract**: `data/benchmarks/zalo/splits.json` seed 20260715, train 2,556 / dev 319 / test 321. Never regenerate. The **test split is queried ONCE per candidate model**.
- **Zalo win rule (level B, hybrid, test)**: recall@10 ≥ **0.885** (baseline 0.875 + 1.0pt) **AND** MRR@10 ≥ **0.669**. Anything else = lose.
- **Internal win rule (conditional)**: overall Tier-1 routed recall@5 ≥ **0.845** AND no sector drops **> 2pt** vs the v4 baseline report. Exact parity (≥ but not >) → flag for user decision, do not silently pass.
- **Dev-proxy early-stop rule**: if the best fine-tuned checkpoint's dense-proxy recall@10 on dev is below stock BGE-M3's by more than 0.5pt, stop — the experiment is answered (lose), go to the writeup task.
- **$0**: Kaggle free GPU only. No paid API. Production Qdrant Cloud is touched only by the additive `luat_law_ctx_v4_ftv1` collection (deleted after verdict). Production default collection and REFUSAL_THRESHOLD are never modified.
- **Seeds**: every sampling/training seed is **20260715**.
- **Data hygiene**: `train.jsonl` stays gitignored (regenerable, deterministic); only `train_stats.json` is committed. Model checkpoints never enter git (`models/` gitignored).
- **Names (verbatim)**: model dir `models/bge-m3-viet-legal-ftv1`, collections `zalo_legal_2021_ftv1` (local) and `luat_law_ctx_v4_ftv1` (cloud), writeup `docs/notes/finetune-biencoder-zalo.md`, guide `docs/notes/kaggle-finetune-guide.md`.
- **Verify** with `.venv/bin/python -m pytest tests/ -q` before claiming any task done (CLAUDE.md).
- Python for local scripts is `.venv/bin/python`; Kaggle uses its own `python`.

---

### Task 1: Pure data-prep functions (`src/eval/finetune_data.py`)

**Files:**
- Create: `src/eval/finetune_data.py`
- Test: `tests/test_finetune_data.py`

**Interfaces:**
- Consumes: nothing from this plan (corpus/question dict shapes match `scripts/download_zalo.py` loaders: corpus = `[{"law_id", "articles": [{"article_id","title","text"}]}]`, question = `{"question_id","question","relevant_articles":[{"law_id","article_id"}]}`).
- Produces (Task 2 relies on these exact names):
  - `NEGATIVES_PER_QUERY = 7`, `RANK_LO = 11`, `RANK_HI = 100`
  - `article_text(article: Dict) -> str`
  - `build_article_lookup(corpus: List[Dict]) -> Dict[Tuple[str,str], str]`
  - `sample_negatives(ranked, positives, seed, question_id, n=7, lo=11, hi=100) -> List[Tuple[str,str]]`
  - `assert_no_leakage(question_ids, train_ids) -> None` (raises `ValueError`)
  - `build_training_row(question, lookup, negative_keys) -> Dict` (`{"query", "pos", "neg"}`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_finetune_data.py`:

```python
"""
Unit tests for src/eval/finetune_data.py — M3.2 training-data contracts:
no dev/test leakage, positives never sampled as negatives, rank-window
sampling, per-question determinism, FlagEmbedding jsonl row schema.

Run: .venv/bin/python -m pytest tests/test_finetune_data.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.eval.finetune_data import (
    NEGATIVES_PER_QUERY,
    RANK_HI,
    RANK_LO,
    article_text,
    assert_no_leakage,
    build_article_lookup,
    build_training_row,
    sample_negatives,
)

CORPUS = [
    {"law_id": "01/2020/qh14", "articles": [
        {"article_id": "1", "title": "Điều 1. Phạm vi", "text": "Nội dung điều 1."},
        {"article_id": "2", "title": "Điều 2. Đối tượng", "text": "Nội dung điều 2."},
    ]},
    {"law_id": "02/2020/qh14", "articles": [
        {"article_id": "1", "title": "Điều 1. Khác", "text": "Nội dung."},
    ]},
]

# Ranked candidate list: ranks 1..100 → keys ("law", "1")..("law", "100")
RANKED = [("law", str(i)) for i in range(1, 101)]


def test_article_text_matches_index_format():
    # Must equal ingest_zalo_corpus.flatten_corpus's embed_text: title\ntext
    assert article_text(CORPUS[0]["articles"][0]) == "Điều 1. Phạm vi\nNội dung điều 1."


def test_build_article_lookup_keys():
    lookup = build_article_lookup(CORPUS)
    assert len(lookup) == 3
    assert lookup[("01/2020/qh14", "2")] == "Điều 2. Đối tượng\nNội dung điều 2."


def test_sample_negatives_respects_rank_window():
    negs = sample_negatives(RANKED, positives=set(), seed=1, question_id="q1")
    assert len(negs) == NEGATIVES_PER_QUERY
    ranks = [int(key[1]) for key in negs]
    assert all(RANK_LO <= r <= RANK_HI for r in ranks)


def test_sample_negatives_never_returns_a_positive():
    positives = {("law", "15"), ("law", "42")}
    for seed in range(50):
        negs = sample_negatives(RANKED, positives, seed=seed, question_id="q1")
        assert not (set(negs) & positives)


def test_sample_negatives_deterministic_per_question_id():
    a = sample_negatives(RANKED, set(), seed=20260715, question_id="q7")
    b = sample_negatives(RANKED, set(), seed=20260715, question_id="q7")
    c = sample_negatives(RANKED, set(), seed=20260715, question_id="q8")
    assert a == b
    assert a != c  # different question → different (seeded) sample


def test_sample_negatives_raises_when_window_too_small():
    with pytest.raises(ValueError, match="only"):
        sample_negatives(RANKED[:12], set(), seed=1, question_id="q1")


def test_assert_no_leakage():
    assert_no_leakage(["a", "b"], train_ids={"a", "b", "c"})  # no raise
    with pytest.raises(ValueError, match="outside the frozen train split"):
        assert_no_leakage(["a", "z"], train_ids={"a"})


def test_build_training_row_schema():
    lookup = build_article_lookup(CORPUS)
    question = {
        "question_id": "q1",
        "question": "Phạm vi điều chỉnh là gì?",
        "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "1"}],
    }
    row = build_training_row(
        question, lookup, [("01/2020/qh14", "2"), ("02/2020/qh14", "1")])
    assert row == {
        "query": "Phạm vi điều chỉnh là gì?",
        "pos": ["Điều 1. Phạm vi\nNội dung điều 1."],
        "neg": ["Điều 2. Đối tượng\nNội dung điều 2.", "Điều 1. Khác\nNội dung."],
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_finetune_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.finetune_data'`

- [ ] **Step 3: Write the implementation**

Create `src/eval/finetune_data.py`:

```python
"""
M3.2 fine-tune training-data helpers — pure functions, no I/O.
Spec: docs/superpowers/specs/2026-07-15-m3-2-finetune-biencoder-design.md §2.

scripts/prepare_finetune_data.py wires these to the local Qdrant collection
(hard-negative mining) and writes the FlagEmbedding jsonl. Kept pure here so
the leakage / sampling / formatting contracts are unit-testable without
Qdrant or an embedder.
"""
import random
from typing import Dict, List, Sequence, Set, Tuple

ArticleKey = Tuple[str, str]  # (law_id, article_id)

NEGATIVES_PER_QUERY = 7
RANK_LO = 11   # 1-indexed, inclusive — skip ranks 1-10 (false-negative hedge)
RANK_HI = 100  # inclusive


def article_text(article: Dict) -> str:
    """Exactly the indexed embed_text format (ingest_zalo_corpus.flatten_corpus):
    title + newline + text — training pairs and index share one distribution."""
    return f"{article['title']}\n{article['text']}"


def build_article_lookup(corpus: List[Dict]) -> Dict[ArticleKey, str]:
    """(law_id, article_id) → article_text for every article in the corpus."""
    return {
        (law["law_id"], art["article_id"]): article_text(art)
        for law in corpus
        for art in law["articles"]
    }


def sample_negatives(
    ranked: Sequence[ArticleKey],
    positives: Set[ArticleKey],
    seed: int,
    question_id: str,
    n: int = NEGATIVES_PER_QUERY,
    lo: int = RANK_LO,
    hi: int = RANK_HI,
) -> List[ArticleKey]:
    """Sample n hard negatives from the 1-indexed rank window [lo, hi] of a
    ranked candidate list, never a positive. Seeded per (seed, question_id)
    so output is deterministic regardless of processing order."""
    window = [key for key in ranked[lo - 1:hi] if key not in positives]
    if len(window) < n:
        raise ValueError(
            f"{question_id}: only {len(window)} candidates in rank window "
            f"[{lo},{hi}] after dropping positives — need {n}")
    return random.Random(f"{seed}:{question_id}").sample(window, n)


def assert_no_leakage(question_ids: Sequence[str], train_ids: Set[str]) -> None:
    """Spec §2 contract: every training question must belong to the frozen
    train split — dev/test may never leak into training data."""
    leaked = set(question_ids) - set(train_ids)
    if leaked:
        raise ValueError(
            f"{len(leaked)} question(s) outside the frozen train split, "
            f"e.g. {sorted(leaked)[:5]}")


def build_training_row(
    question: Dict,
    lookup: Dict[ArticleKey, str],
    negative_keys: Sequence[ArticleKey],
) -> Dict:
    """FlagEmbedding finetune row: {"query", "pos": [...], "neg": [...]}."""
    positive_keys = [(ra["law_id"], ra["article_id"])
                     for ra in question["relevant_articles"]]
    return {
        "query": question["question"],
        "pos": [lookup[key] for key in positive_keys],
        "neg": [lookup[key] for key in negative_keys],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_finetune_data.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/eval/finetune_data.py tests/test_finetune_data.py
git commit -m "feat(M3.2): pure data-prep functions for bi-encoder fine-tune (leakage guard, seeded hard-negative sampling)"
```

---

### Task 2: Mining script + generate `train.jsonl` (`scripts/prepare_finetune_data.py`)

**Files:**
- Create: `scripts/prepare_finetune_data.py`
- Modify: `.gitignore` (add stats-file exception after the `!data/benchmarks/zalo/splits.json` line)
- Create (generated, committed): `data/benchmarks/zalo/finetune/train_stats.json`
- Create (generated, gitignored): `data/benchmarks/zalo/finetune/train.jsonl`

**Interfaces:**
- Consumes: Task 1 functions (exact names in Task 1's Produces block); `scripts/download_zalo.py` (`CORPUS_FILE`, `QA_FILE`, `load_zalo_corpus`, `load_zalo_questions`); `src.eval.external_bench.load_splits`; `LegalVectorStore.search_hybrid(dense_vector, sparse_vector, top_k) -> List[SearchResult]` where `SearchResult.document_number` = law_id and `SearchResult.article` = article_id (M3.1 payload contract); `BGEEmbedder.encode(texts, batch_size, return_dense=True, return_sparse=True)` → dict with `dense_vecs` (np arrays) and `lexical_weights`.
- Produces: `data/benchmarks/zalo/finetune/train.jsonl` — 2,556 FlagEmbedding rows consumed by Kaggle training (Task 5/6); `train_stats.json` with sha256 for determinism checks.

**Prereq:** local Qdrant running with the M3.1 collection: `docker compose up -d qdrant`, collection `zalo_legal_2021` has 61,425 points.

- [ ] **Step 1: Write the script**

Create `scripts/prepare_finetune_data.py`:

```python
"""
Build the M3.2 fine-tune training set (spec
docs/superpowers/specs/2026-07-15-m3-2-finetune-biencoder-design.md §2):
FlagEmbedding-format jsonl rows {"query", "pos", "neg"} for the 2,556
frozen train-split questions, with 7 hard negatives/query mined from the
local zalo_legal_2021 collection (stock BGE-M3, hybrid top-100, sampled
from ranks 11-100, seed 20260715).

Prereq: docker compose up -d qdrant + M3.1 collection (61,425 points).
Usage:
    .venv/bin/python scripts/prepare_finetune_data.py --device mps
Output:
    data/benchmarks/zalo/finetune/train.jsonl      (gitignored, regenerable)
    data/benchmarks/zalo/finetune/train_stats.json (committed)
Rerun-safe: overwrites both outputs. Sampling is deterministic per
(seed, question_id); the sha256 in stats verifies end-to-end determinism
(caveat: Qdrant RRF rank ties could in principle reorder candidates
between runs — if two runs differ, diff the jsonl before assuming a bug).
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_zalo import (CORPUS_FILE, QA_FILE, load_zalo_corpus,
                                   load_zalo_questions)
from src.eval.external_bench import load_splits
from src.eval.finetune_data import (NEGATIVES_PER_QUERY, RANK_HI, RANK_LO,
                                    assert_no_leakage, build_article_lookup,
                                    build_training_row, sample_negatives)

OUT_DIR = "data/benchmarks/zalo/finetune"
TRAIN_JSONL = os.path.join(OUT_DIR, "train.jsonl")
STATS_FILE = os.path.join(OUT_DIR, "train_stats.json")
SEED = 20260715
MINE_TOP_K = 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="zalo_legal_2021")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    splits = load_splits()
    train_ids = set(splits["train"])
    questions = [q for q in load_zalo_questions(QA_FILE)
                 if q["question_id"] in train_ids]
    assert_no_leakage([q["question_id"] for q in questions], train_ids)
    print(f"{len(questions)} train-split questions")

    lookup = build_article_lookup(load_zalo_corpus(CORPUS_FILE))

    from src.data.embedder import BGEEmbedder
    from src.db.qdrant_client import LegalVectorStore
    embedder = BGEEmbedder(device=args.device)
    store = LegalVectorStore(url=args.qdrant_url, api_key="",
                             collection_name=args.collection)

    os.makedirs(OUT_DIR, exist_ok=True)
    texts = [q["question"] for q in questions]
    rows = []
    for start in range(0, len(texts), args.batch_size):
        batch = questions[start:start + args.batch_size]
        out = embedder.encode(texts[start:start + args.batch_size],
                              batch_size=args.batch_size,
                              return_dense=True, return_sparse=True)
        for i, q in enumerate(batch):
            ranked = [(r.document_number, r.article)
                      for r in store.search_hybrid(out["dense_vecs"][i].tolist(),
                                                   out["lexical_weights"][i],
                                                   top_k=MINE_TOP_K)]
            positives = {(ra["law_id"], ra["article_id"])
                         for ra in q["relevant_articles"]}
            negatives = sample_negatives(ranked, positives, seed=SEED,
                                         question_id=q["question_id"])
            rows.append(build_training_row(q, lookup, negatives))
        print(f"  mined {min(start + args.batch_size, len(texts))}/{len(texts)}")

    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(TRAIN_JSONL, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    stats = {
        "rows": len(rows),
        "negatives_per_query": NEGATIVES_PER_QUERY,
        "rank_window": [RANK_LO, RANK_HI],
        "mine_top_k": MINE_TOP_K,
        "seed": SEED,
        "collection": args.collection,
        "sha256": sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"✅ {TRAIN_JSONL}: {len(rows)} rows")
    print(f"✅ {STATS_FILE}: sha256={sha[:12]}…")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the gitignore exception**

In `.gitignore`, directly after the line `!data/benchmarks/zalo/splits.json`, add:

```
!data/benchmarks/zalo/finetune
data/benchmarks/zalo/finetune/*
!data/benchmarks/zalo/finetune/train_stats.json
```

Verify: `git check-ignore data/benchmarks/zalo/finetune/train.jsonl` prints the path (still ignored); `git check-ignore data/benchmarks/zalo/finetune/train_stats.json` prints nothing (exit 1, trackable).

- [ ] **Step 3: Run the full test suite (regression)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (349+ tests, no new failures).

- [ ] **Step 4: Generate the training data**

```bash
docker compose up -d qdrant
.venv/bin/python scripts/prepare_finetune_data.py --device mps
```

Expected output ends with `✅ data/benchmarks/zalo/finetune/train.jsonl: 2556 rows`. Budget: query embedding on MPS ~5–10 min; mining ~1 min.

- [ ] **Step 5: Verify the artifact**

```bash
wc -l data/benchmarks/zalo/finetune/train.jsonl        # expect 2556
.venv/bin/python - <<'EOF'
import json
rows = [json.loads(l) for l in open("data/benchmarks/zalo/finetune/train.jsonl", encoding="utf-8")]
assert len(rows) == 2556, len(rows)
assert all(set(r) == {"query", "pos", "neg"} for r in rows)
assert all(len(r["pos"]) >= 1 and len(r["neg"]) == 7 for r in rows)
assert all(p not in r["neg"] for r in rows for p in r["pos"])
print("✅ schema + counts + no positive-in-negatives OK")
EOF
```

Expected: `✅ schema + counts + no positive-in-negatives OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_finetune_data.py .gitignore data/benchmarks/zalo/finetune/train_stats.json
git commit -m "feat(M3.2): hard-negative mining script + frozen train.jsonl stats (2,556 rows, 7 negs @ ranks 11-100, seed 20260715)"
```

---

### Task 3: `--model` flag on the three harness scripts

**Files:**
- Modify: `scripts/ingest_zalo_corpus.py` (embed subcommand + extract `build_parser()`)
- Modify: `scripts/bench_external.py` (query embedding + report field + extract `build_parser()`)
- Modify: `scripts/eval.py` (click option, query embedding, config field)
- Test: `tests/test_model_flag.py`

**Interfaces:**
- Consumes: `BGEEmbedder(model_name: str, device: str)` — already accepts a local checkpoint dir as `model_name`.
- Produces: `scripts.ingest_zalo_corpus.build_parser() -> argparse.ArgumentParser` (subcommand `embed` gains `--model`, default `"BAAI/bge-m3"`); `scripts.bench_external.build_parser() -> argparse.ArgumentParser` (gains `--model`, default `"BAAI/bge-m3"`; report JSON gains a `"model"` key); `scripts/eval.py` click option `--model` (default `"BAAI/bge-m3"`), recorded in the report `config` dict. Tasks 6–8 rely on these flags.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_flag.py`:

```python
"""
Arg-plumbing tests for the M3.2 --model flag (spec §6: light, no model
loads). Parsers only — the flag's value reaches BGEEmbedder(model_name=...)
by construction in each script.

Run: .venv/bin/python -m pytest tests/test_model_flag.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_zalo_embed_model_flag_default_and_override():
    from scripts.ingest_zalo_corpus import build_parser
    args = build_parser().parse_args(["embed"])
    assert args.model == "BAAI/bge-m3"
    args = build_parser().parse_args(["embed", "--model", "models/bge-m3-viet-legal-ftv1"])
    assert args.model == "models/bge-m3-viet-legal-ftv1"


def test_bench_external_model_flag_default_and_override():
    from scripts.bench_external import build_parser
    args = build_parser().parse_args([])
    assert args.model == "BAAI/bge-m3"
    args = build_parser().parse_args(["--model", "models/bge-m3-viet-legal-ftv1"])
    assert args.model == "models/bge-m3-viet-legal-ftv1"


def test_eval_cli_has_model_option():
    from scripts.eval import main as eval_main
    params = {p.name: p for p in eval_main.params}
    assert "model" in params
    assert params["model"].default == "BAAI/bge-m3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model_flag.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_parser'` (first two) and `KeyError: 'model'` (third).

- [ ] **Step 3: Modify `scripts/ingest_zalo_corpus.py`**

In `cmd_embed`, change the embedder construction (currently `embedder = BGEEmbedder(device=args.device)`):

```python
    embedder = BGEEmbedder(model_name=args.model, device=args.device)
```

Rename `main()`'s parser-building portion into `build_parser()` and add the flag to the embed subparser:

```python
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--corpus", default="data/benchmarks/zalo/legal_corpus.json")
    e.add_argument("--out-dir", default="data/benchmarks/zalo/vectors")
    e.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--shard-size", type=int, default=2000)
    e.add_argument("--model", default="BAAI/bge-m3",
                   help="HF id or local checkpoint dir (M3.2 fine-tuned model)")
    e.set_defaults(func=cmd_embed)

    u = sub.add_parser("upsert")
    u.add_argument("--shards", default="data/benchmarks/zalo/vectors/vectors_*.parquet")
    u.add_argument("--collection", default="zalo_legal_2021")
    u.add_argument("--qdrant-url", default="http://localhost:6333")
    u.set_defaults(func=cmd_upsert)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)
```

- [ ] **Step 4: Modify `scripts/bench_external.py`**

Change `_embed_queries` to accept the model:

```python
def _embed_queries(questions, device, model="BAAI/bge-m3", batch_size=64):
    from src.data.embedder import BGEEmbedder
    embedder = BGEEmbedder(model_name=model, device=device)
```

(the rest of the function body is unchanged.)

Extract the parser from `main()` into `build_parser()` and add `--model`:

```python
def build_parser():
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
    ap.add_argument("--model", default="BAAI/bge-m3",
                    help="HF id or local checkpoint dir for QUERY embedding "
                         "(must match the model that embedded --collection)")
    return ap
```

In `main()`: `args = build_parser().parse_args()`, pass the model through — `dense, sparse = _embed_queries(questions, args.device, args.model)` — and add `"model": args.model,` to the `report` dict (right after `"collection": args.collection,`).

- [ ] **Step 5: Modify `scripts/eval.py`**

Add a click option directly under the existing `--collection` option (line ~544):

```python
@click.option("--model", default="BAAI/bge-m3",
              help="HF id hoặc local checkpoint dir cho query embedding (M3.2). "
                   "Phải khớp model đã embed collection. Không áp dụng cho --retry.")
```

Add `model: str` to `main()`'s signature (after `collection: str`). Change the embedder line `embedder = BGEEmbedder()` to:

```python
    embedder = BGEEmbedder(model_name=model)
```

Add `"model": model,` to the `config` dict (after `"collection": collection,`).

- [ ] **Step 6: Run tests to verify they pass, plus regression**

Run: `.venv/bin/python -m pytest tests/test_model_flag.py tests/test_zalo_ingest.py -q` then `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/ingest_zalo_corpus.py scripts/bench_external.py scripts/eval.py tests/test_model_flag.py
git commit -m "feat(M3.2): --model flag on zalo embed / bench_external / eval.py for fine-tuned checkpoints"
```

---

### Task 4: Dense-proxy checkpoint selector

**Files:**
- Create: `src/eval/dense_proxy.py`
- Create: `scripts/select_checkpoint_dev.py`
- Test: `tests/test_dense_proxy.py`

**Interfaces:**
- Consumes: `scripts.ingest_zalo_corpus.flatten_corpus(corpus) -> List[{"embed_text", "metadata"}]`; `load_splits()["dev"]`; question/corpus loaders from `scripts/download_zalo.py`.
- Produces: `dense_recall_at_10(query_vecs, corpus_vecs, corpus_keys, relevant_per_query) -> float` — used by `scripts/select_checkpoint_dev.py`, which Task 6 runs on Kaggle.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dense_proxy.py`:

```python
"""
Unit tests for src/eval/dense_proxy.py — the M3.2 dev-split checkpoint
selection proxy (dense-only recall@10, in-memory numpy).

Run: .venv/bin/python -m pytest tests/test_dense_proxy.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.dense_proxy import dense_recall_at_10


def _norm(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_relevant_outside_top10_is_a_miss():
    # 11 corpus vectors hug e0; 1 ("target") sits at e1.
    # Query 1 = e0 → target ranks 12th → miss. Query 2 = e1 → target 1st → hit.
    near = np.stack([np.array([1.0, eps]) for eps in np.linspace(0.01, 0.11, 11)])
    corpus = _norm(np.vstack([near, np.array([[0.0, 1.0]])]).astype(np.float32))
    keys = [f"n{i}" for i in range(11)] + ["target"]
    queries = _norm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    score = dense_recall_at_10(queries, corpus, keys,
                               [{"target"}, {"target"}])
    assert score == 0.5


def test_small_corpus_caps_k():
    corpus = _norm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    queries = _norm(np.array([[1.0, 0.1]], dtype=np.float32))
    assert dense_recall_at_10(queries, corpus, ["a", "b"], [{"b"}]) == 1.0
    assert dense_recall_at_10(queries, corpus, ["a", "b"], [{"zz"}]) == 0.0


def test_multiple_relevant_any_hit_counts():
    corpus = _norm(np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32))
    queries = _norm(np.array([[1.0, 0.0]], dtype=np.float32))
    assert dense_recall_at_10(queries, corpus, ["a", "b", "c"], [{"b", "a"}]) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dense_proxy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.dense_proxy'`

- [ ] **Step 3: Write the implementation**

Create `src/eval/dense_proxy.py`:

```python
"""
Dense-only recall@10 proxy for M3.2 checkpoint selection (spec §4.1).

Deliberately dense-only and in-memory: cheap enough to run on Kaggle right
after training, so only the winning checkpoint (~2.3GB) gets downloaded.
The real verdict metric is hybrid RRF on the frozen test split — this proxy
only picks WHICH checkpoint gets that one shot.
"""
import numpy as np


def dense_recall_at_10(query_vecs, corpus_vecs, corpus_keys, relevant_per_query):
    """
    query_vecs: (nq, d) float32, L2-normalized (BGE-M3 dense output already is)
    corpus_vecs: (nc, d) float32, L2-normalized
    corpus_keys: list of hashable keys aligned with corpus_vecs rows
    relevant_per_query: list (len nq) of sets of keys
    Returns the fraction of queries with ≥1 relevant key in the dense top-10.
    """
    assert len(corpus_keys) == corpus_vecs.shape[0]
    assert len(relevant_per_query) == query_vecs.shape[0]
    k = min(10, corpus_vecs.shape[0])
    sims = query_vecs @ corpus_vecs.T  # cosine == dot on normalized vectors
    # argpartition: unordered top-k — recall@k only needs membership, not order
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    hits = sum(
        1 for qi in range(query_vecs.shape[0])
        if {corpus_keys[i] for i in top[qi]} & relevant_per_query[qi]
    )
    return hits / query_vecs.shape[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dense_proxy.py -q`
Expected: `3 passed`

- [ ] **Step 5: Write the selector script**

Create `scripts/select_checkpoint_dev.py`:

```python
"""
M3.2 dev-split dense-proxy checkpoint selection (spec §4.1). Run on Kaggle
right after training (works locally too). For each checkpoint dir — plus
stock BAAI/bge-m3 with --include-stock — embeds the 61,425-article corpus
and the 319 dev queries dense-only, and reports proxy recall@10. Pick the
best checkpoint; only that one gets downloaded and gets the one test shot.

Early-stop rule (plan Global Constraints): best checkpoint proxy more than
0.5pt BELOW the stock row → the experiment is answered (lose) — skip to
the writeup task.

Usage (Kaggle):
    python scripts/select_checkpoint_dev.py \
        --checkpoints "/kaggle/working/bge-m3-viet-legal-ftv1/checkpoint-*" \
        --include-stock --device cuda --batch-size 128
"""
import argparse
import gc
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_zalo import (CORPUS_FILE, QA_FILE, load_zalo_corpus,
                                   load_zalo_questions)
from scripts.ingest_zalo_corpus import flatten_corpus
from src.eval.dense_proxy import dense_recall_at_10
from src.eval.external_bench import load_splits

# Article-level passages: 1024 tokens covers typical Điều; identical for every
# candidate so the comparison is fair (proxy consistency > absolute fidelity).
CORPUS_MAX_LEN = 1024
QUERY_MAX_LEN = 128


def encode_dense(model_path, corpus_texts, query_texts, device, batch_size):
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(model_path, use_fp16=(device != "cpu"), devices=device)
    c = np.asarray(model.encode(corpus_texts, batch_size=batch_size,
                                max_length=CORPUS_MAX_LEN, return_dense=True,
                                return_sparse=False)["dense_vecs"],
                   dtype=np.float32)
    q = np.asarray(model.encode(query_texts, batch_size=batch_size,
                                max_length=QUERY_MAX_LEN, return_dense=True,
                                return_sparse=False)["dense_vecs"],
                   dtype=np.float32)
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return c, q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                    help='glob of checkpoint dirs, e.g. "out/checkpoint-*"')
    ap.add_argument("--include-stock", action="store_true",
                    help="also score stock BAAI/bge-m3 as the reference row")
    ap.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--corpus", default=CORPUS_FILE)
    ap.add_argument("--qa", default=QA_FILE)
    args = ap.parse_args()

    rows = flatten_corpus(load_zalo_corpus(args.corpus))
    corpus_texts = [r["embed_text"] for r in rows]
    corpus_keys = [(r["metadata"]["document_number"], r["metadata"]["article"])
                   for r in rows]
    dev_ids = set(load_splits()["dev"])
    dev_qs = [q for q in load_zalo_questions(args.qa)
              if q["question_id"] in dev_ids]
    query_texts = [q["question"] for q in dev_qs]
    relevant = [{(ra["law_id"], ra["article_id"]) for ra in q["relevant_articles"]}
                for q in dev_qs]
    print(f"{len(corpus_texts)} articles, {len(query_texts)} dev queries")

    candidates = sorted(glob.glob(args.checkpoints))
    if not candidates:
        sys.exit(f"No checkpoint dirs match {args.checkpoints}")
    if args.include_stock:
        candidates = ["BAAI/bge-m3"] + candidates

    results = {}
    for cand in candidates:
        print(f"── {cand} ──")
        c_vecs, q_vecs = encode_dense(cand, corpus_texts, query_texts,
                                      args.device, args.batch_size)
        results[cand] = dense_recall_at_10(q_vecs, c_vecs, corpus_keys, relevant)
        print(f"   dev dense recall@10 = {results[cand]:.4f}")

    print("\n=== dev dense-proxy recall@10 (selection metric) ===")
    for cand, score in sorted(results.items(), key=lambda t: -t[1]):
        print(f"  {score:.4f}  {cand}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test suite (regression — the script imports must not break collection)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/eval/dense_proxy.py scripts/select_checkpoint_dev.py tests/test_dense_proxy.py
git commit -m "feat(M3.2): dense-proxy dev checkpoint selector (in-memory recall@10, Kaggle-side)"
```

---

### Task 5: Kaggle fine-tune guide

**Files:**
- Create: `docs/notes/kaggle-finetune-guide.md`

**Interfaces:**
- Consumes: `train.jsonl` (Task 2), `scripts/select_checkpoint_dev.py` (Task 4), `--model` flags (Task 3).
- Produces: the runbook Task 6 executes. No code.

- [ ] **Step 1: Write the guide**

Create `docs/notes/kaggle-finetune-guide.md` with exactly these sections (reuse the repo/secret setup conventions from `docs/notes/kaggle-ingest-guide.md` and reference it rather than duplicating):

````markdown
# Kaggle guide — M3.2 fine-tune BGE-M3 (train → dev-proxy → re-embed)

> Spec: docs/superpowers/specs/2026-07-15-m3-2-finetune-biencoder-design.md
> Repo/secret setup (clone, pip, dataset attach): docs/notes/kaggle-ingest-guide.md
> Seed mọi thứ: 20260715. GPU: T4 x2 hoặc P100 (16GB), fp16.

## 0. Inputs

- Kaggle Dataset `zalo-finetune-m32` chứa `train.jsonl` (2,556 rows — output
  scripts/prepare_finetune_data.py, sha256 phải khớp
  data/benchmarks/zalo/finetune/train_stats.json).
- Repo clone + `pip install -q -r requirements-ingest.txt`.
- Fine-tune cần thêm: `pip install -q -U "FlagEmbedding[finetune]"`.

## 1. Verify recipe flags (BẮT BUỘC trước khi train)

Flag names dưới đây viết theo FlagEmbedding 1.3.x. Chạy:

    python -m FlagEmbedding.finetune.embedder.encoder_only.m3 --help

và đối chiếu từng flag trước khi chạy thật — nếu version cài trên Kaggle
khác tên flag, chỉnh theo --help (ghi lại diff vào writeup).

## 2. Train (3 epoch, save mỗi epoch)

    torchrun --nproc_per_node 1 \
      -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
      --model_name_or_path BAAI/bge-m3 \
      --train_data /kaggle/input/zalo-finetune-m32/train.jsonl \
      --cache_path /kaggle/working/cache \
      --output_dir /kaggle/working/bge-m3-viet-legal-ftv1 \
      --overwrite_output_dir \
      --learning_rate 1e-5 \
      --fp16 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 4 \
      --gradient_checkpointing \
      --train_group_size 8 \
      --query_max_len 128 \
      --passage_max_len 1024 \
      --pad_to_multiple_of 8 \
      --knowledge_distillation False \
      --unified_finetuning True \
      --use_self_distill True \
      --self_distill_start_step 0 \
      --temperature 0.02 \
      --sentence_pooling_method cls \
      --normalize_embeddings True \
      --dataloader_drop_last True \
      --warmup_ratio 0.1 \
      --logging_steps 20 \
      --save_strategy epoch \
      --seed 20260715

Kỳ vọng: ~639 step/epoch (2556/4), loss giảm rõ trong epoch 1.
Ngân sách: ~1.5–3h. (2 GPU T4: `--nproc_per_node 2` + thêm
`--negatives_cross_device` — optional, chỉ khi 1 GPU quá chậm.)

### Bậc thang VRAM khi OOM (theo thứ tự, spec §3)

1. `--per_device_train_batch_size 2` (rồi 1)
2. `--passage_max_len 512`
3. thêm `--gradient_accumulation_steps 4`
4. Vẫn OOM → dừng, quay về plan: LoRA fallback là kết quả có ghi chép,
   KHÔNG tự ý đổi recipe.

## 3. Verify checkpoints

    ls /kaggle/working/bge-m3-viet-legal-ftv1/checkpoint-*/

Mỗi checkpoint dir PHẢI có model weights (model.safetensors) + sparse_linear.pt
(+ colbert_linear.pt). Nếu checkpoint-N thiếu sparse_linear.pt → chỉ bản save
cuối ở output_dir root là candidate hợp lệ (unified head chỉ save đủ ở final
save của một số version) — ghi nhận và dùng bản đó.

## 4. Dev-proxy selection (chọn 1 checkpoint, kèm stock reference)

    python scripts/select_checkpoint_dev.py \
      --checkpoints "/kaggle/working/bge-m3-viet-legal-ftv1/checkpoint-*" \
      --include-stock --device cuda --batch-size 128

~10–15 phút/candidate (61,425 điều). Đọc bảng cuối:
- Best FT > stock − 0.5pt → tiếp tục.
- Best FT < stock − 0.5pt → DỪNG (early-stop rule) → writeup "lose".

## 5. Re-embed corpus Zalo bằng checkpoint thắng

    python scripts/ingest_zalo_corpus.py embed \
      --corpus data/benchmarks/zalo/legal_corpus.json \
      --out-dir /kaggle/working/vectors_ftv1 \
      --model /kaggle/working/bge-m3-viet-legal-ftv1/checkpoint-<N> \
      --device cuda --batch-size 128 --shard-size 2000

## 6. Download về máy

- Checkpoint thắng → `models/bge-m3-viet-legal-ftv1/` (local, gitignored)
- `vectors_ftv1/*.parquet` (31 shards) → `data/benchmarks/zalo/vectors_ftv1/`
````

- [ ] **Step 2: Commit**

```bash
git add docs/notes/kaggle-finetune-guide.md
git commit -m "docs(M3.2): Kaggle fine-tune runbook (train, VRAM ladder, dev-proxy, re-embed)"
```

---

### Task 6: EXECUTE — train on Kaggle, pick checkpoint, bring artifacts home

> Execution task (human-in-the-loop Kaggle session). No new code. Follow `docs/notes/kaggle-finetune-guide.md` exactly; record every number.

**Interfaces:**
- Consumes: guide (Task 5), `train.jsonl` (Task 2), selector (Task 4).
- Produces: `models/bge-m3-viet-legal-ftv1/` local checkpoint dir; `data/benchmarks/zalo/vectors_ftv1/vectors_*.parquet` (31 shards); dev-proxy table (numbers for the writeup).

- [ ] **Step 1: Add `models/` to `.gitignore`**

Append the line `models/` to `.gitignore`. Verify: `git check-ignore models/x` prints the path. Commit:

```bash
git add .gitignore
git commit -m "chore(M3.2): gitignore local model checkpoints"
```

- [ ] **Step 2: Package + upload training data**

Upload `data/benchmarks/zalo/finetune/train.jsonl` as Kaggle Dataset `zalo-finetune-m32`. Verify on Kaggle: `wc -l` = 2556 and `sha256sum` matches `train_stats.json`.

- [ ] **Step 3: Run guide §1–§2 (flag verify + training)**

Gate: training completes 3 epochs; loss clearly decreased during epoch 1 (log excerpt saved for the writeup). If OOM: walk guide §2 ladder in order; if the ladder is exhausted → STOP, report to user (LoRA fallback is a plan change requiring a user decision).

- [ ] **Step 4: Run guide §3–§4 (checkpoint verify + dev proxy)**

Record the full proxy table (stock + 3 checkpoints) verbatim. Apply the early-stop rule: best FT more than 0.5pt below stock → skip Tasks 7–8, go to Task 9 (writeup, verdict = lose at dev). Otherwise note the winning checkpoint id.

- [ ] **Step 5: Run guide §5–§6 (re-embed + download)**

Verify locally: `ls data/benchmarks/zalo/vectors_ftv1/vectors_*.parquet | wc -l` → 31; `ls models/bge-m3-viet-legal-ftv1/` contains model weights + `sparse_linear.pt`.

---

### Task 7: EXECUTE — Zalo verdict (dev sanity → ONE test run → margin rule)

> Execution task, local machine. Requires: docker Qdrant up, artifacts from Task 6.

**Interfaces:**
- Consumes: `--model` flags (Task 3), ftv1 shards + local checkpoint (Task 6), `zalo_legal_2021` baseline collection (M3.1).
- Produces: reports `data/eval_reports/<ts>_zalo_bench.json` (dev-stock if needed, dev-ft, test-ft, optional test-ft level C); the recorded verdict (win/lose) that gates Task 8.

- [ ] **Step 1: Upsert the fine-tuned vectors**

```bash
.venv/bin/python scripts/ingest_zalo_corpus.py upsert \
    --shards "data/benchmarks/zalo/vectors_ftv1/vectors_*.parquet" \
    --collection zalo_legal_2021_ftv1 --qdrant-url http://localhost:6333
```

Expected: `✅ point count matches corpus` (61,425).

- [ ] **Step 2: Stock dev baseline (only if missing)**

Check `ls data/eval_reports/*_zalo_bench.json` for an existing report with `"split": "dev"` and stock model. If none:

```bash
.venv/bin/python scripts/bench_external.py --levels a,b --split dev --device mps
```

Record hybrid (level B) dev recall@10 + MRR@10 as the stock dev reference.

- [ ] **Step 3: Fine-tuned dev sanity run**

```bash
.venv/bin/python scripts/bench_external.py --levels a,b --split dev --device mps \
    --collection zalo_legal_2021_ftv1 --model models/bge-m3-viet-legal-ftv1
```

Sanity gate (not the verdict): FT hybrid dev recall@10 should beat the stock dev reference — if it's clearly below despite the Task 6 proxy saying otherwise, STOP and debug (likely a wrong checkpoint dir or query/corpus model mismatch) before spending the test shot.

- [ ] **Step 4: THE one test run (levels A/B)**

```bash
.venv/bin/python scripts/bench_external.py --levels a,b --split test --device mps \
    --collection zalo_legal_2021_ftv1 --model models/bge-m3-viet-legal-ftv1
```

This is the single test-split query for this candidate (Global Constraints). Do not rerun to "double-check" a disappointing number.

- [ ] **Step 5: Apply the margin rule and record the verdict**

Win ⇔ level B on test: `recall@10 ≥ 0.885` AND `mrr@10 ≥ 0.669`. Write the verdict + both numbers + report filename into a note for Task 9. **Lose → skip Task 8, go to Task 9.**

- [ ] **Step 6 (win only): level C for the writeup**

```bash
.venv/bin/python scripts/bench_external.py --levels c --split test --device mps \
    --collection zalo_legal_2021_ftv1 --model models/bge-m3-viet-legal-ftv1
```

(~45 min CPU rerank; informative only, not gated.)

---

### Task 8: EXECUTE (win only) — internal Tier-1 comparison on `luat_law_ctx_v4_ftv1`

> Conditional execution task. Runs the M2 Kaggle-ingest pattern (`docs/notes/kaggle-ingest-guide.md`) with the fine-tuned model into a parallel CLOUD collection, then compares Tier-1.

**Interfaces:**
- Consumes: `scripts/ingest.py` (existing `--model-path --collection --quantized --resume`), winner checkpoint (Task 6), `scripts/backfill_is_in_effect.py`, `scripts/eval.py --model` (Task 3).
- Produces: Tier-1 sectors report on `luat_law_ctx_v4_ftv1`; internal win/lose/parity verdict for Task 9.

- [ ] **Step 1: Kaggle ingest with the fine-tuned model**

Per `docs/notes/kaggle-ingest-guide.md` (secrets → Qdrant Cloud), with the winner checkpoint attached as a Kaggle model/dataset input:

```bash
python scripts/ingest.py \
    --data-path data/raw/corpus_600docs.parquet \
    --collection luat_law_ctx_v4_ftv1 \
    --quantized --resume --device cuda --embed-batch-size 64 \
    --model-path /kaggle/input/<winner-checkpoint-mount>
```

Gate: final point count = **60,958** (same chunks, same deterministic point IDs as v4 — only vectors differ).

- [ ] **Step 2: Backfill `is_in_effect` (routed eval depends on it)**

```bash
.venv/bin/python scripts/backfill_is_in_effect.py --collection luat_law_ctx_v4_ftv1
```

Expected: same backfill count as v4 (statuses come from the shared MetadataStore; no LLM).

- [ ] **Step 3: Tier-1 routed eval with FT query embedding**

```bash
.venv/bin/python scripts/eval.py --set sectors \
    --collection luat_law_ctx_v4_ftv1 --model models/bge-m3-viet-legal-ftv1
```

- [ ] **Step 4: Compare against the v4 baseline and record the internal verdict**

Baseline = the M2 Phase E Tier-1 sectors report on `luat_law_ctx_v4` (overall routed recall@5 **0.845**; per-sector values from that report JSON in `data/eval_reports/` — see `.superpowers/sdd/progress.md` Phase E log for the filename). Internal win rule (Global Constraints): overall ≥ 0.845 AND no sector > 2pt below v4; exact parity → flag for user decision.

- [ ] **Step 5: Delete the parallel collection (spec §4.2)**

After the verdict is recorded (numbers + report filename saved for Task 9):

```bash
.venv/bin/python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from src.db.qdrant_client import LegalVectorStore
store = LegalVectorStore(collection_name="luat_law_ctx_v4_ftv1")
store.client.delete_collection("luat_law_ctx_v4_ftv1")
print("deleted luat_law_ctx_v4_ftv1")
EOF
```

Note: if the user wants to move straight into the adoption follow-up, they may choose to keep it — ask before deleting **only if** the internal verdict was a win; on a lose, delete without asking.

---

### Task 9: Writeup + status updates (ALWAYS runs — win or lose)

**Files:**
- Create: `docs/notes/finetune-biencoder-zalo.md`
- Modify: `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md` (§3 M3 table row 3.2 + §7 M3 status row)
- Modify: memory `vlb-project-state.md` (via memory tool, not git)

**Interfaces:**
- Consumes: every number recorded in Tasks 6–8 (proxy table, dev/test reports, internal report, verdicts).
- Produces: the M3.2 portfolio artifact.

- [ ] **Step 1: Write `docs/notes/finetune-biencoder-zalo.md`**

Same conventions as `docs/notes/external-benchmark-zalo-alqac.md` (Vietnamese, TL;DR first, report-file links). Required sections:

1. **TL;DR** — verdict in one line (win/lose/lose-at-dev), headline numbers vs baseline.
2. **Setup** — training data stats (from `train_stats.json`), recipe command (final flags actually used, including any VRAM-ladder steps taken), seed, GPU, wall-clock.
3. **Dev-proxy table** — stock + per-epoch checkpoints, selection decision.
4. **Zalo verdict** — dev sanity + test A/B table (FT vs M3.1 baseline rows), margin rule applied explicitly; level C table if run.
5. **Internal comparison** (if run) — overall + per-sector vs v4, verdict.
6. **Decision + next** — one of: (a) double win → adoption follow-up proposal (swap default, recalibrate REFUSAL_THRESHOLD, refusal gates — separate user decision); (b) Zalo win + internal lose → synthetic-query iteration proposal (spec §0.1); (c) lose → BGE-M3 zero-shot stays, experiment closed with numbers.
7. **Reproducibility** — exact commands end-to-end (prepare → train → select → verdict).

- [ ] **Step 2: Update the plan document**

In `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md`: append the M3.2 outcome to the §7 M3 row (mirror the 3.1 entry style: date, verdict, headline numbers, report/writeup links); update the 3.2 row of the §3/M3 table if the outcome closes it (e.g. "✅ thua, giữ BGE-M3 — writeup" or "✅ thắng — chờ adoption follow-up").

- [ ] **Step 3: Update project memory**

Update `vlb-project-state.md` memory: M3.2 done, verdict, where the numbers live.

- [ ] **Step 4: Commit**

```bash
git add docs/notes/finetune-biencoder-zalo.md docs/Plan_Giai_Doan_Cuoi_2026-07-10.md data/eval_reports/
git commit -m "docs(M3.2): fine-tune bi-encoder writeup + milestone status (verdict: <win|lose>)"
```

---

## Self-review notes (run after drafting — resolved inline)

- Spec §2 (data pipeline) → Tasks 1–2. Spec §3 (training + VRAM ladder) → Tasks 5–6. Spec §4.1 (proxy + `--model` + verdict) → Tasks 3, 4, 6, 7. Spec §4.2 (internal, delete collection) → Task 8. Spec §5 (artifacts) → Tasks 2, 5, 9. Spec §6 (tests) → Tasks 1, 3, 4. Spec §7 risks: VRAM → guide ladder; overfit → 3-epoch dev selection; false negatives → rank window; proxy mispick → Task 7 Step 3 sanity gate; win-Zalo-lose-internal → Task 9 decision branch (b); session drop → per-epoch saves + resume-safe shards.
- Type consistency: `build_parser` names match between Task 3 code and tests; `(document_number, article)` key order matches `SearchResult` usage in `bench_external._ids` and Task 2 mining; `models/bge-m3-viet-legal-ftv1` spelled identically in Tasks 3, 6, 7, 8.
- Known judgment call (documented, not a placeholder): FlagEmbedding recipe flag names are written per 1.3.x and MUST be re-verified via `--help` on Kaggle (guide §1) — flag drift is an expected, handled contingency, as is the checkpoint `sparse_linear.pt` presence check (guide §3).

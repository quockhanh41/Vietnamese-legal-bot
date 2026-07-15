# AITeamVN off-the-shelf embedding + reranker vs M2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure a full-swap stack — `AITeamVN/Vietnamese_Embedding` dense (mixed-hybrid with BGE-M3 sparse) + `AITeamVN/Vietnamese_Reranker` — on the 600-doc production corpus and compare against M2's own re-measured results.

**Architecture:** Add one `MixedHybridEmbedder` that exposes the exact `encode()`/`encode_queries()`/`encode_documents()` interface of `BGEEmbedder` (dense from a SentenceTransformer, sparse from BGE-M3), so it drops into `ingest.py`, `eval.py`, and the orchestrator with no logic changes. Thread `--embedder/--embed-model` through `ingest.py` (re-embed the corpus into a parallel collection) and `eval.py` Tier-1, and `--reranker` through `eval.py` Tier-2 into `AdaptiveRAGPipeline`. Then run Tier-1 (retrieval) + Tier-2 (answer/refusal) and write up the before/after.

**Tech Stack:** Python 3.13, FlagEmbedding (`BGEM3FlagModel`, `FlagReranker`), sentence-transformers 5.6.0, Qdrant, Click, pytest.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-15-aiteamvn-internal-comparison-design.md`. Every task serves it.
- Default behavior MUST stay byte-for-byte unchanged: `--embedder` defaults to `bge-m3`, `--reranker` defaults to `BAAI/bge-reranker-v2-m3`. Existing callers that pass nothing get the current behavior.
- Do NOT touch the production default collection (`luat_law_ctx_v4`), `REFUSAL_THRESHOLD`, or routing. The new collection is `luat_law_ctx_v4_aiteamvn` and is deleted after the verdict.
- Dense model id: `AITeamVN/Vietnamese_Embedding` (SentenceTransformer, 1024-dim, dot-product → embed with `normalize_embeddings=True`). Reranker id: `AITeamVN/Vietnamese_Reranker` (loads via `FlagReranker`, same architecture as `bge-reranker-v2-m3`).
- Sparse always comes from `BAAI/bge-m3` (AITeamVN embedding has no sparse head).
- Gate = M2's actual re-measured results (Tier-1 strict, run `20260714T164130Z`): dense recall@5 ≥ **0.8675**, hybrid ≥ **0.8675**, routed ≥ **0.8193**; no routed sector drops > 2pt. Tier-2: not worse than M2 v4 `--answers` (false_refusal ≤ 0.136, refusal recall ≥ 0.765 as documented floor). Do NOT use the non-reproducible 0.845.
- TDD: failing test first. Run `make test` green before each commit. Unit tests must NOT load real models (~4GB) — monkeypatch the model classes.

---

### Task 1: `MixedHybridEmbedder` + `build_embedder` factory

**Files:**
- Create: `src/data/mixed_embedder.py`
- Test: `tests/test_mixed_embedder.py`

**Interfaces:**
- Consumes: `sentence_transformers.SentenceTransformer`, `src.data.embedder.BGEEmbedder` (its `encode(texts, batch_size, return_dense, return_sparse) -> dict` with key `"lexical_weights"`).
- Produces:
  - `class MixedHybridEmbedder(dense_model="AITeamVN/Vietnamese_Embedding", sparse_model="BAAI/bge-m3", device="cpu")` with:
    - `encode(texts: List[str], batch_size=32, return_dense=True, return_sparse=True) -> Dict` — keys `"dense_vecs"` (np.ndarray (n,1024)) and/or `"lexical_weights"` (List[dict]).
    - `encode_queries(queries: List[str]) -> np.ndarray` (dense only, (n,1024)).
    - `encode_documents(documents: List[str], batch_size=32) -> Tuple[np.ndarray, List[dict]]` — `(dense, lexical_weights)`.
    - `vector_size -> int` property.
  - `build_embedder(embedder_type: str, embed_model: str = None, device: str = "cpu")` → `MixedHybridEmbedder` when `embedder_type=="mixed"`, `BGEEmbedder` when `"bge-m3"`, else `ValueError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mixed_embedder.py`:

```python
import numpy as np
import pytest

import src.data.mixed_embedder as mod


class _FakeST:
    """Stand-in for SentenceTransformer — no weights loaded."""
    def __init__(self, *a, **k):
        pass

    def get_sentence_embedding_dimension(self):
        return 1024

    def encode(self, texts, batch_size=32, normalize_embeddings=False):
        assert normalize_embeddings is True  # dot-product model must be normalized
        return np.ones((len(texts), 1024), dtype="float32")


class _FakeBGE:
    """Stand-in for BGEEmbedder — returns sparse only."""
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, batch_size=32, return_dense=True, return_sparse=True):
        assert return_dense is False  # mixed embedder only wants BGE's sparse
        return {"lexical_weights": [{1: 0.5} for _ in texts]}


@pytest.fixture
def emb(monkeypatch):
    monkeypatch.setattr(mod, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(mod, "BGEEmbedder", _FakeBGE)
    return mod.MixedHybridEmbedder(device="cpu")


def test_encode_returns_dense_and_sparse(emb):
    out = emb.encode(["a", "b"])
    assert out["dense_vecs"].shape == (2, 1024)
    assert out["lexical_weights"] == [{1: 0.5}, {1: 0.5}]


def test_encode_documents_returns_tuple(emb):
    dense, sparse = emb.encode_documents(["x"])
    assert dense.shape == (1, 1024)
    assert sparse == [{1: 0.5}]


def test_encode_queries_dense_only(emb):
    q = emb.encode_queries(["hello"])
    assert q.shape == (1, 1024)


def test_vector_size(emb):
    assert emb.vector_size == 1024


def test_build_embedder_mixed(monkeypatch):
    monkeypatch.setattr(mod, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(mod, "BGEEmbedder", _FakeBGE)
    e = mod.build_embedder("mixed", device="cpu")
    assert isinstance(e, mod.MixedHybridEmbedder)


def test_build_embedder_unknown_raises():
    with pytest.raises(ValueError):
        mod.build_embedder("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mixed_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.mixed_embedder'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/data/mixed_embedder.py`:

```python
"""Mixed-hybrid embedder: dense from a SentenceTransformer, sparse from BGE-M3.

AITeamVN/Vietnamese_Embedding is a dense-only SentenceTransformer fine-tune of
BGE-M3 (no sparse/ColBERT head). To keep the production hybrid + routed paths
working we take its dense vectors and BGE-M3's lexical (sparse) weights, and
expose the SAME interface as BGEEmbedder so eval.py, ingest.py, and the
orchestrator use it unchanged.
"""

import numpy as np
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
from src.data.embedder import BGEEmbedder


class MixedHybridEmbedder:
    def __init__(
        self,
        dense_model: str = "AITeamVN/Vietnamese_Embedding",
        sparse_model: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        self._dense = SentenceTransformer(dense_model, device=device)
        self._sparse = BGEEmbedder(model_name=sparse_model, device=device)
        self.dim = self._dense.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict:
        result: Dict = {}
        if return_dense:
            # normalize: AITeamVN uses dot-product similarity; normalized dense
            # makes dot == cosine, matching the v4 collection's distance metric.
            result["dense_vecs"] = self._dense.encode(
                texts, batch_size=batch_size, normalize_embeddings=True
            )
        if return_sparse:
            out = self._sparse.encode(
                texts, batch_size=batch_size, return_dense=False, return_sparse=True
            )
            result["lexical_weights"] = out["lexical_weights"]
        return result

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self._dense.encode(queries, normalize_embeddings=True)

    def encode_documents(
        self, documents: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        out = self.encode(
            documents, batch_size=batch_size, return_dense=True, return_sparse=True
        )
        return out["dense_vecs"], out["lexical_weights"]

    @property
    def vector_size(self) -> int:
        return self.dim


def build_embedder(embedder_type: str, embed_model: str = None, device: str = "cpu"):
    """Factory used by ingest.py and eval.py. Default 'bge-m3' = current behavior."""
    if embedder_type == "mixed":
        return MixedHybridEmbedder(
            dense_model=embed_model or "AITeamVN/Vietnamese_Embedding",
            sparse_model="BAAI/bge-m3",
            device=device,
        )
    if embedder_type == "bge-m3":
        return BGEEmbedder(model_name=embed_model or "BAAI/bge-m3", device=device)
    raise ValueError(f"unknown embedder_type: {embedder_type!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mixed_embedder.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data/mixed_embedder.py tests/test_mixed_embedder.py
git commit -m "feat(m3): MixedHybridEmbedder (AITeamVN dense + BGE-M3 sparse) + build_embedder factory"
```

---

### Task 2: `--embedder/--embed-model` in `scripts/ingest.py`

**Files:**
- Modify: `scripts/ingest.py` (`ingest()` signature ~95-105; embed sites at `:218` and `:249`; CLI options `:283-301`; `main` `:302-303`)
- Test: `tests/test_ingest_embedder_arg.py`

**Interfaces:**
- Consumes: `build_embedder` from Task 1.
- Produces: `ingest(..., embedder_type="bge-m3", embed_model=None)` and CLI flags `--embedder` / `--embed-model`. When `--embedder mixed`, both embed sites build via `build_embedder("mixed", embed_model, device)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_embedder_arg.py`:

```python
import src.data.mixed_embedder as mod
from scripts.ingest import _make_embedder


def test_make_embedder_bge(monkeypatch):
    calls = {}
    monkeypatch.setattr(mod, "BGEEmbedder", lambda **k: calls.update(k) or "BGE")
    out = _make_embedder("bge-m3", embed_model=None, model_path="/local/bge", device="cpu")
    assert out == "BGE"
    assert calls["model_name"] == "/local/bge"  # --model-path routed as the bge model


def test_make_embedder_mixed(monkeypatch):
    seen = {}

    class _Fake(mod.MixedHybridEmbedder):
        def __init__(self, **k):
            seen.update(k)

    monkeypatch.setattr(mod, "MixedHybridEmbedder", _Fake)
    out = _make_embedder("mixed", embed_model="AITeamVN/Vietnamese_Embedding",
                         model_path=None, device="cuda")
    assert isinstance(out, _Fake)
    assert seen["dense_model"] == "AITeamVN/Vietnamese_Embedding"
    assert seen["device"] == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ingest_embedder_arg.py -v`
Expected: FAIL — `ImportError: cannot import name '_make_embedder' from 'scripts.ingest'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/ingest.py`, add the import near the top (after line 28 `from src.data.embedder import BGEEmbedder`):

```python
from src.data.mixed_embedder import build_embedder
```

Add this helper above `ingest()` (its `model_path` routing keeps the existing `--model-path` offline-BGE flag working):

```python
def _make_embedder(embedder_type, embed_model, model_path, device):
    """Build the ingest embedder. For 'bge-m3', --model-path (local offline dir)
    still wins as the model source; for 'mixed', --embed-model is the dense model."""
    source = embed_model or model_path
    return build_embedder(embedder_type, embed_model=source, device=device)
```

Change `ingest()` signature to add the two params (after `model_path: str = None,`):

```python
    embedder_type: str = "bge-m3",
    embed_model: str = None,
```

Replace the embedder construction at BOTH embed sites — the current lines:

```python
                embedder = BGEEmbedder(model_name=model_path or "BAAI/bge-m3", device=device)
```
and
```python
            embedder = BGEEmbedder(model_name=model_path or "BAAI/bge-m3", device=device)
```

with (respectively, same indentation):

```python
                embedder = _make_embedder(embedder_type, embed_model, model_path, device)
```
```python
            embedder = _make_embedder(embedder_type, embed_model, model_path, device)
```

Add two Click options (after the existing `--model-path` option block, before `def main`):

```python
@click.option("--embedder", "embedder_type", type=click.Choice(["bge-m3", "mixed"]),
              default="bge-m3",
              help="bge-m3 (default, dense+sparse) hoặc mixed (AITeamVN dense + BGE-M3 sparse).")
@click.option("--embed-model", default=None,
              help="Model dense cho --embedder mixed (mặc định AITeamVN/Vietnamese_Embedding). "
                   "HF id hoặc đường dẫn local.")
```

Update `main(...)` signature and body to thread them through:

```python
def main(data_path, recreate, limit, embed_batch_size, device, collection, quantized,
         resume, model_path, embedder_type, embed_model):
    ingest(data_path, recreate, limit, embed_batch_size, device, collection, quantized,
           resume, model_path, embedder_type, embed_model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ingest_embedder_arg.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the ingest-curated test to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_ingest_curated_filter.py -v`
Expected: PASS (unchanged — default path untouched).

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest.py tests/test_ingest_embedder_arg.py
git commit -m "feat(m3): ingest.py --embedder/--embed-model (mixed AITeamVN dense + BGE-M3 sparse)"
```

---

### Task 3: `--embedder/--embed-model` in `scripts/eval.py` Tier-1 + `RETRIEVAL_K`

**Files:**
- Modify: `scripts/eval.py` (`RETRIEVAL_K` `:78`; `BGEEmbedder()` at `:509` and `:561`; CLI options `:536-548`; `main` `:549-585`; `run_retry` `:483-531`)
- Test: `tests/test_eval_retrieval_k.py`

**Interfaces:**
- Consumes: `build_embedder` from Task 1.
- Produces: `eval.py` CLI flags `--embedder {bge-m3,mixed}` + `--embed-model`; `RETRIEVAL_K = (1, 3, 5, 10)`. Tier-1 builds its embedder via `build_embedder`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_retrieval_k.py`:

```python
import scripts.eval as ev
from src.eval.metrics import recall_at_k


def test_retrieval_k_includes_1_and_3():
    assert ev.RETRIEVAL_K == (1, 3, 5, 10)
    assert ev.FETCH_K == 10  # still fetch top-10; @1/@3 are slices, no extra search


def test_recall_at_1_and_3_are_slices():
    # top-3 has the match at rank 3 -> recall@1 miss, recall@3 hit
    results = [
        {"document_id": 9, "article": "Điều 9", "chunk_text": ""},
        {"document_id": 8, "article": "Điều 8", "chunk_text": ""},
        {"document_id": 1, "article": "Điều 1", "chunk_text": ""},
    ]
    expected = [{"doc_id": 1, "article": "Điều 1"}]
    assert recall_at_k(results, expected, 1) == 0.0
    assert recall_at_k(results, expected, 3) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_retrieval_k.py -v`
Expected: FAIL on `test_retrieval_k_includes_1_and_3` — `assert (5, 10) == (1, 3, 5, 10)`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/eval.py`:

Change line 78:
```python
RETRIEVAL_K = (1, 3, 5, 10)
```

Add import after line 50 (`from src.data.embedder import BGEEmbedder`):
```python
from src.data.mixed_embedder import build_embedder
```

In `main(...)`, replace the Tier-1 embedder line (`:561`):
```python
    embedder = BGEEmbedder()
```
with:
```python
    embedder = build_embedder(embedder_type, embed_model, device="cpu")
```

Add two Click options after the `--collection` option (`:544`):
```python
@click.option("--embedder", "embedder_type", type=click.Choice(["bge-m3", "mixed"]),
              default="bge-m3", help="bge-m3 (default) hoặc mixed (AITeamVN dense + BGE-M3 sparse).")
@click.option("--embed-model", default=None,
              help="Model dense cho --embedder mixed (mặc định AITeamVN/Vietnamese_Embedding).")
```

Update `main(...)` signature to accept them (add `embedder_type: str, embed_model: Optional[str]` params). The `--retry` path (`run_retry`, line 509) rebuilds a `BGEEmbedder()` for Tier-2 retries of the ORIGINAL run — leave it as `BGEEmbedder()` (retry replays a prior bge-m3 tier2 report; out of scope here).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_retrieval_k.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Full suite green**

Run: `make test`
Expected: PASS (existing eval tests unaffected — default `--embedder bge-m3`).

- [ ] **Step 6: Commit**

```bash
git add scripts/eval.py tests/test_eval_retrieval_k.py
git commit -m "feat(m3): eval.py Tier-1 --embedder/--embed-model + recall@1/@3 (RETRIEVAL_K)"
```

---

### Task 4: Reranker model param — `orchestrator.py` + `eval.py --reranker`

**Files:**
- Modify: `src/rag/orchestrator.py` (`AdaptiveRAGPipeline.__init__` `:297-302`; `_get_reranker` `:310-324`)
- Modify: `scripts/eval.py` (CLI options; `AdaptiveRAGPipeline(...)` at `:575`)
- Test: `tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `AdaptiveRAGPipeline(..., reranker_model="BAAI/bge-reranker-v2-m3")` — `_get_reranker()` loads `self._reranker_model`. `eval.py --reranker <id>` passes it through (default None → orchestrator default).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def test_reranker_model_is_configurable(monkeypatch):
    from src.rag.orchestrator import AdaptiveRAGPipeline

    captured = {}

    class _FakeReranker:
        def __init__(self, model, use_fp16=True, device="cpu"):
            captured["model"] = model

    import FlagEmbedding
    monkeypatch.setattr(FlagEmbedding, "FlagReranker", _FakeReranker)

    p = AdaptiveRAGPipeline(
        vector_store=None, embedder=None, llm_gateway=None,
        reranker_model="AITeamVN/Vietnamese_Reranker",
    )
    assert p._reranker_model == "AITeamVN/Vietnamese_Reranker"
    p._get_reranker()
    assert captured["model"] == "AITeamVN/Vietnamese_Reranker"


def test_reranker_model_defaults_to_bge(monkeypatch):
    from src.rag.orchestrator import AdaptiveRAGPipeline

    p = AdaptiveRAGPipeline(vector_store=None, embedder=None, llm_gateway=None)
    assert p._reranker_model == "BAAI/bge-reranker-v2-m3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator.py::test_reranker_model_is_configurable tests/test_orchestrator.py::test_reranker_model_defaults_to_bge -v`
Expected: FAIL — `AttributeError: 'AdaptiveRAGPipeline' object has no attribute '_reranker_model'` (and `__init__` rejects `reranker_model`).

- [ ] **Step 3: Write minimal implementation**

In `src/rag/orchestrator.py`, change `AdaptiveRAGPipeline.__init__` (`:297`):

```python
    def __init__(self, *args, reranker_model: str = "BAAI/bge-reranker-v2-m3", **kwargs):
        super().__init__(*args, **kwargs)
        self._reranker_model = reranker_model
        self._reranker = None  # Lazy-loaded FlagReranker
        self._confidence_scorer = None  # Lazy-loaded
        self._verifier = None  # Lazy-loaded
        self._citation_lookup = None  # Lazy-loaded CitationLookup
```

Change `_get_reranker` (`:315`) to use `self._reranker_model`:

```python
                self._reranker = FlagReranker(
                    self._reranker_model,
                    use_fp16=True,
                    device='cpu',
                )
                print(f"✅ Reranker loaded: {self._reranker_model}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orchestrator.py::test_reranker_model_is_configurable tests/test_orchestrator.py::test_reranker_model_defaults_to_bge -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Wire `--reranker` into eval.py Tier-2**

Add a Click option (next to the Task 3 options):
```python
@click.option("--reranker", "reranker_model", default=None,
              help="Reranker model cho Tier-2 (mặc định BAAI/bge-reranker-v2-m3). "
                   "Ví dụ: AITeamVN/Vietnamese_Reranker.")
```

Add `reranker_model: Optional[str]` to `main(...)`. Where Tier-2 builds the pipeline (`:575`):
```python
        pipeline = AdaptiveRAGPipeline(vector_store=vector_store, embedder=embedder, llm_gateway=llm)
```
change to (only pass the kwarg when set, so the default stays):
```python
        rr_kwargs = {"reranker_model": reranker_model} if reranker_model else {}
        pipeline = AdaptiveRAGPipeline(vector_store=vector_store, embedder=embedder,
                                       llm_gateway=llm, **rr_kwargs)
```

- [ ] **Step 6: Full suite green**

Run: `make test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rag/orchestrator.py scripts/eval.py tests/test_orchestrator.py
git commit -m "feat(m3): configurable reranker model (orchestrator + eval.py --reranker)"
```

---

### Task 5: Runbook + writeup doc

**Files:**
- Create: `docs/notes/aiteamvn-internal-comparison.md`

This task is the execution guide + results skeleton. The runs themselves need GPU (embed) and Groq budget (Tier-2) — the user executes them; this doc records exact commands and holds the before/after tables.

- [ ] **Step 1: Write the runbook + writeup skeleton**

Create `docs/notes/aiteamvn-internal-comparison.md`:

````markdown
# AITeamVN off-the-shelf embedding + reranker vs M2 (600-doc)

Spec: `docs/superpowers/specs/2026-07-15-aiteamvn-internal-comparison-design.md`.
Feeds the M3.2 decision (is a self fine-tune worth it if an off-the-shelf VN
bge-m3 already wins?). Win or lose, this note gets the tables.

## Config
- Dense: `AITeamVN/Vietnamese_Embedding` (SentenceTransformer, 1024-dim, normalized).
- Sparse: `BAAI/bge-m3` (via `MixedHybridEmbedder`).
- Reranker: `AITeamVN/Vietnamese_Reranker`.
- Corpus: `data/raw/corpus_600docs.parquet` → collection `luat_law_ctx_v4_aiteamvn` (config = v4).
- Eval set: `data/eval/eval_set_sectors_v2.json` (83 q).

## Runbook

### 1. Re-embed the 600-doc corpus (GPU: Kaggle per docs/notes/kaggle-ingest-guide.md; or local)
```bash
.venv/bin/python scripts/ingest.py \
  --data-path data/raw/corpus_600docs.parquet \
  --collection luat_law_ctx_v4_aiteamvn \
  --embedder mixed --embed-model AITeamVN/Vietnamese_Embedding \
  --quantized --recreate --device cuda
```
Expect ~60,958 points. Enrichment/metadata_store/graph are collection-agnostic — NOT re-run.

### 2. Tier-1 on the new collection (no LLM)
```bash
.venv/bin/python scripts/eval.py --set sectors \
  --collection luat_law_ctx_v4_aiteamvn --embedder mixed
```

### 3. Re-measure M2 baseline at @1/@3 (no LLM — old report only had @5/@10)
```bash
.venv/bin/python scripts/eval.py --set sectors --collection luat_law_ctx_v4
```

### 4. Tier-2 answer/refusal, full swap (LLM — Groq budget)
```bash
.venv/bin/python scripts/eval.py --set sectors --answers \
  --collection luat_law_ctx_v4_aiteamvn --embedder mixed \
  --reranker AITeamVN/Vietnamese_Reranker --sleep 3
```
If Groq TPD dies mid-run: `scripts/eval.py --retry <report.json>`.

### 5. Delete the parallel collection after the verdict (free RAM).

## Results — Tier-1 retrieval (strict recall, sectors, n=83)

| mode | model | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| dense  | M2 v4 (bge-m3)   | _ | _ | 0.8675 | 0.9157 | _ |
| dense  | AITeamVN mixed   | _ | _ | _ | _ | _ |
| hybrid | M2 v4 (bge-m3)   | _ | _ | 0.8675 | 0.8976 | _ |
| hybrid | AITeamVN mixed   | _ | _ | _ | _ | _ |
| routed | M2 v4 (bge-m3)   | _ | _ | 0.8193 | 0.8916 | _ |
| routed | AITeamVN mixed   | _ | _ | _ | _ | _ |

Per-sector routed recall@5 (must not drop > 2pt vs v4): dan_su 0.844, dat_dai
0.967, doanh_nghiep 0.844, hinh_su 0.800, hngd 0.688, lao_dong 0.767.

## Results — Tier-2 answer/refusal (sectors)

| stack | refusal_acc | false_refusal | citation_acc |
|---|---|---|---|
| M2 v4              | _ | _ | _ |
| AITeamVN full-swap | _ | _ | _ |

## Verdict
- Tier-1 gate (per-mode ≥ M2): _
- Tier-2 gate (not worse than M2): _
- Decision for M3.2: _
````

- [ ] **Step 2: Commit the skeleton**

```bash
git add docs/notes/aiteamvn-internal-comparison.md
git commit -m "docs(m3): AITeamVN internal comparison runbook + results skeleton"
```

- [ ] **Step 3: (after runs) fill tables, write verdict, reference from M3.2 writeup, drop the parallel collection.**

---

## Notes for the executor
- Tasks 1-4 are code (CPU, no GPU, no LLM) — implement and land them first; `make test` is the gate.
- Task 5 step 3 (the measurements) needs GPU for re-embedding and Groq budget for Tier-2 — coordinate with the user; those numbers are the actual deliverable.
- If AITeamVN's dense underperforms on this internal set, that is a valid result ("stock BGE-M3 hybrid stays; our own fine-tune (M3.2) still justified") — write it up, don't tune to win.

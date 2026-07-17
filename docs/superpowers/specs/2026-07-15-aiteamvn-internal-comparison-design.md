# AITeamVN off-the-shelf embedding + reranker — đo trên 600-doc vs M2 — Design Spec

> **Ngày**: 2026-07-15 · **Milestone**: nhánh M3 — **de-risk M3.2** (thử off-the-shelf trước khi tự fine-tune)
> **Tiền đề**: M2 hoàn thành (default `luat_law_ctx_v4`, `REFUSAL_THRESHOLD` 0.700; baseline sectors Tier-1 thật, run `20260714T164130Z`, n=83, strict: **routed recall@5 = 0.8193**, **dense = hybrid = 0.8675**);
> M3.2 design spec [`2026-07-15-m3-2-finetune-biencoder-design.md`](./2026-07-15-m3-2-finetune-biencoder-design.md).
> Spec này đo một giả thuyết **rẻ hơn M3.2**: "đã có người fine-tune bge-m3 cho tiếng Việt (~300K triplet) —
> cặp model đó có thắng stock **trên corpus của mình** không?" Kết quả feed quyết định có nên tự fine-tune (M3.2).
> **Models**:
> - `AITeamVN/Vietnamese_Embedding` — dense-only **SentenceTransformer**, base `bge-m3`, 1024-dim, 2048 tok. **Không có** sparse/ColBERT head.
> - `AITeamVN/Vietnamese_Reranker` — base `bge-reranker-v2-m3`, load qua `FlagReranker` (drop-in).
> **Trạng thái**: design đã duyệt (user 2026-07-15, per-section approval) — spec này là bản ghi.

---

## 0. Quyết định khung (đã chốt với user 2026-07-15)

| # | Câu hỏi | Chốt |
|---|---|---|
| 1 | Đo ở đâu? | **Corpus production 600-doc** (collection song song), Tier-1 + Tier-2 của `scripts/eval.py`, so **mốc M2**. **KHÔNG** dùng cổng Zalo external. |
| 2 | Cấu hình nào? | **Full-swap, một cấu hình duy nhất**: AITeamVN dense (**mixed-hybrid** với bge-m3 sparse) + AITeamVN reranker. Bỏ staging/isolation (user chọn 2026-07-15). |
| 3 | Khoảng trống sparse? | AITeamVN embedding **dense-only** → giữ hybrid bằng **mixed**: dense = AITeamVN, sparse = bge-m3. Dense vẫn **1024-dim** ⇒ schema collection = v4, **không đổi**. |
| 4 | Reranker đo bằng gì? | **Tier-2 (LLM answer/refusal)**. Reranker **không đụng** Tier-1 recall@5 — nó nằm trong `orchestrator.py` (MEDIUM/COMPLEX: hybrid → rerank → generate) + là 1 tín hiệu confidence (weight 0.20), không nằm trong đường retrieval-only. |
| 5 | Metric? | **Thêm recall@1/@3** (`RETRIEVAL_K = (1, 3, 5, 10)`) — gần như **free** (top-10 đã fetch sẵn; recall@k chỉ là cắt list). Re-đo baseline v4 để có @1/@3. Bonus: so trực tiếp được với leaderboard AITeamVN (Acc@1/3/5/10). |

**Nguyên tắc kế thừa**: đo trước tối ưu sau — thắng hay thua đều ra writeup. **Không đụng** production default / threshold / refusal gate (adoption là follow-up riêng, quyết định với user).

## 1. Thang thí nghiệm (mỗi bậc là điểm dừng/đi)

```
1. MixedHybridEmbedder (local)      → verify: unit test interface (dense từ ST + sparse từ bge-m3), no real model load
2. Re-embed 600-doc (ingest.py)     → verify: collection luat_law_ctx_v4_aiteamvn đủ 60,958 điểm, config = v4
3. Tier-1 retrieval (rẻ, no LLM)    → verify: dense/hybrid/routed recall@{1,3,5,10} + per-sector, so M2
4. Re-đo baseline v4 @1/@3 (no LLM) → verify: chạy lại Tier-1 trên v4 gốc với K mới (report cũ chỉ có @5/@10)
5. Tier-2 answer/refusal (LLM)      → verify: refusal / false_refusal / citation / confidence, so M2 Tier-2
6. Writeup (luôn luôn)              → verify: bảng trước/sau đủ tái lập + decision trail
```

## 2. `MixedHybridEmbedder` — `src/data/mixed_embedder.py` (mới)

Gói **2 model** nhưng **phơi đúng interface `encode()` như `BGEEmbedder`**, nên drop-in vào `run_tier1`, `EnhancedRetriever`, và `orchestrator` mà **không sửa logic hybrid/routed**:

- `dense_vecs` ← `SentenceTransformer("AITeamVN/Vietnamese_Embedding").encode(..., normalize_embeddings=True)` → (n, 1024).
- `lexical_weights` ← `BGEEmbedder("BAAI/bge-m3")` với `return_sparse=True` (chỉ lấy sparse; bỏ dense của bge-m3).
- `encode(texts, return_dense, return_sparse)` trả `{"dense_vecs", "lexical_weights"}` — dense từ ST, sparse từ bge-m3.
- `encode_queries` / `encode_documents` tương ứng (query: dense ST + sparse bge-m3).
- **YAGNI**: chỉ implement những method mà `eval.py` + `ingest.py` + `orchestrator` thực sự gọi. **Không** đụng `BGEEmbedder`.

**Đánh đổi (ghi rõ)**: load đồng thời cả AITeamVN ST (~2GB) và bge-m3 (~2GB) lúc index/query → RAM & latency cao hơn. Chấp nhận được cho eval; không đụng path production mặc định.

## 3. Re-embed corpus + collection song song

- **Nguồn text**: contextual embed_text **không persist** — dựng lại **deterministic** bằng `src/data/contextualizer.build_contextual_text(meta)` từ `data/raw/corpus_600docs.parquet` (**đúng nguồn v4**, không LLM). Payload `chunk_text` giữ nguyên bản gốc (như v4).
- **Cơ chế**: mở rộng `scripts/ingest.py` thêm `--embedder {bge-m3,mixed}` + `--embed-model`; chạy trên `corpus_600docs.parquet` → collection **`luat_law_ctx_v4_aiteamvn`**. `ingest.py` tạo collection với **cùng config production = v4** (int8 quantization, on-disk, dense 1024 + sparse). **Chỉ đổi embedder** — mọi thứ khác y hệt v4 ⇒ apples-to-apples.
- **Enrichment/metadata_store/graph** (SQLite) **collection-agnostic** (keyed theo `doc_id`) → **KHÔNG** chạy lại.
- **Embed offline**: reuse pattern Kaggle GPU của M2 ([`docs/notes/kaggle-ingest-guide.md`](../../notes/kaggle-ingest-guide.md)); local MPS khả thi vì dense ST rẻ. ~60,958 điểm.
- Collection song song **xóa sau verdict** (RAM free-tier; tạm +~9.5% → ~19% tổng, an toàn <70%).

## 4. Evaluation

### 4.1. Tier-1 — retrieval (`scripts/eval.py`, không LLM)

- `RETRIEVAL_K = (1, 3, 5, 10)`; `FETCH_K = max = 10` (không đổi). recall@1/@3 = cắt top-10 đã fetch ⇒ **không thêm search/embedding/LLM**. `run_tier1` + `_aggregate_tier1` đã lặp `for k in RETRIEVAL_K` nên @1/@3 tự chảy vào JSON + bảng tổng/per-type/per-sector.
- `eval.py` thêm `--embedder {bge-m3,mixed}` + `--embed-model` → Tier-1 chạy `MixedHybridEmbedder` trên `luat_law_ctx_v4_aiteamvn`.
- **Set**: `sectors` (`data/eval/eval_set_sectors_v2.json`, **83 câu**). Modes: **dense / hybrid / routed**; per-type + per-sector + `leakage@5`.
- (Tùy chọn cosmetic) bảng in `by_type`/`by_sector` (`eval.py:419, 431`) đang hardcode cột @5/@10 — thêm @1/@3 ở đó nếu muốn; JSON có đủ bất kể.

### 4.2. Re-đo baseline v4 ở @1/@3

- Report M2 cũ chỉ lưu @5/@10 (`recall@1` = `None`) ⇒ chạy lại **Tier-1 trên `luat_law_ctx_v4` gốc** với `K = (1,3,5,10)` (không LLM, sectors set, vài phút). Dữ liệu retrieval không đổi — chỉ recompute metric ở K mới. Đây là baseline apples-to-apples cho bảng so sánh.
- **Lưu ý reconcile**: M3.2 §4.2 ghi "routed recall@5 ≥ 0.845" nhưng **không report v4 nào tái lập được 0.845**. Số thật (strict, `20260714T164130Z`, n=83): routed **0.8193**, dense = hybrid **0.8675**. Spec này dùng số **đo-lại-được** làm cổng; **0.845 trong M3.2 cần đính chính riêng** (ngoài phạm vi spec này).

### 4.3. Tier-2 — answer/refusal (`scripts/eval.py --answers`, **LLM**)

- Pipeline dùng `MixedHybridEmbedder` + **AITeamVN reranker** (qua param mới của orchestrator) trên `luat_law_ctx_v4_aiteamvn`.
- `orchestrator.py:315` hardcode `FlagReranker("BAAI/bge-reranker-v2-m3")` → đổi thành **param** (default giữ nguyên `bge-reranker-v2-m3`), truyền từ `AdaptiveRAGPipeline` ← `eval.py --reranker`.
- **Metrics**: `refusal_accuracy`, `false_refusal_rate`, `mean_citation_accuracy`, `confidence_distribution` — so M2 Tier-2.
- **Chi phí**: 83 câu × LLM (Groq free-tier, `--sleep` ~2s tránh rate-limit) — phần tốn nhất của cả spec. `--retry` sẵn có nếu Groq TPD chết giữa chừng (merge, không đốt lại token câu đã xong).

### 4.4. Verdict vs M2 (baseline-relative, **không** đụng threshold)

- **Gate = kết quả M2 thật** (tái đo §4.2 trong cùng đợt, run tham chiếu `20260714T164130Z`, strict). **Không** dùng số 0.845.
- **Tier-1** — **mỗi mode ≥ baseline M2 của chính nó**: dense recall@5 ≥ **0.8675**, hybrid ≥ **0.8675**, routed ≥ **0.8193** (production path). Báo cả 3 mode. **VÀ** không sector nào (routed) tụt > **2pt** so per-sector v4 (routed@5: `dan_su` 0.844, `dat_dai` 0.967, `doanh_nghiep` 0.844, `hinh_su` 0.800, `hngd` 0.688, `lao_dong` 0.767). recall@1/@3 là thông tin bổ sung, không phải cổng.
- **Tier-2** — **không tệ hơn M2 Tier-2 (v4)**: `false_refusal_rate` không tăng, refusal recall không giảm, `mean_citation_accuracy` không giảm. Số M2 lấy từ report v4 `--answers` (đo lại nếu chưa có v4 Tier-2). Sàn tài liệu M3 (baseline-relative): false_refusal ≤ **0.136**, refusal recall ≥ **0.765**.
- **Full-swap** ⇒ verdict là trên **stack gộp** (embedding + reranker cùng lúc — chủ đích). Nếu regress: một run **isolation** phụ (đổi từng model) để quy trách — ngoài phạm vi trừ khi cần.

## 5. Artefact + phạm vi

| Thứ | Chi tiết |
|---|---|
| Writeup | `docs/notes/aiteamvn-internal-comparison.md` — **thắng/thua đều viết**: bảng Tier-1 (recall@1/3/5/10 + per-sector, dense/hybrid/routed) + Tier-2 trước/sau, config, decision trail; ref từ M3.2 writeup |
| Reports | `data/eval_reports/<ts>_eval.json` (convention sẵn, đã stamp `collection` → phân biệt run) |
| Code | `src/data/mixed_embedder.py` + tests; `--embedder/--embed-model` cho `eval.py` + `ingest.py`; `--reranker` cho `eval.py`; reranker model thành param của `orchestrator` |
| Collection | `luat_law_ctx_v4_aiteamvn` — **xóa sau verdict** |

**Ngoài phạm vi (chốt)**: swap production default; recalibrate `REFUSAL_THRESHOLD`; refusal/OOS re-check (adoption follow-up); cổng Zalo external; `Vietnamese_Embedding_v2` (tied với v1 — optional add-on cùng loader); isolation run quy trách (chỉ mở nếu regress).

## 6. Testing (Tier-1, không GPU)

- **`MixedHybridEmbedder`**: `encode()` trả `dense_vecs` (1024-dim) từ nhánh ST + `lexical_weights` từ nhánh bge-m3; compose đúng 2 model (mock/no real load).
- **Arg-plumbing** `--embedder/--embed-model` (`eval.py` + `ingest.py`), `--reranker` (`eval.py`), reranker param (`orchestrator`) — test nhẹ, không load model thật; default không đổi hành vi.
- **`RETRIEVAL_K = (1,3,5,10)`**: `recall_at_k` slice đúng ở k=1,3 (reuse test `metrics`/`bench_metrics`).
- `make test` giữ xanh.

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Full-swap confound embedding + reranker | Verdict trên stack gộp là chủ đích; isolation run nếu regress |
| Load 2 model embedding (RAM/latency) | Chấp nhận cho eval; không đụng production path |
| Contextual text tái tạo lệch v4 | Dùng đúng `contextualizer` + `corpus_600docs.parquet` → deterministic, không LLM |
| AITeamVN dense-only phá hybrid | Mixed: giữ sparse bge-m3 → hybrid/routed nguyên vẹn |
| Groq TPD chết giữa Tier-2 | `--retry` merge sẵn có (`eval.py`) |
| RAM collection song song | +~9.5% (M3.2 §4.2), <70% an toàn |

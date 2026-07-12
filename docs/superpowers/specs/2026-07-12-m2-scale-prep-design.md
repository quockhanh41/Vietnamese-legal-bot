# M2 Scale-Prep — Design Spec

> **Ngày**: 2026-07-12 · **Milestone**: M2 (bậc 600 docs) — phần code/scale-prep
> **Tiền đề**: [`docs/Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3 (M2 tasks 2.3–2.6)
> **Trạng thái**: đã duyệt design (user 2026-07-12) — chờ implementation plan

---

## 0. Phạm vi phiên này (đã chốt với user)

M2 trong plan gốc có 7 việc (2.1–2.7). Phiên này làm **4 việc code/scale-prep**
mà có thể **implement + verify tại chỗ** hôm nay:

- **2.3** Citation lookup → server-side filter
- **2.4** Metadata store JSON → SQLite
- **2.5** `is_in_effect` → Qdrant payload filter
- **2.6** Reranker latency P95 harness

Các việc phụ thuộc dữ liệu/hạ tầng — **2.1** (capacity gate bằng số thật),
**2.2** (ingest 600 docs qua Kaggle GPU), **2.7** (eval + recalibrate trên 600) —
**để lại làm giai đoạn tiếp theo** trên hạ tầng của user (Qdrant Cloud + Kaggle/Colab).

**Lý do thứ tự này**: 4 việc code là "làm cho 600 docs query hiệu quả + đúng".
Chúng **phải land TRƯỚC** ingest 600 (2.5 ghi payload tại ingest-time), nên làm
trước sẽ khử rủi ro để ingest 600 chạy một lần sạch.

---

## 1. Phát hiện cập nhật giả định plan gốc

Plan gốc lập 2026-07-10, TRƯỚC khi M0/M1 land. Đối chiếu HEAD (`44626dc`):

| Giả định plan | Thực tế HEAD | Hệ quả |
|---|---|---|
| Payload index chưa có (blocker) | **Đã có** — `_create_payload_indexes()` (`qdrant_client.py:161`) tạo index cho `document_id`, `article_number`, `legal_type`, `issuance_date`, `is_in_effect`, `sector` | 2.3 chỉ còn là refactor client→server, tiền đề đã đủ |
| Quantization/on-disk chưa bật | **Đã implement** — `create_collection(quantized=True)` (int8 + fp32 on-disk) | Sẵn sàng cho collection 600 mới |
| `is_in_effect` post-filter client-side | Đúng — filter ở `orchestrator.py:212` qua annotation từ MetadataStore; **field CHƯA ghi vào chunk payload** (index có sẵn nhưng rỗng) | 2.5 = ghi field tại ingest + đổi filter sang server-side |
| Metadata store JSON, load hết RAM | Đúng — `MetadataStore` load `documents/graph/timelines/amendments.json` vào RAM lúc khởi tạo | 2.4 real |

**Môi trường xác nhận**: Qdrant Cloud reachable, `luat_law_ctx_v3` = 23,727 points
(corpus 100 docs). Không CUDA, có MPS. Source pool `legal_docs_merged.parquet`
= 518,255 rows có sẵn local. Corpus 100 docs được **tuyển tay** qua
`scripts/select_corpus.py` (600 sẽ cần chiến lược tuyển — thuộc giai đoạn ingest,
ngoài phạm vi spec này).

---

## 2. Nguyên tắc kế thừa (không thương lượng)

1. *Thà từ chối còn hơn trả lời sai* — 2 cổng từ chối giữ nguyên; không refactor
   nào được làm tụt refusal recall.
2. *Bảo toàn interface* — 2.3/2.4/2.5 đổi backend/đường đi, KHÔNG đổi hành vi quan
   sát được với caller (trừ hiệu năng/RAM). Test cũ phải xanh không sửa.
3. *Đo trước, tối ưu sau* — 2.6 chỉ ĐO, không đổi hành vi pipeline; nếu số xấu →
   khuyến nghị cap candidates, không tự bake vào.
4. Mỗi việc: TDD → chạy full suite → commit riêng.

---

## 3. Thiết kế từng việc

### 2.3 — Citation lookup → server-side filter

**Vấn đề** (chính module ghi `TODO(scale)`): `CitationLookup._load_collection_cache()`
scroll **toàn bộ** collection và cache **mọi chunk** client-side (`_chunks_by_doc`).
Ở 600 docs = 72–167K chunks nằm RAM.

**Hai concern tách biệt trong cache đó:**

1. **`lookup(parsed, cap)`** — với (doc_ids, article_number) cụ thể, lấy chunk
   khớp. → Đổi sang **server-side filtered scroll**:
   - Strict: `Filter(must=[document_id ∈ candidates, article_number == target])`
     (cả hai đã index). Lấy trực tiếp, không quét cache.
   - Lenient (chunk mở đầu bằng `Điều N.` nhưng `article` payload stale): chỉ
     chạy khi strict < cap — filtered scroll trên `document_id` của các candidate
     rồi lọc `_CHUNK_OPEN_WINDOW` như cũ. Bounded theo số doc candidate.
2. **Doc inventory** (`resolve_doc`, `corpus_summary`, `corpus_summary_compact`
   cần doc_id→title/number/sector) — nhỏ (600 entries). → thêm
   `LegalVectorStore.get_document_index()`: **một scroll doc-level projected**
   (`with_payload=[document_id, document_title, document_number, legal_type,
   sector]`, `with_vectors=False`), dedup theo `document_id`, **KHÔNG giữ
   chunk_text**. Memory O(docs) thay O(chunks). Cache một lần như hiện tại.

**Quyết định (a) đã chốt**: doc inventory dùng light projected scroll, **KHÔNG**
đọc từ SQLite store (2.4) — giữ hai việc decoupled. SQLite store hiện không có
`document_number`/`sector` nên không phải nguồn đủ; và tách unit giúp test độc lập.

**Interface đổi:**
- `qdrant_client.py`: thêm `get_document_index() -> Dict[int, Dict]` (doc-level
  projected scroll) và một helper filtered-scroll cho lookup (vd
  `scroll_article(doc_ids, article_number, cap)`), tái dùng payload index.
- `citation_lookup.py`: `_load_collection_cache()` → tách thành
  `_load_doc_index()` (gọi `get_document_index`) cho alias/number/sector; bỏ
  `_chunks_by_doc`. `lookup()` gọi filtered scroll thay vì quét cache.
- Public API của `CitationLookup` (`lookup`, `resolve_doc`, `corpus_summary`,
  `corpus_summary_compact`) **giữ nguyên chữ ký + ngữ nghĩa**.

**Verify**: unit test parse/resolve giữ nguyên; test mới cho filtered scroll
(ephemeral collection nhỏ); **end-to-end read-only trên live v3** — so kết quả
`lookup()` mới với đường cũ trên vài citation query (crim-*, đa lĩnh vực) phải
trùng point_ids.

**Rủi ro/lưu ý**: `article_number` index là KEYWORD (giữ hậu tố chữ "12a"), so
khớp string — không cast int (đã đúng ở `upsert_chunks`). Chunk có `article`
payload stale nhưng text mở đầu `Điều N` vẫn phải bắt được qua nhánh lenient.

---

### 2.4 — Metadata store JSON → SQLite

**Vấn đề**: `MetadataStore` load toàn bộ `documents/graph/timelines/amendments.json`
vào RAM lúc khởi tạo; `get_documents_by_target` quét O(N) mọi doc.

**Thiết kế**: một file `data/metadata/store.db` (SQLite, stdlib `sqlite3`).

Bảng:
- `documents(doc_id INTEGER PRIMARY KEY, title, ref_count, internal_refs,
  external_refs, effective_date, expiry_date, is_in_effect, is_amendment,
  amendment_count, graph_node_id, graph_in_degree, graph_out_degree,
  enriched_at, articles_referenced JSON, replaced_laws JSON, version_chain JSON,
  target_laws JSON, amended_articles JSON, errors JSON)` — list fields lưu JSON
  text.
- `timelines(doc_id INTEGER, effective_date, expiry_date, is_in_effect, blob JSON)`.
- `amendments(amending_doc_id INTEGER, target, blob JSON)`.
- `graph(id INTEGER PRIMARY KEY CHECK(id=1), blob JSON)` — single-row (nodes+edges);
  `get_article_neighbors` đọc edges từ blob (traversal vẫn NetworkX phía enricher).

Index: `documents(is_amendment)`, `timelines(doc_id)`, `amendments(amending_doc_id)`.
`get_effective_laws(at_date)` → `WHERE effective_date<=? AND (expiry_date IS NULL
OR expiry_date>?)`.

**Bảo toàn TOÀN BỘ public API** của `MetadataStore` (`save_document`,
`get_document`, `get_all_documents`, `get_documents_by_target`,
`get_articles_amended`, `save_graph`, `get_graph`, `get_article_neighbors`,
`save_timelines`, `get_timelines`, `get_timeline`, `get_effective_laws`,
`save_amendments`, `get_amendments`, `get_amendments_for_doc`, `build_index`,
`get_index`, `clear`). Caller (`enricher.py`, `retriever.py`, `orchestrator.py`)
**không đổi một dòng**.

**Migration**: hàm/CLI `import_json_to_sqlite(data_dir)` đọc `*.json` cũ → nạp
SQLite (giữ enrichment 100 docs hiện có; cũng tái tạo được bằng re-enrich).
Giữ khả năng export JSON để portable (optional).

**Hiệu năng**: `documents` vẫn cache dict trong RAM (600 entries rẻ) cho hot
lookup O(1); cái được là (i) không load JSON khổng lồ graph/timelines lúc init,
(ii) `get_documents_by_target`/`get_effective_laws` thành truy vấn có index thay
vì quét Python.

**Verify**: test round-trip từng bảng; test API parity (mọi method trả kết quả
= backend JSON cũ trên fixture); test migration importer; chạy full suite (caller
không đổi → phải xanh).

---

### 2.5 — `is_in_effect` → Qdrant payload filter

**Hai phần:**

1. **Ingest-time** — sau enrichment tính `is_in_effect` per doc, **backfill** vào
   payload chunk: `client.set_payload(payload={"is_in_effect": <bool>},
   points=Filter(document_id==doc_id))` cho mọi doc đã biết trạng thái. Idempotent,
   additive (chỉ thêm field). Wiring:
   - thêm bước vào pipeline ingest/enrich (sau khi enricher chạy xong);
   - + script standalone `scripts/backfill_is_in_effect.py` để chạy lại trên
     collection có sẵn (vd v3) khi cần — **không** tự động mutate v3 trong dev loop.
   - Chỉ ghi khi `is_in_effect is not None` (None = unknown, để trống → filter giữ).

2. **Query-time** — thay client-side annotation filter (`orchestrator.py:212`)
   bằng **server filter** truyền vào search:
   `must_not=[FieldCondition(key="is_in_effect", match=MatchValue(value=False))]`
   → giữ `True` **và** field vắng/None, chỉ loại `False`. **Đúng y ngữ nghĩa cũ**
   ("giữ unknown, chỉ loại known-superseded").
   - Annotation path (MetadataStore) **giữ lại** chỉ để dựng **amendment badge**
     (UI) trên các result sống sót — không còn dùng để LỌC.
   - `_build_filter` hiện chỉ hỗ trợ `must`+MatchValue; mở rộng để nhận `must_not`
     (hoặc thêm tham số riêng cho is_in_effect) để search_dense/hybrid nhận được.

**Quyết định (b) đã chốt**: chỉ boolean `is_in_effect` lên server-side. Point-in-time
`at_date` (`retriever.py`) **giữ client-side** phiên này — server-side date-range
cần `datetime` index (`effective_date` đang KEYWORD, không range được). Flag lại
như follow-up nhỏ, không lén thêm.

**Verify**: ephemeral collection — upsert vài chunk với is_in_effect True/False/absent,
test filter trả đúng tập (giữ True+absent, loại False); test backfill set_payload
đúng theo document_id; so ngữ nghĩa với đường annotation cũ trên fixture. Read-only
trên v3: v3 chưa có field → filter giữ tất cả (khớp "unknown kept").

**Rủi ro**: đảm bảo route lookup (citation) cũng đi qua filter nhất quán (hoặc chủ
ý bỏ qua — citation lookup là exact-match, thường muốn trả cả điều luật cũ nếu user
hỏi đích danh). → **Quyết**: filter is_in_effect chỉ áp cho route hybrid/vector
(giữ hành vi hiện tại: orchestrator lọc sau retrieval), KHÔNG áp cho citation
lookup exact-match. Ghi rõ trong code.

---

### 2.6 — Reranker latency P95 harness

**Mục đích**: đo, không đổi hành vi. Plan: nếu CPU quá chậm → cap candidates (top-10).

**Thiết kế**: `scripts/bench_reranker.py`:
- Nạp reranker hiện dùng (cùng model/đường như pipeline).
- Dựng candidate pool thật: retrieve trên tập query eval (hoặc sample chunk từ
  collection) → pool kích thước {10, 20, 50, 100}.
- Mỗi kích thước: chạy rerank M lần (warm-up rồi đo), báo **P50/P95/P99** latency
  + throughput (pairs/s) trên máy hiện tại (MPS/CPU).
- Xuất artifact: bảng markdown + JSON vào `data/eval_reports/reranker_bench_<ts>.{md,json}`.
- Ghi rõ device (mps/cpu) trong report; **re-runnable trên HF Spaces (M5)** cho số
  CPU thật.

**Không** đổi pipeline. Nếu P95 xấu → khuyến nghị cap candidates (config một dòng)
trong phần kết report, để user quyết ở M5 gate.

**Verify**: chạy harness ra được bảng số; smoke test script không lỗi; giá trị
hợp lý (P95 ≥ P50).

---

## 4. Cross-cutting

- **Branch**: `m2-scale-prep` (nhánh từ `main`), mỗi việc một commit; message theo
  convention repo (`feat(M2.x): ...`).
- **Workflow**: theo memory dự án — sub-agent implement → Fable/độc lập verify →
  user duyệt → commit. (Quyết định cụ thể dispatch ở bước executing-plans.)
- **Test**: full suite (116+ tests) phải xanh sau mỗi việc; test mới cho từng đường.
- **Không mutate live v3** trong dev loop; write path test trên ephemeral collection.
- **Cập nhật trạng thái**: sau khi 4 việc xong, ghi vào bảng §7 Plan gốc là "M2
  scale-prep xong; còn 2.1/2.2/2.7 (ingest+eval) ở giai đoạn sau".

## 5. Ngoài phạm vi (cố tình)

- 2.1 capacity gate, 2.2 ingest 600, 2.7 eval/recalibrate — giai đoạn ingest sau.
- Chọn/tuyển 600 docs — quyết định giai đoạn ingest.
- Server-side date-range (`at_date`) point-in-time — cần datetime index, follow-up.
- Backfill is_in_effect lên v3 production — có script sẵn, user chạy khi muốn, không
  tự động.

## 6. Definition of Done (spec này)

- [ ] 2.3: `lookup()` server-side filter; doc inventory qua projected scroll;
  RAM O(docs); e2e read-only trùng đường cũ trên live v3.
- [ ] 2.4: `MetadataStore` backend SQLite, API parity, importer JSON→SQLite,
  full suite xanh.
- [ ] 2.5: is_in_effect ghi tại ingest (+ script backfill); query filter server-side
  cho route hybrid/vector; ngữ nghĩa "keep unknown" giữ nguyên.
- [ ] 2.6: `bench_reranker.py` chạy ra bảng P50/P95/P99 theo pool size; report artifact.
- [ ] Mỗi việc commit riêng trên `m2-scale-prep`; trạng thái Plan gốc cập nhật.

# M2 Completion — 600 docs ingest — Design Spec

> **Ngày**: 2026-07-13 · **Milestone**: M2 (bậc 600 docs) — phần ingest/eval còn lại
> **Tiền đề**: [`docs/Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3 (M2 tasks 2.1/2.2/2.7)
> và [spec scale-prep](2026-07-12-m2-scale-prep-design.md) (2.3–2.6 đã land)
> **Trạng thái**: đã duyệt design (user 2026-07-13, "all ok") — chờ implementation plan

---

## 0. Phạm vi phiên này (đã chốt với user 2026-07-13)

M2 còn **3 việc** sau scale-prep: **2.1** capacity gate bằng số thật, **2.2** ingest
600 docs (Kaggle GPU), **2.7** eval mở rộng + recalibrate. Ba việc này **phụ thuộc
GPU + hạ tầng** nên milestone vốn dĩ human-in-the-loop.

**Ba quyết định khung đã chốt:**

| # | Câu hỏi | Chốt |
|---|---|---|
| 1 | Phiên này sản xuất gì? | **Chuẩn bị mọi thứ TỚI trước ingest.** Tôi làm 2.1 + tuyển corpus 600 + mở rộng eval + Kaggle guide. User chạy ingest GPU. Recalibrate (2.7) làm chung SAU ingest. |
| 2 | Hình dạng corpus 600? | **Đào sâu đúng 6 lĩnh vực** hiện có (hình sự/dân sự/lao động/doanh nghiệp/đất đai/HNGĐ). Lớp độ sâu = **Thông tư** (implementing circulars — thứ trả lời "phạt bao nhiêu / thủ tục / mẫu đơn"). KHÔNG thêm lĩnh vực mới → giữ nguyên OOS anchors, không phải redefine eval lần 3. |
| 3 | Cách tuyển ~500 docs? | **Bán tự động + cổng review của user.** Giữ 100 docs tuyển tay làm spine khoá cứng; sinh ~500 candidate tự động (theo legal_type + sector keyword + superseded-drop guard); xuất bảng review; user prune → freeze → parquet. |
| 4 | Cân bằng chunk giữa sector? | **Bounded ratio ≤ ~3×** (chống frequency bias — xem B.5). Budget theo chunk-count, không phải doc-count. Enrichment (Phase E) **mặc định chạy trên Kaggle**. |

**Chia việc (division of labor):**
- **Trong phiên (tôi)**: A capacity baseline · B tuyển corpus → cổng review · C validate trước ingest + mở rộng eval · cập nhật Kaggle guide.
- **User (Kaggle GPU)**: D chạy ingest → `luat_law_ctx_v4`.
- **Chung, sau ingest**: E enrichment · recalibrate · gates · flip default.

---

## 1. Môi trường xác nhận (đo tại chỗ 2026-07-13)

Plan §2 lập 2026-07-10 với ước lượng; phiên này **đo số thật**:

| Giả định plan | Thực tế đo | Hệ quả |
|---|---|---|
| RAM Qdrant là ràng buộc chính | **KHÔNG** — v3 = 23,727 points (~237 chunks/doc). 600 docs: ~142K points (ngoại suy same-mix) tới 167K (worst-case plan §2.2, toàn luật lớn) ≈ 145–171MB quantized dense RAM ≪ 1GB. Thông tư nhỏ hơn code lớn nên thực tế nghiêng cận dưới. Ràng buộc thật = **retrieval dilution** (distractor), không phải dung lượng. | Gate M2 capacity gần chắc pass; per-sector recall mới là gate đau. |
| — | Cluster free tier đang mang **5 collections**: `luat_law` (1670), `luat_law_ctx` (1670), `luat_law_ctx_v2` (2242), `luat_law_ctx_v3` (23727), `vietnamese_laws` (3441). | **Drop 3 baseline cũ** (`luat_law`, `luat_law_ctx`, `vietnamese_laws`) trước ingest để thu hồi RAM/disk. Giữ v2 (baseline M0) + v3 (production hiện tại) tới khi v4 pass gate. |
| Đo bytes/chunk cần dashboard | Qdrant `/telemetry?details_level=10` **phơi bày byte thật**: mỗi shard/segment có `vectors_size_bytes` + `payloads_size_bytes`; node có `ram_size`/`disk_size`. | **2.1 tự động hoá được** — script đọc telemetry, KHÔNG cần eyeball dashboard. |
| Enrichment chạy trước/sau ingest? | Enricher **đọc docs TỪ Qdrant** (`enricher.py` step 1: collect chunks → reconstruct text). `is_in_effect` backfill vào payload ở cuối enrichment. | **Enrichment chạy SAU ingest** (Phase E), không phải trước. Giữa ingest↔enrichment, `is_in_effect` = None → server filter giữ nguyên (fail-safe), không lọc oan. |

Source pool `data/raw/legal_docs_merged.parquet` = 518,255 rows, sẵn local. Phân bố
legal_type: Luật 595 · Nghị định 5,748 · **Thông tư 16,764** · Thông tư liên tịch 2,598
· Văn bản hợp nhất 3,579 · (Quyết định 249K + Công văn 124K = noise phi quy phạm, loại).

---

## 2. Nguyên tắc kế thừa (không thương lượng)

1. *Thà từ chối còn hơn trả lời sai* — 2 cổng từ chối giữ nguyên; refusal recall = 1.0 trên OOS không được tụt.
2. *Đo trước, tối ưu sau* — mỗi bậc scale chạy Tier-1 + validate; recall tụt >5 điểm → **dừng, tìm nguyên nhân**, không nhảy tiếp.
3. *Threshold là hàm phân bố* — corpus đổi → re-run Tier-2 + `calibrate_threshold.py`; không tin threshold 0.650 cũ sau 600.
4. *Không mất doc âm thầm* — mọi doc bị drop (superseded guard, content guard, chunker) phải log; corpus curated không rơi rớt lặng lẽ.
5. *Collection mới versioned* — 600 docs = `luat_law_ctx_v4`; giữ v3 tới khi v4 pass gate; flip `DEFAULT_COLLECTION_NAME` + threshold **cùng lúc**.

---

## 3. Thiết kế từng Phase

### Phase A — Capacity baseline (2.1, phần 1) · in-session

**Deliverable**: `scripts/measure_capacity.py`.

- Đọc `GET {QDRANT_URL}/telemetry?details_level=10` (api-key header).
- Với mỗi collection: sum `vectors_size_bytes + payloads_size_bytes` qua các shard/segment → **bytes/point**. In bảng tất cả collections (points, MB vectors, MB payload, bytes/point).
- **Projection function**: cho trước `target_points` → ước RAM (quantized dense = points × 1024 B + sparse + HNSW overhead) + disk (fp32 on-disk = points × 4096 B + payload) → so với trần free tier (1GB RAM / 4GB disk) và ngưỡng **<70%** (gate M2).
- In cảnh báo các collection baseline cũ nên drop (reclaim).

**Chạy ngay** trên v3 → baseline bytes/point. Ghi report `data/eval_reports/capacity_<ts>.{md,json}`.

**Verify**: số telemetry khớp `points_count` từ API; bytes/point > 0 và hợp lý (dense 1024-dim); projection cho v3-points ≈ số đo thực.

---

### Phase B — Tuyển corpus → ~600 (2.2 prep) · in-session → cổng review

**Deliverable**: `scripts/select_corpus_600.py` + bảng review + (sau duyệt) `data/raw/corpus_600docs.parquet`.

**B.1 Spine khoá cứng**: 100 doc-id trong `scripts/select_corpus.py::CORPUS` là **bất biến** — union vào 600 y nguyên (giữ sector label tay + so sánh M1→M2).

**B.2 Sinh candidate tự động** (~500):
- **Sector keyword map**: 6 lĩnh vực → tập token `legal_sectors` (vd doanh_nghiep → {Doanh nghiệp, Đầu tư, Thương mại, Chứng khoán}; dat_dai → {Bất động sản, Tài nguyên - Môi trường}; hinh_su → {Trách nhiệm hình sự, Thủ tục Tố tụng, Vi phạm hành chính}...). HNGĐ không có token riêng → dùng title-match ("hôn nhân", "gia đình", "trẻ em", "nuôi con nuôi", "bình đẳng giới") + spine anchoring.
- **Filter**: `legal_type ∈ {Luật, Bộ luật, Nghị định, Thông tư, Thông tư liên tịch, Văn bản hợp nhất}` **AND** sector match **AND** `len(content) ≥ MIN_CONTENT_LENGTH` **AND** không đã trong spine.
- **Superseded-drop guard**: loại doc có title/nội dung cho thấy đã hết hiệu lực mà bản thay thế KHÔNG trong tập candidate+spine (tránh bẫy "trích luật chết như luật sống" — chính là lý do 100-doc curation loại Nhà ở 2014/KDBĐS 2014). Heuristic: (a) title chứa năm cũ + tồn tại doc cùng chủ đề năm mới hơn trong tập → ưu tiên bản mới, gắn cờ cặp temporal; (b) doc có legal_type "Văn bản hợp nhất" ưu tiên hơn các sửa đổi rời cùng gốc. Guard là **best-effort**; cổng review bắt phần còn lại.
- **Ranking + budget per-sector — theo CHUNK count, không phải doc count** (xem B.5): Thông tư kích thước rất lệch nên budget theo doc không kiểm soát được phân bố chunk. Ước chunk/doc rẻ = `len(content) / ~700` (v3: p50=650 ký tự/chunk, 15.3M ký tự → ~23.7K chunks). Phân bổ để **max/min chunk-count giữa 6 sector ≤ ~3×** (quyết định user 2026-07-13). doanh_nghiep (M1 recall@5 0.7727, yếu nhất) **cap thận trọng**. Rank trong sector theo: (i) anchored-to-spine, (ii) **ưu tiên Văn bản hợp nhất** (giảm near-duplicate — xem B.5), (iii) recency, (iv) issuing authority cấp bộ chính, (v) content length hợp lý.

**B.3 Cổng review (user)**: xuất **bảng markdown/CSV** nhóm theo sector → parent-law, cột (id, legal_type, năm, title, sector, anchored?, lý-do-chọn). User prune/thêm/bớt. Danh sách duyệt → freeze thành `CORPUS_600` (list id + sector) trong script, tái lập được như `select_corpus.py`.

**B.4 Build parquet**: đọc `legal_docs_merged.parquet` theo row-group (như `select_corpus.py::extract_corpus`), lọc theo id đã duyệt, gắn `sector`, ghi `corpus_600docs.parquet`. In summary per-sector (docs / Luật / NĐ / Thông tư / M ký tự / **chunk ước tính**).

**B.5 Frequency bias & cân bằng chunk (đã chốt với user 2026-07-13)**

Chunk-count lệch giữa sector gây bias ở **3 chỗ**, reranker chỉ cứu 1:
- **First-stage recall (hại thật)**: cross-encoder chấm (query, chunk) độc lập tần suất → NẾU true positive vào được candidate pool thì rerank sửa thứ tự bất kể sector lớn/nhỏ. Hại nằm ở thượng nguồn: sector lớn (nhiều Thông tư lặp "phạt tiền từ… đến…") đẩy near-duplicate vào top-K, hất true positive RA khỏi pool TRƯỚC rerank. ⇒ **near-duplicate density trong sector** quan trọng hơn volume liên-sector.
- **Sparse/lexical skew**: BGE-M3 sparse là learned (không phải corpus-IDF) nên IDF-drift kiểu BM25 bị giảm, nhưng boilerplate tràn vẫn lệch tín hiệu lexical phần nào.
- **Ảo giác đánh giá**: recall trung bình đơn che bias → per-sector recall (đã có) là macro-fix.

**3 nhánh giảm thiểu, đòn bẩy chính ở nguồn:**
1. **Nguồn (Phase B)**: budget theo chunk-count, **bounded ratio ≤ ~3×** (B.2); **ưu tiên Văn bản hợp nhất** thay các sửa đổi rời cùng gốc → cắt near-duplicate density trực tiếp; superseded-drop hỗ trợ.
2. **Đo (Phase B/C + E)**: report chunk/sector + tín hiệu redundancy nội-sector (vd tỉ lệ chunk gần trùng theo article/embedding); eval thêm **per-sector first-stage recall (pre-rerank)** + **cross-sector leakage@k** (query sector S → % top-k rơi vào sector khác, cắt theo cỡ sector) → bias **quan sát được**, có gate, không giả định.
3. **Retrieval-level (defer M3 trừ khi số đòi)**: nếu diagnostic lộ first-stage bias dư → K first-stage lớn hơn cho sector dày / MMR diversity trong candidate / sector-aware normalization. Theo nguyên tắc 2 (đo trước) — KHÔNG build tới khi số yêu cầu.

**Verify**: mọi id duyệt tồn tại trong dataset (raise nếu thiếu, như `select_corpus.py`); 100 spine đều có mặt; per-sector count khớp budget; **max/min chunk-count ≤ ~3×**; in cảnh báo doc content ngắn bất thường.

---

### Phase C — Validate trước ingest + mở rộng eval · in-session

**C.1 Chunker alignment trên Thông tư** (gate mới): chạy `scripts/verify_chunker_alignment.py --data-path data/raw/corpus_600docs.parquet`. **Thông tư là cấu trúc chưa từng ingest** — chunker (hierarchical + article-level cho amendment) chỉ mới validate trên Luật/Nghị định. Gate: **WRONG_LABEL ≈ 0**, strict-lenient gap ≈ 0. Nếu Thông tư phá alignment (vd đánh số Điều/Mục khác, nhiều biểu mẫu) → **fix chunker** (mở rộng pattern) trước ingest; ghi diagnosis note như M1.2b.

**C.2 Re-project capacity + kiểm cân bằng chunk** với chunk-count THẬT từ C.1 (thay ước lượng): (a) xác nhận <70% trần — nếu vượt → trim theo ưu tiên sector (600 co giãn — plan §2.1); (b) **xác nhận max/min chunk-count per-sector ≤ ~3×** đúng như budget B.2/B.5 (ước lượng `len/700` có thể lệch thực tế) — nếu vượt ngưỡng → điều chỉnh danh sách trước ingest. In tín hiệu redundancy nội-sector.

**C.3 Mở rộng eval (2.7 phần 1)** — mục tiêu ~**30–50 câu mới** (đã chốt với user):
- Nhắm **nội dung Thông tư + docs mới**: mỗi sector thêm ~5–10 câu kiểu citation-lookup (Thông tư số X quy định gì), "phạt bao nhiêu/mức nào", "thủ tục/hồ sơ/mẫu đơn", so sánh.
- **Giữ nguyên OOS anchors** (thuế TNCN/TNDN, giao thông, xây dựng, nhà ở/KDBĐS, phi pháp lý) — không đổi vì không thêm lĩnh vực. Có thể thêm vài OOS trap mới nhắm Thông tư của lĩnh vực KHÔNG phủ.
- **Validate grounding**: `scripts/validate_eval_set.py --corpus-parquet data/raw/corpus_600docs.parquet` → 0 lỗi (mọi `expected_articles` tồn tại đúng document trong parquet). Đây là guard chống label stale (đã cắn 2 lần — mọi label verify parquet).

**Deliverable C**: corpus đã validate + eval set mở rộng đã validate grounding + (nếu cần) chunker fix.

---

### Phase D — Kaggle ingest (2.2) · USER chạy

**Deliverable (tôi)**: cập nhật `docs/notes/kaggle-ingest-guide.md` cho 600 docs.

- Attach `corpus_600docs.parquet` làm Kaggle Dataset.
- **Bước mới: drop 3 collection baseline cũ** trước ingest (reclaim RAM) — nêu rõ lệnh + xác nhận.
- Lệnh: `python scripts/ingest.py --data-path {DATA_PATH} --collection luat_law_ctx_v4 --quantized --device cuda --embed-batch-size 128 --resume`.
- `--quantized` **bắt buộc lần đầu** (int8 + fp32 on-disk chỉ set lúc tạo collection). Curated mode tự nhận `sector` column. Point ID deterministic → resume an toàn.
- Verify: `points_count` v4 ≈ tổng chunks script in ra.

**User chạy** → v4 sống trên Qdrant Cloud. (Đây là điểm handoff duy nhất.)

---

### Phase E — Post-ingest close-out (2.7 phần 2 + gates) · chung, sau ingest

Theo thứ tự phụ thuộc:

1. **Enrichment — MẶC ĐỊNH chạy trên Kaggle** (đã chốt với user 2026-07-13): chạy `EnrichmentPipeline` trên v4 (đọc chunks từ Qdrant → graph/timeline/amendment → SQLite MetadataStore `store.db` → **backfill `is_in_effect`** vào payload v4). Chạy **ngay trong notebook Kaggle sau ingest** (tránh scroll 140K chunks qua mạng nhà; CPU Kaggle đủ vì rule-based). Local against Qdrant Cloud chỉ là fallback. **Cần thêm**: (a) script runner enrichment standalone (hiện chỉ có API endpoint `/api/pipeline/enrich`) để gọi trong notebook; (b) sau enrichment, **mang `store.db` + `*.json` về repo** (Kaggle Dataset output rồi commit) để MetadataStore local + eval dùng được. Đo thời gian (quyết #13 batch/incremental có gấp không).
2. **Capacity re-measure** (2.1 gate cuối): `measure_capacity.py` trên v4 → **<70% trần free tier**. Ghi số thật.
3. **Tier-1 retrieval** trên v4: `scripts/eval.py` 3 bộ (criminal/sectors/general) → **per-sector recall@5**. Gate: **tụt ≤5 điểm** vs M1; báo cáo từng sector (không để 1 lĩnh vực gánh trung bình). doanh_nghiep là watch-item. **+ Frequency-bias diagnostic** (B.5): per-sector **first-stage recall pre-rerank** + **cross-sector leakage@k** — nếu sector dày hút true-positive của sector mỏng ra khỏi pool trước rerank → cân nhắc nhánh 3 M3 (không block M2, chỉ ghi watch-item).
4. **Tier-2 + recalibrate** (2.7): full Tier-2 → `scripts/calibrate_threshold.py` → **REFUSAL_THRESHOLD mới** cho v4 + bảng calibration + margin. Refusal recall = 1.0 trên OOS; false_refusal ghi rõ.
5. **Flip default**: `DEFAULT_COLLECTION_NAME` → `luat_law_ctx_v4` + threshold mới **cùng commit** (như commit v3 pairing).
6. **Cập nhật trạng thái**: bảng §7 Plan gốc → M2 ✅; memory dự án; ARCHITECTURE nếu cần.

**Artefact M2**: bảng **scale journey 6 → 100 → 600** kèm recall@5 per-sector từng bậc — câu chuyện "scale có kỷ luật" cho portfolio.

---

## 4. Gates M2 (tổng hợp từ plan + spec này)

| Gate | Ngưỡng | Phase |
|---|---|---|
| Chunker alignment | WRONG_LABEL ≈ 0, strict-lenient gap ≈ 0 (gồm Thông tư) | C.1 |
| Capacity | Qdrant < 70% trần free tier (RAM + disk) | A / C.2 / E.2 |
| Retrieval recall | recall@5 tụt ≤ 5 điểm vs M1, báo cáo per-sector | E.3 |
| Refusal | refusal recall = 1.0 trên OOS set | E.4 |
| Threshold | threshold mới có bảng calibration + margin | E.4 |
| P95 | end-to-end đo & ghi lại (chưa cần đạt — CPU đo ở M5) | (kế thừa 2.6) |

---

## 5. Rủi ro chính & guard

| Rủi ro | Giảm thiểu |
|---|---|
| Thông tư phá chunker alignment | Gate C.1 chặn trước ingest; fix pattern + diagnosis note; không ingest tới khi WRONG_LABEL≈0. |
| **Frequency bias** — chunk-count lệch giữa sector → sector dày hất true-positive sector mỏng khỏi pool trước rerank (B.5) | Nguồn: budget theo chunk-count bounded ratio ≤~3× + ưu tiên Văn bản hợp nhất (B.2/B.5). Đo: per-sector first-stage recall + cross-sector leakage@k (E.3). Retrieval-level (K lớn hơn / MMR / sector-norm) defer M3 tới khi số đòi. |
| Distractor dilution làm recall tụt (nhất là doanh_nghiep) | Budget per-sector thận trọng; spine anchoring; per-sector recall gate E.3; tụt >5pt → trim sector, không nhảy tiếp. |
| Auto-select nhặt luật chết (superseded, bản thay thế vắng) | Superseded-drop guard B.2 + cổng review user B.3 + enrichment is_in_effect E.1 (fail-safe: None→giữ). |
| Label eval stale (đã cắn 2 lần) | `validate_eval_set.py --corpus-parquet` bắt buộc pass 0 lỗi (C.3). |
| Enrichment scroll 140K chunks qua network chậm | Đo thời gian; fallback chạy enrichment trên Kaggle nếu cần. |
| Ingest rớt session | `--resume` + point ID deterministic (đã có); guide ghi rõ. |

---

## 6. Ngoài phạm vi (cố tình)

- **Không thêm lĩnh vực mới** → không redefine OOS (quyết định #2).
- **Không chạy ingest GPU trong phiên** — handoff Kaggle (quyết định #1).
- **Không fine-tune / benchmark ngoài / corrective RAG** — đó là M3.
- **Retrieval-level frequency-bias mitigation** (K theo sector / MMR diversity / sector-aware normalization) — chỉ build ở M3 NẾU diagnostic E.3 lộ bias dư (nguyên tắc 2: đo trước). M2 chỉ làm nguồn (B.5 nhánh 1) + đo (nhánh 2).
- **Server-side date-range point-in-time (`at_date`)** — vẫn client-side (cần datetime index), follow-up nhỏ như spec scale-prep đã flag.
- **Fully hand-curate 600** — loại (quyết định #3).

---

## 7. Definition of Done (spec này)

**Trong phiên (A–C):**
- [ ] A: `measure_capacity.py` chạy ra bytes/point v3 + projection 600 vs free tier; report artifact.
- [ ] B: `select_corpus_600.py` sinh candidate (budget chunk-count, bounded ratio ≤~3×, ưu tiên VBHN) + bảng review; user duyệt; `corpus_600docs.parquet` build xong, 100 spine có mặt, summary per-sector + chunk ước tính.
- [ ] C: chunker alignment trên 600 WRONG_LABEL≈0 (fix nếu cần); capacity re-project <70% + xác nhận chunk-count per-sector ≤~3×; eval mở rộng ~30–50 câu (+ diagnostic first-stage recall/leakage), validate grounding 0 lỗi.
- [ ] D-prep: `kaggle-ingest-guide.md` cập nhật cho 600 + bước drop collection cũ + collection v4.

**Sau ingest (D–E, phiên tiếp):**
- [ ] D: user ingest v4; points_count khớp.
- [ ] E: enrichment **trên Kaggle** + is_in_effect backfill, mang `store.db` về repo; capacity gate; Tier-1 per-sector ≤5pt + frequency-bias diagnostic (first-stage recall/leakage) ghi lại; Tier-2 recalibrate threshold mới; flip default+threshold cùng commit; bảng scale journey; cập nhật plan §7 + memory.

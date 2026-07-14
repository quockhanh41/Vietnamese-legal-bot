# M3 Kickoff — Sequencing + 3.1 Benchmark ngoài (Zalo/ALQAC) — Design Spec

> **Ngày**: 2026-07-15 · **Milestone**: M3 — SOTA theo số đo (mở sau M2 ✅ 2026-07-15)
> **Tiền đề**: [`docs/Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3/M3
> và roadmap §3.5/§3.7/§3.8
> **Trạng thái**: design đã duyệt (user 2026-07-15, plan-mode approval) — spec này là bản ghi

---

## 0. Bốn quyết định khung (đã chốt với user 2026-07-15)

| # | Câu hỏi | Chốt |
|---|---|---|
| 1 | Kickoff M3 phạm vi nào? | **Sequencing toàn milestone + design/execute riêng 3.1.** 3.1 là tiền đề gate của 3.2 (fine-tune) và là baseline khách quan cho mọi mục sau. Các mục sau có spec→plan riêng. |
| 2 | Benchmark nào trước? | **Zalo Legal Text Retrieval 2021 trước** (public hoàn toàn, ~61K điều, ~3.2K query train). **ALQAC thêm nếu tải được sạch** (2021–2023, timebox 1–2h; không được thì ghi lý do, đi tiếp Zalo-only). |
| 3 | Gate M3 cũ ("refusal recall 1.0") mâu thuẫn kết quả M2? | **Restate baseline-relative**: mọi mục M3 giữ **refusal recall ≥ 0.765 VÀ false_refusal ≤ 0.136** trên OOS set, **không câu OOS nào leak MỚI**. Trần confident-OOS (gen-903, deferred từ M2) thành **mục 3.7 mới** — mục tiêu NÂNG recall lên trên 0.765. |
| 4 | Harness 3.1 hình dạng nào? | **Local Qdrant + ablation 3 tầng** (dense / hybrid RRF / hybrid+rerank), index điều-level as-is, embed Kaggle GPU. Production defaults không đụng. |

## 1. Thứ tự thực thi M3 (theo ROI, đã chốt)

| Thứ tự | Mục | Lý do |
|---|---|---|
| 1 | **3.1 Benchmark ngoài (Zalo → ALQAC)** | Baseline khách quan; tiền đề gate 3.2 |
| 2 | **3.7 (mới) Confident-OOS ceiling (gen-903)** → **3.3 Corrective RAG** → **3.4 comparison-fix** | Cả ba đánh vào ranh giới refusal; đo riêng từng mục; gen-903 trước để vòng retry của 3.3 không mở rộng leak chưa đo được |
| 3 | **3.2 Fine-tune bi-encoder** | Cần harness + split cố định từ 3.1; bet SOTA lớn nhất |
| 4 | **3.5 Graph expansion (route complex)** | Độc lập; multi-hop recall |
| 5 | **3.6 Verbatim-span quotes** | Độc lập; siết verifier |
| — | Chore: `store.db` → JSON export cho fresh-clone (M2 deferral) | Ghép cùng mục sớm bất kỳ |

Giữ nguyên tắc plan: mỗi mục có số trước/sau trên eval set; không cải thiện → bỏ có ghi chép.

## 2. Design 3.1 — Benchmark ngoài

**Goal**: retrieval stack thật của mình đo trên benchmark pháp lý VN công khai —
recall@{1,5,10}, MRR@10, nDCG@10 — ablation 3 tầng, so sánh được với số đã công bố.

### 2.1. Các bước

1. **Data** — `scripts/download_zalo.py` → `data/benchmarks/zalo/`: `legal_corpus.json`
   (kỳ vọng **61,425 điều / 8,586 văn bản**) + `train_question_answer.json` (kỳ vọng
   **~3,196 query** với qrel `(law_id, article_id)`). Nguồn: mirror HF/Kaggle —
   **validate count trước khi dùng** (chống mirror hỏng). ALQAC → `data/benchmarks/alqac/`
   (timebox, xem §0.2).
2. **Split cố định NGAY** — 80/10/10 train/dev/test query Zalo, seed cố định, persist
   `data/benchmarks/zalo/splits.json` + test determinism. 3.1 báo zero-shot trên full set
   VÀ test split; 3.2 sau này chỉ so trên held-out test (chống leakage).
3. **Index corpus** — điều-level as-is (điều = đơn vị ground-truth của benchmark; KHÔNG
   chunker, KHÔNG LLM enrichment; embed `title + text`). Embed offline **Kaggle GPU** theo
   pattern parquet→upsert của M2 (`docs/notes/kaggle-ingest-guide.md`); upsert vào
   **local docker Qdrant** collection `zalo_legal_2021`, payload `{law_id, article_id}`.
   Qdrant Cloud free tier không đụng.
4. **Runner** — `scripts/bench_external.py`:
   - Tầng A: dense-only (`search_dense`, `src/db/qdrant_client.py:281`)
   - Tầng B: hybrid RRF (`search_hybrid`, `src/db/qdrant_client.py:326`)
   - Tầng C: hybrid + cross-encoder rerank (đúng reranker prod), pool top-50
   - A/B chạy toàn bộ query; **C chỉ chạy test split ~320 query** (CPU rerank @pool50
     ≈ 2.2s/query theo bench M2 — full set ~2h; test split ~12 phút)
   - Metrics module có unit test (TDD): recall@k, MRR@10, nDCG@10 trên qrel `(law_id, article_id)`
   - Report JSON → `data/eval_reports/` (naming convention hiện có)
5. **Writeup** — `docs/notes/external-benchmark-zalo-alqac.md`: bảng ablation, so sánh số
   đã công bố (có citation), định nghĩa split, runtime, giới hạn (điều-level ≠ chunking
   production — phạm vi đo: embedding + fusion + rerank stack).
6. **Docs**: plan doc §7 M3 → 🔄; gate restated ghi vào §3/M3.

### 2.2. Reuse (không viết lại)

- `search_dense` / `search_hybrid` (`src/db/qdrant_client.py`) — không tự implement fusion.
- Reranker prod (như `scripts/bench_reranker.py` tham chiếu) cho tầng C.
- Pattern embed offline Kaggle của M2.
- Convention report của `scripts/eval.py` / `data/eval_reports/`.

### 2.3. Cố tình KHÔNG làm trong 3.1

- Không đổi production collection / threshold / routing.
- Không đo LLM/generation (retrieval-only — "đo retrieval tách khỏi generation", roadmap §3.5).
- Không fine-tune (đó là 3.2, gate bằng chính số 3.1).

## 3. Verification & gate đóng 3.1

- Unit test metrics + split determinism pass; `make test` giữ xanh (suite 349+).
- Số không degenerate: dense recall@10 nằm trong dải khả tín (BGE-M3-class zero-shot trên
  Zalo công bố quanh ~0.7–0.9 recall@10); lệch xa → debug harness trước khi tin số.
- Count dataset validate lúc download.
- **Gate đóng 3.1**: bảng ablation đủ 3 tầng trên Zalo (+ALQAC hoặc blocker có ghi chép),
  writeup committed, plan doc cập nhật.

## 4. Workflow

Sonnet sub-agent implement → Fable verify độc lập → commit (brief tiếng Anh cho sub-agent).

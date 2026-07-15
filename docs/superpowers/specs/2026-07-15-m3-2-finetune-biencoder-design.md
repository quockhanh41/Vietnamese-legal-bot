# M3.2 — Fine-tune bi-encoder legal tiếng Việt — Design Spec

> **Ngày**: 2026-07-15 · **Milestone**: M3.2 (sau 3.1 ✅ — plan gốc xếp 3.2 sau nhóm 3.7/3.3/3.4, user chọn làm 3.2 ngay theo yêu cầu phiên này)
> **Tiền đề**: [`docs/Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3/M3.2 (roadmap §3.8),
> M3.1 writeup [`docs/notes/external-benchmark-zalo-alqac.md`](../../notes/external-benchmark-zalo-alqac.md),
> split đóng băng [`data/benchmarks/zalo/splits.json`](../../../data/benchmarks/zalo/splits.json)
> **Trạng thái**: design đã duyệt (user 2026-07-15, per-section approval) — spec này là bản ghi

---

## 0. Quyết định khung (đã chốt với user 2026-07-15)

| # | Câu hỏi | Chốt |
|---|---|---|
| 1 | Training data v1? | **Zalo-only**: 2,556 cặp train (split đóng băng) + hard negatives mined. ALQAC blocked (gated). Synthetic queries từ corpus 600-doc = **iteration 2 CHỈ KHI** v1 thắng Zalo nhưng thua eval nội bộ (tín hiệu domain shift). |
| 2 | Luật thắng/thua trên Zalo test? | **Margin rule tầng B (hybrid — đường production)**: thắng = recall@10 ≥ **0.885** (baseline 0.875 + 1.0pt ≈ 3 query) **VÀ** MRR@10 ≥ **0.669** trên test 321 query. Checkpoint chọn trên dev; test chạy **một lần** mỗi candidate. |
| 3 | Phạm vi M3.2 khi thắng? | **Verdict incl. internal**: train → verdict Zalo → (nếu thắng) so sánh Tier-1 nội bộ trên collection song song → writeup. **Swap production default + recalibrate threshold + refusal gates = follow-up riêng** (quyết định với user), KHÔNG thuộc M3.2. |
| 4 | Phương pháp fine-tune? | **A — FlagEmbedding unified fine-tune chính chủ** (`FlagEmbedding.finetune.embedder.encoder_only.m3`, dense+sparse train cùng nhau, self-distillation) → checkpoint chuẩn BGE-M3, `BGEM3FlagModel` load không đổi code production. LoRA dense-only = fallback VRAM có ghi chép. Model VN nhỏ hơn: loại (mất sparse/hybrid). |

**Nguyên tắc kế thừa**: đo trước tối ưu sau — thua vẫn ra writeup (plan: "BGE-M3 zero-shot vẫn tốt hơn" cũng là kết quả); split 20260715 là hợp đồng đo, không regenerate.

## 1. Thang thí nghiệm (mỗi bậc là điểm dừng/đi)

```
1. Data prep (local)          → verify: unit tests leakage/schema/determinism
2. Train (Kaggle GPU)         → verify: loss giảm, 3 checkpoint/epoch lưu được
3. Chọn checkpoint (dev)      → verify: dense-proxy recall@10 dev, chọn 1 checkpoint
4. Verdict Zalo (local, test) → verify: margin rule §0.2 — THUA → bước 6
5. (thắng) So sánh nội bộ     → verify: gate §4.2
6. Writeup (luôn luôn)        → verify: bảng trước/sau + decision trail đủ tái lập
```

## 2. Data pipeline — `scripts/prepare_finetune_data.py` (mới)

- **Input**: `train_question_answer.json` lọc theo `splits.json["train"]` (2,556 query) + `legal_corpus.json`. **Assert trong code**: không id nào thuộc dev/test lọt vào training data.
- **Positive** (1/query): điều liên quan, format `title + "\n" + text` — **trùng khớp `embed_text` đã index** (train và index cùng phân phối).
- **Hard negatives**: mine local trên collection `zalo_legal_2021` sẵn có bằng BGE-M3 stock — top-100 hybrid, loại positive, **sample 7 từ rank 11–100, seed cố định**. Bỏ rank 1–10 để né false negatives (mirror rớt ~1% multi-relevant qrel). Điều khác cùng luật giữ nguyên — hard negative hợp lệ.
- **Output**: `data/benchmarks/zalo/finetune/train.jsonl` format FlagEmbedding `{"query", "pos": [...], "neg": [...]}`. **Không commit** (hàng chục MB) — regenerate deterministic; commit file stats/checksum nhỏ thay thế.

## 3. Training — Kaggle GPU, recipe chính chủ

- `torchrun -m FlagEmbedding.finetune.embedder.encoder_only.m3` từ `BAAI/bge-m3`:
  `--unified_finetuning True --use_self_distill True`, fp16 + gradient checkpointing,
  LR 1e-5, temperature 0.02, `train_group_size 8` (1 pos + 7 neg),
  `query_max_len 128` / `passage_max_len 1024`, **3 epoch, save mỗi epoch**, seed cố định.
- Guide chạy Kaggle trong `docs/notes/` (pattern như `kaggle-ingest-guide.md`), lệnh exact + pip install.
- **Bậc thang VRAM (theo thứ tự)**: giảm batch → `passage_max_len` 1024→512 → gradient accumulation → **cuối cùng: LoRA fallback (phương án B)** — nếu tới đây, tự nó là một kết quả có ghi chép.
- **Forgetting hedge**: 2,556 cặp là nhỏ so với 568M params — LR thấp, ≤3 epoch, chọn checkpoint trên dev.
- Ngân sách: ~3–5h GPU tổng — trong 30h/tuần Kaggle.
- Session drop: checkpoint mỗi epoch vào `/kaggle/working`; embed script đã resume-safe theo shard.

## 4. Evaluation

### 4.1. Chọn checkpoint (dev, trên Kaggle) + verdict Zalo (local, test)

- **Dev proxy**: script nhỏ chấm mỗi epoch-checkpoint bằng **dense in-memory recall@10 trên dev (319 query)** ngay trên Kaggle — khỏi tải 3×2.3GB về; chỉ checkpoint thắng về máy. Proxy dense-only là chủ đích (rẻ); verdict thật là hybrid trên test.
- **`--model <path>` flag mới** cho `scripts/ingest_zalo_corpus.py embed` và `scripts/bench_external.py` (default không đổi = BGE-M3 stock; `BGEEmbedder(model_name=...)` nhận local path sẵn).
- Re-embed corpus Zalo bằng checkpoint chọn (Kaggle, ~10–15 phút) → shards → upsert local `zalo_legal_2021_ftv1` → chạy **dev trước** (sanity) rồi **test MỘT lần**, levels A/B.
- **Verdict = margin rule §0.2.** Level C (rerank) chạy thêm trên winner cho writeup — informative, không gate.

### 4.2. So sánh nội bộ (CHỈ KHI thắng Zalo)

- Re-embed chunk 600-doc corpus (embed text sẵn có — **không cần LLM/enrichment lại**) bằng model FT trên Kaggle → collection song song `luat_law_ctx_v4_ftv1` **config y hệt v4** (quantization int8, on-disk) — apples-to-apples.
- Chạy đúng Tier-1 routed recall@5 eval hiện có trên collection song song.
- **Gate nội bộ: overall routed recall@5 ≥ 0.845 (baseline v4) VÀ không sector nào tụt > 2pt.** Nếu đúng bằng parity (≥ nhưng không >) → flag cho user quyết ở follow-up, không tự pass im lặng.
- Collection song song **xóa sau verdict** (tạm chiếm thêm ~9.5% RAM free tier → ~19% tổng, an toàn <70%).
- Refusal/threshold: **không đụng trong M3.2** — gate M3 baseline-relative (recall ≥ 0.765, false_refusal ≤ 0.136) chỉ áp khi adoption follow-up đổi default.

## 5. Artefact + phạm vi

| Thứ | Chi tiết |
|---|---|
| Writeup | `docs/notes/finetune-biencoder-zalo.md` — **thắng hay thua đều viết**: bảng trước/sau dev+test, config train, data stats, decision trail |
| Reports | `data/eval_reports/<ts>_zalo_bench.json` (convention sẵn); so sánh nội bộ theo convention Tier-1 |
| Checkpoint | Không vào git; nếu thắng → cân nhắc publish HF Hub (quyết định nhỏ riêng, portfolio artifact) |
| Data prep script + tests | `scripts/prepare_finetune_data.py` + unit tests |
| Kaggle guide | `docs/notes/` (train + dev-proxy + re-embed) |

**Ngoài phạm vi (chốt)**: swap production default, recalibrate REFUSAL_THRESHOLD, refusal/OOS re-check (adoption follow-up); synthetic-query iteration (chỉ mở nếu thắng-Zalo-thua-nội-bộ); ALQAC (blocked); model VN khác BGE-M3.

## 6. Testing (Tier-1, không GPU)

- Data prep: **không leakage dev/test** (assert + test), positive không nằm trong negatives, schema jsonl đúng, output deterministic với seed cố định.
- Arg plumbing `--model` cho 2 script (test nhẹ, không load model thật).
- Metrics/split module: đã có test từ M3.1, không đổi.

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| VRAM 16GB không đủ full FT 568M | Bậc thang §3; LoRA fallback là kết quả có ghi chép, không phải thất bại |
| Overfit/forgetting trên 2,556 cặp | LR 1e-5, ≤3 epoch, chọn trên dev, verdict held-out test một lần |
| False negatives trong mining | Sample rank 11–100; qrel 1-relevant/query đã ghi nhận ở M3.1 §3 |
| Dev proxy (dense) chọn sai checkpoint cho gate hybrid | Chấp nhận có chủ đích (rẻ); dev sanity-run hybrid local trước test bắt lệch lớn |
| Thắng Zalo, thua nội bộ | Chính là lý do gate 2 tầng; mở iteration 2 synthetic queries (quyết định riêng) |
| Kaggle session drop giữa train | Checkpoint mỗi epoch; train ~giờ, retry rẻ |

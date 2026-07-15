# M3.1 — Benchmark ngoài: Zalo Legal Text Retrieval 2021 (+ALQAC)

> **Ngày**: 2026-07-15 · **Milestone**: M3.1 (spec [`docs/superpowers/specs/2026-07-15-m3-external-benchmark-design.md`](../superpowers/specs/2026-07-15-m3-external-benchmark-design.md), plan [`docs/superpowers/plans/2026-07-15-m3-external-benchmark.md`](../superpowers/plans/2026-07-15-m3-external-benchmark.md))
> **Mục tiêu**: đo retrieval stack thật của mình (BGE-M3 dense / hybrid RRF / +BGE-reranker) trên bộ đo pháp lý VN công khai — recall@{1,5,10}, MRR@10, nDCG@10 — ablation 3 tầng, split cố định cho M3.2.

## TL;DR

- Zero-shot trên **3,196 query** (toàn train set Zalo-2021): dense recall@10 **0.885**, hybrid RRF recall@10 **0.911** — nằm đúng dải khả tín BGE-M3-class (~0.7–0.9) → harness đo đúng, không degenerate.
- Rerank (tầng C) **không** nâng recall@10 (cap bởi pool hybrid) nhưng nâng mạnh **precision đầu bảng**: recall@1 0.551→0.685, MRR@10 0.669→0.757, nDCG@10 0.720→0.786 trên held-out test — đổi lại latency CPU **~8.3s/query** (pool 50, article-level passage dài).
- Split 80/10/10 seed 20260715 đóng băng ([`data/benchmarks/zalo/splits.json`](../../data/benchmarks/zalo/splits.json)) — hợp đồng đo cho M3.2 fine-tune.
- ALQAC: chưa lấy được data trong pass này (cổng đăng ký/liên hệ ban tổ chức) — ghi blocker bên dưới.

## 1. Kết quả

### 1.1. Headline — zero-shot toàn train set (3,196 query)

Nguồn: [`data/eval_reports/20260715T054132Z_zalo_bench.json`](../../data/eval_reports/20260715T054132Z_zalo_bench.json)

| Tầng | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 | P50 (ms) | P95 (ms) | #query |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A — dense (`search_dense`) | 0.571 | 0.830 | 0.885 | 0.680 | 0.730 | 3.6 | 4.6 | 3196 |
| B — hybrid RRF (`search_hybrid`) | **0.596** | **0.857** | **0.911** | **0.707** | **0.757** | 4.4 | 5.8 | 3196 |

Hybrid RRF thắng dense trên **mọi** metric (+2.6pt recall@10) với latency gần như không đổi (~ms, server-side RRF của Qdrant).

### 1.2. Ablation 3 tầng — held-out test split (321 query, apples-to-apples)

Nguồn: [`data/eval_reports/20260715T062649Z_zalo_bench.json`](../../data/eval_reports/20260715T062649Z_zalo_bench.json) (pool rerank = 50)

| Tầng | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 | P50 (ms) | P95 (ms) | #query |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A — dense | 0.564 | 0.794 | 0.847 | 0.662 | 0.707 | 3.7 | 5.1 | 321 |
| B — hybrid RRF | 0.551 | 0.832 | **0.875** | 0.669 | 0.720 | 4.5 | 6.5 | 321 |
| C — hybrid + rerank (BGE-reranker-v2-m3, CPU) | **0.685** | **0.838** | **0.875** | **0.757** | **0.786** | 8270.6 | 8752.5 | 321 |

**Đọc bảng:**
- **C không nâng recall@10** so với B (0.875 = 0.875): cross-encoder chỉ **sắp xếp lại** trong pool 50 của hybrid — không thêm được ứng viên mới, nên hit-rate@10 bị chặn bởi chính pool hybrid. Nếu điều liên quan đã nằm trong top-50 nhưng ngoài top-10 thì rerank kéo lên; nếu đã ngoài top-50 thì rerank bó tay.
- **C nâng mạnh chất lượng đầu bảng**: recall@1 0.551→**0.685** (+13.4pt), MRR@10 0.669→**0.757**, nDCG@10 0.720→**0.786**. Đây đúng vai trò cross-encoder: đẩy điều đúng lên hạng 1.
- **Giá phải trả**: P50 latency **8.27s/query** (CPU, fp16, pool 50) so với 4.5ms của hybrid — chậm ~1800×. Xem §4 (Limitations) về passage article-level dài làm phồng latency.

> Lưu ý: recall@1 của B trên test (0.551) < A (0.564) là dao động split nhỏ (321 query); trên full set B ≥ A ở mọi metric. Test split chốt để làm **baseline held-out cho M3.2**, không phải để tuyên bố A>B.

## 2. Split cố định (hợp đồng đo với M3.2)

- File: [`data/benchmarks/zalo/splits.json`](../../data/benchmarks/zalo/splits.json), committed (`6ea497c`).
- 80/10/10 = **train 2556 / dev 319 / test 321**, seed **20260715**, `src/eval/external_bench.make_splits` (sort → seeded shuffle → slice; thứ tự input không ảnh hưởng, có unit test determinism).
- **Vì sao đóng băng**: M3.1 báo zero-shot trên full set VÀ trên test; M3.2 fine-tune được train trên `train`, tune trên `dev`, và **chỉ so với M3.1 trên `test`**. Regenerate split = phá so sánh → `scripts/download_zalo.py` từ chối ghi đè `splits.json` nếu đã tồn tại.

## 3. So với số đã công bố

BGE-M3 trên Zalo-2021 (fine-tune + hard-negative mining, pipeline hybrid dense+BM25+cross-encoder) được báo quanh **recall@10 ≈ 0.8985, MRR ≈ 0.7444** ([johal.in writeup](https://www.johal.in/finetuning-embedding-model-multilingual-legal-documents-bgem3-vs/); tham chiếu code [hieudx149/ZaloAI2021_LTR](https://github.com/hieudx149/ZaloAI2021_LTR)).

**KHÔNG so sánh trực tiếp được** — nêu rõ để khỏi over-claim:
1. **Tập đo khác**: số công bố thường trên **test set** của competition (leaderboard, qrel private không phát hành); số của mình là **zero-shot trên public train set** — tập duy nhất có qrel trong mirror. Không train trên nó.
2. **Zero-shot vs fine-tuned**: số công bố là model đã fine-tune trên chính Zalo; của mình là BGE-M3 stock (chưa fine-tune — đó là việc của M3.2).
3. **Mirror qrel 1-relevant/query**: mirror HF này giữ **đúng 1 điều liên quan/câu** (~1% multi-relevant của bản gốc bị rớt), nên recall@k ≡ hit-rate@k. Số leaderboard gốc dùng F2 trên qrel đa-điều → thang khác.

Kết luận thận trọng: hybrid RRF zero-shot của mình (recall@10 0.911 full / 0.875 test) **ngang tầm** số fine-tuned công bố (~0.90) — hợp lý cho BGE-M3 stock, và là baseline sạch để M3.2 chứng minh fine-tune có thắng thật hay không.

## 4. Limitations (phạm vi đo)

- **Article-level ≠ chunking production**: index nguyên điều (`title + "\n" + text`) vì điều là đơn vị ground-truth của benchmark — KHÔNG qua chunker/LLM-enrichment của production. Phạm vi đo = **embedding + fusion + rerank stack**, không phải pipeline chunk 600-doc thật.
  - Hệ quả: passage article-level **dài hơn** chunk production → cross-encoder rerank tốn hơn (P50 8.3s/pool50 ở đây vs P95 2180ms/pool50 đo trên chunk M2 warm-repeats). Latency C là chỉ dấu tương đối, không phải số production.
- **Corpus 2021-frozen**: luật Zalo-2021, không phải corpus 600-doc của mình. Đo khả năng retrieval trên phân phối khác, không phải chất lượng dữ liệu của mình.
- **Mirror**: 3,271 văn bản / 61,425 điều / 3,196 query; qrel đúng 1-relevant/query (xem §3).
- **Retrieval-only**: không gọi LLM/generation ở đâu trong M3.1 (đo retrieval tách khỏi generation).

## 5. ALQAC — trạng thái

Kiểm tra nhanh trong pass này (không dùng trọn timebox 2h của Task 6 — ngoài phạm vi 2 task được giao):
- `https://alqac.github.io/` (ALQAC 2026 là năm mới nhất): trang không có link tải trực tiếp; data access qua **liên hệ ban tổ chức / đăng ký**, không grant tức thì.
- **Blocker**: dataset gated sau form/đăng ký. Chưa lấy được bản sạch để chạy pipeline.
- Khi lấy được: schema retrieval cùng dạng qrel `(law_id, article_id)` như Zalo (ALQAC 2021 xây trên chính corpus Zalo-2021) → tái dùng Task 4/5, thêm loader `load_alqac_*` trong `scripts/download_zalo.py`, collection `alqac_<year>`, report full-set (trainset ALQAC nhỏ, không cần split).

## 6. Cách chạy (tái lập)

**Hardware**: embed corpus trên Kaggle T4 (CUDA); upsert + query-embed + rerank trên máy local (Apple Silicon MPS cho embed, CPU cho reranker — pin CPU khớp production, xem `scripts/bench_reranker.py`); Qdrant local qua docker.

```bash
# 1) Embed corpus → 31 parquet shard (Kaggle T4, ~10–15 phút)
python scripts/ingest_zalo_corpus.py embed \
    --corpus data/benchmarks/zalo/legal_corpus.json \
    --out-dir /kaggle/working/vectors \
    --device cuda --batch-size 128 --shard-size 2000
# → tải shard về data/benchmarks/zalo/vectors/

# 2) Upsert vào local Qdrant (docker compose up -d qdrant trước)
.venv/bin/python scripts/ingest_zalo_corpus.py upsert \
    --shards "data/benchmarks/zalo/vectors/vectors_*.parquet" \
    --collection zalo_legal_2021 --qdrant-url http://localhost:6333
# Gate: "✅ point count matches corpus" (61,425)

# 3) Ablation
.venv/bin/python scripts/bench_external.py --levels a,b --split full --device mps   # headline A/B, ~vài phút
.venv/bin/python scripts/bench_external.py --levels a,b,c --split test --device mps  # ablation A/B/C, C ~30–45 phút CPU rerank
```

Report → `data/eval_reports/<UTC-ts>_zalo_bench.json` (cùng convention `scripts/eval.py`).

## 7. Artefact

| Thứ | Path |
|---|---|
| Downloader + validate + freeze split | `scripts/download_zalo.py` |
| Split module (frozen, determinism-tested) | `src/eval/external_bench.py`, `data/benchmarks/zalo/splits.json` |
| Metrics exact-ID (recall@k, MRR@10, nDCG@10) | `src/eval/bench_metrics.py` |
| Embed→shard + local upsert | `scripts/ingest_zalo_corpus.py` |
| Runner ablation 3 tầng | `scripts/bench_external.py` |
| Report full A/B | `data/eval_reports/20260715T054132Z_zalo_bench.json` |
| Report test A/B/C | `data/eval_reports/20260715T062649Z_zalo_bench.json` |

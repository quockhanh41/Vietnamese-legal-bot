# AITeamVN adoption GO/NO-GO (evidence-only) — Design Spec

> **Ngày**: 2026-07-16 · **Milestone**: follow-up sau nhánh so sánh AITeamVN
> **Tiền đề**: nhánh `m3-aiteamvn-comparison` — AITeamVN mixed **thắng stock** trên 600-doc
> ([`docs/notes/aiteamvn-internal-comparison.md`](../../notes/aiteamvn-internal-comparison.md));
> **gate #1 refusal PASS** (refusal_recall 0.643→**0.786**, false_refusal 0.074→0.077 phẳng;
> gen-903 confident-OOS của M2 giờ được refuse đúng).
> **Trạng thái**: design đã duyệt (user 2026-07-16, per-question) — spec này là bản ghi.

---

## 0. Quyết định khung (đã chốt với user 2026-07-16)

| # | Câu hỏi | Chốt |
|---|---|---|
| 1 | Hình dạng embedding production? | **Mixed** (AITeamVN dense + bge-m3 sparse), recall@5 **0.922** (best). Chấp nhận gánh **2 model** → **gate #3 (RAM 16GB) là cổng cứng**. Fallback nếu không vừa: dense-only 0.898 (vẫn > v4 hybrid 0.874). |
| 2 | Milestone kết thúc ở đâu? | **Dừng ở evidence GO/NO-GO — KHÔNG đụng production.** Rewire embedder + flip default + đổi threshold = implementation riêng CHỈ KHI GO. |
| 3 | Reranker? | **Bundled** — AITeamVN reranker đi cùng mixed embedding trong mọi phép đo (đã validate chung ở nhánh so sánh). |

**Nguyên tắc**: đo để quyết, không tự đổi production. Mọi cổng baseline-relative so v4 trên cùng bộ/collection.

## 1. Bốn workstream → một verdict

```
1. Gate #2  recalibrate threshold cho stack mới      → ngưỡng đề xuất + refusal/false-refusal tại đó
2. Gate #3  deploy RAM/latency (CỔNG CỨNG)           → fits-headroom / tight / doesn't-fit vs 16GB
3. Broaden  criminal Tier-1 + Tier-2 (new vs v4)     → verdict không chỉ dựa sectors+general
4. Debug    gen-907 / gen-912 (2 OOS leak mới)       → root cause
→ Writeup GO/NO-GO (docs/notes/aiteamvn-adoption-go-no-go.md)
```

## 2. Gate #2 — Recalibrate threshold (stack mới)

- **Stack**: mixed (AITeamVN dense + bge-m3 sparse) + AITeamVN reranker, collection `luat_law_ctx_v4_aiteamvn`.
- **Tier-2 đã có** trên stack mới: general `20260716T034715Z` (n=41, **1 error** → `--retry` để sạch trước khi calibrate), sectors `20260716T030111Z` (n=83, 0 err). **Chạy MỚI**: criminal Tier-2 (§4).
- **Calibrate**: `scripts/calibrate_threshold.py` — methodology M1.7 (sweep CHỈ trên rows qua confidence gate `pipeline_used != "out-of-scope"`; scope-gate section riêng). Pool 3 bộ.
- **Report**: ngưỡng đề xuất cho stack mới; refusal_recall + false_refusal tại ngưỡng đó **vs 0.700 hiện tại**; phân bố confidence theo expected_behavior.
- **Không** đổi hằng `REFUSAL_THRESHOLD` trong code (chỉ báo cáo con số đề xuất).

## 3. Gate #3 — Deploy RAM/latency (CỔNG CỨNG)

- **Đo peak resident RAM (RSS)** khi nạp **cả 3 model** (AITeamVN dense + bge-m3 sparse qua `MixedHybridEmbedder` + AITeamVN reranker `FlagReranker`) rồi chạy **1 query thật** end-to-end (embed → `search_hybrid` → rerank pool ~10).
- **Đo P95 latency** (reuse pattern `scripts/bench_reranker.py`, pool ~10 theo cap M2.6 CPU deploy).
- **Device = CPU** (khớp mốc M5 HF Spaces, không GPU). MixedHybridEmbedder trên CPU không tách GPU/fp16 → đúng hình dạng deploy.
- **Verdict** so ngân sách **HF Spaces free 16GB**: `fits-headroom` / `tight` / `doesn't-fit`.
  - `doesn't-fit` → **NO-GO cho mixed**; ghi fallback dense-only (0.898, 1 model, vẫn > v4 hybrid).
- **Script mới**: `scripts/measure_stack_memory.py` — peak RSS (qua `resource.getrusage`/`psutil`) + P50/P95 latency, in report `data/eval_reports/`.
- **Caveat ghi rõ**: đo local ≈ xấp xỉ HF Spaces (khác OS/allocator/torch build) — đo **xu hướng + biên độ**, không con số tuyệt đối.

## 4. Broaden eval — criminal

- **Tier-1** (mixed) + **Tier-2** (mixed + AITeamVN reranker) trên `criminal` set, collection `luat_law_ctx_v4_aiteamvn`, **vs v4 baseline criminal**.
- Reuse `scripts/eval.py` (`--set criminal --embedder mixed [--answers --reranker AITeamVN/Vietnamese_Reranker]`).
- Report: recall@1/3/5/10 + MRR (Tier-1); refusal/false-refusal/citation (Tier-2). So v4 baseline criminal gần nhất.

## 5. Investigate gen-907 / gen-912 (2 OOS leak mới)

- 2 câu v4 refuse đúng → AITeamVN **answer** (leak). systematic-debugging: route đã đi, retrieval kéo doc nào vào context, tín hiệu confidence (reranker/context/self-eval) lệch ra sao vs v4.
- **Phân loại root cause**: embedding kéo doc OOS vào top-k? reranker nâng điểm nhầm? confidence signal shift? → ghi, để implementation adoption biết cần siết gì.
- Không sửa code ở đây (evidence-only) — chỉ root cause + khuyến nghị.

## 6. Deliverable + phạm vi

| Thứ | Chi tiết |
|---|---|
| Writeup | `docs/notes/aiteamvn-adoption-go-no-go.md` — 4 kết quả + **1 khuyến nghị GO/NO-GO** (kèm điều kiện: mixed nếu gate #3 vừa, else dense-only) |
| Reports | `data/eval_reports/` (convention sẵn); RAM/latency report riêng |
| Code | `scripts/measure_stack_memory.py` (mới) + test nhẹ; reuse eval/calibrate/bench_reranker |

**Ngoài phạm vi (chốt)**: rewire production embedder sang MixedHybridEmbedder; `DEFAULT_COLLECTION_NAME` flip; đổi hằng `REFUSAL_THRESHOLD`; deploy config — **implementation riêng CHỈ KHI GO**. Xóa collection `luat_law_ctx_v4_aiteamvn` (giữ tới khi có quyết định adoption).

## 7. Testing

- `measure_stack_memory.py`: unit test nhẹ (shape của report / parse args), **không nạp model thật**.
- Không đụng code production path → suite hiện tại (385) giữ xanh.

## 8. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Groq TPD khi recalibrate 3 bộ | `--retry` merge; general+sectors đã có (general 1 err cần retry) |
| RAM local ≠ HF Spaces | Ghi caveat; đo xu hướng + biên, không tuyệt đối |
| Mixed không vừa 16GB | Fallback dense-only 0.898 (>v4 hybrid) — vẫn adopt được, chỉ khác hình dạng → NO-GO cho mixed ≠ NO-GO cho AITeamVN |
| Ngưỡng mới lệch nhiều khỏi 0.700 | Chỉ báo cáo (evidence-only); áp dụng ở implementation adoption |

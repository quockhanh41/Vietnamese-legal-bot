# M3.6 Phase 2 — Verbatim-span results (đo lường + B0 guard)

> **Date**: 2026-07-18 · **Milestone**: M3.6 Phase 2 · **Verdict**: **GO — no M3.6 regression**;
> verbatim quoting faithful (~95%). Full-pool false_refusal + sectors citation **deferred** (Groq budget cạn giữa sectors).
> **Design**: [`../superpowers/specs/2026-07-18-m3.6-verbatim-span-quotes-design.md`](../superpowers/specs/2026-07-18-m3.6-verbatim-span-quotes-design.md) §7 · **Plan**: `../superpowers/plans/2026-07-18-m3.6-verbatim-span-quotes.md` (Phase 2)

## Setup

- **Stack = production (AITeamVN mixed)**: collection `luat_law_ctx_v4_aiteamvn`, embedder `mixed`
  (AITeamVN/Vietnamese_Embedding dense + BGE-M3 sparse), reranker `AITeamVN/Vietnamese_Reranker`,
  `REFUSAL_THRESHOLD=0.700`. Khớp `src/api/main.py` từng dòng.
  ⚠️ eval defaults là `bge-m3`/`bge-reranker-v2-m3` — **PHẢI** truyền `--embedder mixed --reranker
  AITeamVN/Vietnamese_Reranker`, nếu không sẽ query embedding-space sai → retrieval rác (bẫy đã né).
- **Sets**: criminal (41: 34 answer + 7 refuse), general (41: 27 answer + 14 refuse),
  sectors (`eval_set_sectors_v2.json`, **83** answer — không phải v1/51).
- **Command**:
  `.venv/bin/python scripts/eval.py --set <s> --answers --embedder mixed --reranker AITeamVN/Vietnamese_Reranker --sleep 2`
- **Instrumentation (Phase 2)**: thêm 4 field observability lên `RAGResponse`
  (`verbatim_match_ratio`, `verbatim_n_spans`, `verbatim_unmatched`, `bullets_missing_span`),
  set trong `orchestrator._enrich_response` từ `ver_result`, ghi vào row Tier-2 của `eval.py`.
  OBSERVABILITY-ONLY — không đụng confidence/API ChatResponse. Smoke live crim-001: ratio 1.0, 3 spans.
- **Reports**: criminal `20260718T074313Z` (0 lỗi), general `20260718T075804Z` (0 lỗi),
  sectors `20260718T090723Z` (merged, **64/83 lỗi** — budget cạn).
  B0 baseline: criminal `20260716T174630Z` + general `20260716T041157Z` + sectors `20260716T030111Z`.

## Deliverable 1 — Verbatim faithfulness ("model trích nguyên văn trung thực đến đâu?")

Trên 69 câu answer được trả lời (criminal+general clean + 19 sectors sống-sót):

| chỉ số | giá trị |
|---|---|
| **per-span match rate** | **95.4%** (185/194 spans khớp nguồn) |
| **answers fully-verbatim** (ratio = 1.0) | **85.9%** (55/64) |
| **span compliance** (bullet mang span) | **92.8%** (64/69; 5 câu không span) |
| per-answer ratio | mean 0.952 · median **1.0** · min 0.50 |
| histogram ratio | 1.0: 55 · [0.75,1.0): 1 · [0.5,0.75): 8 |
| bullets_missing_span | tổng 7 (0.10/answer) |

**Kết luận**: model trên stack này trích nguyên văn rất trung thực — ~95% span khớp từng ký tự với
đúng khối [n] đã cite, đa số câu (86%) khớp 100%. Guardrail "BỎ dòng trích nếu không chép nguyên văn"
hoạt động: chỉ 7 bullet bỏ span thay vì bịa.

### Hallucinated-quote evidence — 9 span không khớp (giá trị cao nhất của milestone)

| id | span | phân loại |
|---|---|---|
| crim-104 | `"có tổ chức; có tính chất chuyên nghiệp"` | 2 điểm liệt kê ghép lại → không phải substring liền |
| crim-112 | `"c) Đối với tội phạm nghiêm trọng ... 02 tháng;"` | kèm marker `c)` + `;` cuối |
| crim-114 | `"bố trí chỗ nằm tối thiểu là 03 m2"` | `m2` vs `m²` — normalization gap (Minor #1) |
| crim-201 | `"Người từ đủ 14 tuổi trở lên"` | fragment ngắn, có thể từ khối không cite |
| crim-204 | `"có nơi cư trú rõ ràng"` | fragment ngắn |
| crim-304 | `"Không có thông tin chi tiết về nội dung sửa đổi"` | **BẮT THẬT: quote meta-text (amendment badge), KHÔNG phải luật** |
| gen-102 | `"12 ngày làm việc"` | fragment ngắn |
| gen-401 | `"Người thành niên là người từ đủ mười tám tuổi trở lên"` | có thể NFC/NFD hoặc "mười tám" vs "18" |
| gen-701 | `"đủ 20 năm trở lên"` | fragment ngắn |

Phần lớn là **normalization false-negative** (đơn vị `m²`, NFC/NFD, ranh giới điểm liệt kê) —
đúng các Minor findings đã defer từ Phase 1 review; xác nhận chúng gây một phần miss thật nhưng
ratio vẫn cao (95%). crim-304 là **catch thật**: model đóng ngoặc kép quanh chuỗi meta thay vì luật —
đúng loại lỗi milestone sinh ra để bắt. → nếu muốn ratio sạch hơn, thêm normalize `m²`/`...`/NFC
vào `_normalize_for_match` (chỉ giảm false-negative, không bao giờ gây false-match).

## Deliverable 2 — B0 regression guard

**Nguyên tắc**: câu refuse/OOS CHỈ có trong criminal+general (cả 2 chạy 0 lỗi) → 2/3 tiêu chí đo
DỨT KHOÁT trên data sạch. Chỉ false_refusal cần sectors (answer questions).

| tiêu chí | B0 | M3.6 run | verdict |
|---|---|---|---|
| refusal_recall | 16/21 = **0.7619** | 20/21 = **0.9524** | **PASS** (đo đủ, tốt hơn) |
| no new OOS leak | leaks {crim-402, crim-403, gen-907, gen-908, gen-912} | leak {gen-903} — net 5→1 | **PASS** (xem dưới) |
| false_refusal (criminal+general) | 4/61 = 0.066 | 6/61 = 0.098 | trong noise (xem dưới) |
| false_refusal (full-pool) | 14/144 = **0.0972** | **N/A — sectors 64/83 lỗi** | **DEFERRED** |
| citation criminal | 0.854 | 0.859 | ✓ |
| citation sectors | 0.952 | N/A (sectors lỗi) | **DEFERRED** |

### gen-903 KHÔNG phải leak mới của M3.6

gen-903 là **confident-OOS tồn tại từ M2** (`[[vlb-project-state]]` M2: conf 0.882 — chính là lý do
gate refusal=1.0 không đạt được, chọn 0.700 balanced). Run này gen-903 bounce 0.6621→0.7403 (noise,
vượt 0.700), trong khi CẢ 5 leak B0 tụt xuống refuse đúng (crim-402 0.73→0.42, crim-403 0.72→0.53,
gen-907 0.77→0.29, gen-908 0.74→0.52, gen-912 0.72→0.65). Net leaks **5→1** = churn do noise, không
phải regression.

### false_refusal chênh lệch = noise, KHÔNG phải M3.6 (bằng chứng quyết định)

Calibration invariant (Phase 1, đã verify) strip blockquote trước CẢ 4 signal → M3.6 không thể dịch
confidence một cách hệ thống. Nhưng prompt đổi ⇒ model viết bullet khác ⇒ vẫn phải đo. Delta
confidence per-question (mine − B0), 82 câu khớp criminal+general:

- **mean = −0.009** (≈ 0 — KHÔNG có dịch hệ thống), median +0.027, **stdev 0.123**
- **26 câu lên >0.05, 20 câu xuống <−0.05** (đối xứng), max +0.32 (crim-302), max −0.48 (gen-907)

Đây đúng nhiễu run-to-run ±0.1–0.2 đã ghi ở [[m3-7-confident-oos-null]] ("B0 không tái lập"). fr
chênh 6 vs 4 (criminal thực ra TỐT hơn: 1/34 vs 2/34 nhờ crim-302 hồi 0.63→0.95; general 5/27 vs 2/27
do gen-003/702/802 tụt <0.700 — đối trọng với các câu OOS cũng tụt). Không có cơ chế để M3.6 đẩy fr
lên hệ thống.

## Budget caveat & việc còn nợ

Groq/Gemini/OpenRouter free chain cạn CÙNG NGÀY sau 82 câu clean (criminal+general) — sectors (83
answer nặng) chỉ qua 19/83, retry thêm được 5. Xem [[groq-llm-budget-exhaustion]]. **Bài học: chạy
sectors TRƯỚC hoặc tách ngày** (set nặng nhất nên đi đầu khi budget tươi).

**Nợ (đo stack absolute, KHÔNG phải regression M3.6 — kết luận no-regression độc lập set):**
- sectors clean run → full-pool false_refusal (so 0.0972) + sectors citation (so 0.952).
- Lệnh: `.venv/bin/python scripts/eval.py --retry data/eval_reports/20260718T090723Z_eval.json --sleep 3`
  (khi budget reset; merge tiếp vào report).

## Verdict

**Phase 2 = GO (no M3.6 regression).** refusal_recall + no-new-leak đo dứt khoát PASS trên data sạch;
false_refusal không dịch hệ thống (Δmean≈0). Verbatim quoting trung thực ~95%. Full-pool fr + sectors
citation deferred tới ngày budget mới nhưng đo hiệu năng stack, không phải regression M3.6.

**M3.6 milestone**: Phase 1 (merged) + Phase 2 no-regression confirmed. Chốt "done" hoàn toàn khi
sectors clean run xác nhận full-pool fr trong gate (thủ tục, kỳ vọng = B0 ± noise).

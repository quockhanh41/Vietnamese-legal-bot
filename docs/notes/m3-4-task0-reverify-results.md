# M3.4 Task 0 — re-verify crim-203 reranker-on-comparison bug on AITeamVN stack

> Ngày đo: 2026-07-17 · Stack: production AITeamVN mixed (dense `AITeamVN/Vietnamese_Embedding`
> + sparse BGE-M3) + reranker `AITeamVN/Vietnamese_Reranker` + collection `luat_law_ctx_v4_aiteamvn`,
> `REFUSAL_THRESHOLD=0.700`. Đo bằng scratchpad harness (per-signal capture, N=3/câu để đo nhiễu
> run-to-run đã biết ±0.1–0.2). Đọc kèm [`crim-false-refusal-diagnosis-2026-07-07.md`](crim-false-refusal-diagnosis-2026-07-07.md)
> (bug gốc) và [`Plan_Giai_Doan_Cuoi_2026-07-10.md`](../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3 M3.4.

## Charter (Task 0, bắt buộc trước khi lên plan fix)

Bug crim-203 được đo trên reranker **CŨ** (`BAAI/bge-reranker-v2-m3`, corpus 6-doc, 2026-07-07):
câu so sánh cần tổng hợp ≥2 điều → cross-encoder chấm từng chunk độc lập → max single-chunk
relevance **0.20 giả tạo** dù retrieval đúng → confidence sụp dưới threshold → **từ chối oan**.
Production nay dùng reranker khác (`AITeamVN/Vietnamese_Reranker`). Phải đo lại bug còn tái hiện
không TRƯỚC khi thiết kế fix.

## Kết quả — bug KHÔNG tái hiện; reranker signal 0.20 → 0.88

**Cặp so sánh hình sự (charter gốc), N=3:**

| id | refused | score min/mean/max | reranker_signal (raw max) | cite |
|---|---|---|---|---|
| crim-203 (giết người BLHS 1999 vs 2015) | **0/3** | 0.777/0.781/0.790 | **0.883** (was 0.20 on bge) | 0.5 |
| crim-202 (trộm cắp vs lừa đảo) | **0/3** | 0.752/0.763/0.785 | **0.904** (was crashed on bge) | 1.0 |

Reranker signal **cố định** giữa các run (0.8832 cả 3 lần crim-203) — cross-encoder không stochastic;
nhiễu confidence còn lại đến từ self_eval/entailment, không từ reranker. Root cause gốc (bge chấm
comparison chunk 0.20) **đã biến mất**: AITeamVN reranker chấm đúng chunk so sánh 0.88–1.0.

→ **M3.4 (reranker-on-comparison) = NULL / không tái hiện.** Việc swap sang AITeamVN reranker
(Phase 0) đã sửa bug này "miễn phí". Hai hướng fix dự kiến (aggregate rerank score / decompose
để chống reranker under-score) **không còn premise** — reranker không phải chỗ hỏng.

## Nhưng: đo cả cohort so sánh (8 câu sector) lộ ra bug KHÁC, nặng hơn

Đo 8 câu `type=comparison` của sector set v2 (N=3) — 4/8 **từ chối 3/3**:

| id | refused | reranker | old-law article | new-law article | root cause |
|---|---|---|---|---|---|
| ds-201 (Căn cước 2014/2023) | 0/3 | 0.999 | ✓ có | ✓ có | — (answer, cite 1.0) |
| dn-201 (Đầu tư 2014/2025) | 0/3 | 0.936 | ✓ | ✓ | — (answer, cite 1.0) |
| hg-201 (HNGĐ 2000/2014) | 0/3 | 0.875 | — | ✓ | answer, cite 0.5 |
| dd-201 (Đất đai 2013/2024) | 1/3 | **0.775 (khỏe)** | ✗ **excluded** (2013) | ✓ Điều 176 | old-law superseded; refuse 1/3 do entail 0.0 sát ngưỡng |
| ld-201 (BHXH 2014/2024) | **3/3** | 0.043 | ✗ **excluded** (2014, Điều 54) | ✓ Điều 64 (raw yếu) | old-law superseded + new-law rerank yếu |
| ds-405 (Cư trú VBHN 2024/2025) | **3/3** | 0.0004 | 1 trong 2 Điều 23 missing | | retrieval miss (niche consolidated) |
| hg-406 (bạo lực GĐ NĐ/TT) | **3/3** | 0.243 | Điều 12 có (raw yếu) | Điều 10 (669837) missing | retrieval/rerank yếu (cả 2 doc is_in_effect=True) |
| hs-403 (án treo VBHN 2022/2025) | **3/3** | 0.0 | **pool RỖNG — 0 chunk** | | retrieval fail hoàn toàn (cả 2 doc is_in_effect=True) |

### Root cause thật (đã verify), KHÔNG phải reranker-on-comparison

Mọi retrieval path gọi `search_hybrid(..., exclude_superseded=True)` (`orchestrator.py:568`) **cộng**
post-filter `_filter_in_effect` (drop `is_in_effect=False`). Câu so sánh "Luật X (cũ) vs Luật Y (mới)"
**bản chất cần bản luật bị thay thế**, nhưng filter loại đúng bản đó. Verify Qdrant:

| doc | | is_in_effect |
|---|---|---|
| 46056 BLHS 1999 | crim-203 | **False** → excluded |
| 259700 BHXH 2014 (Điều 54) | ld-201 | **False** → excluded |
| 215836 Đất đai 2013 (Điều 129) | dd-201 | **False** → excluded |

**Hai lớp lỗi tách bạch:**
1. **Supersession-exclusion** (crim-203 cite 0.5, ld-201, dd-201): bản luật cũ bị `is_in_effect=False`
   → filter loại → câu so sánh chỉ thấy 1 vế. Nếu vế mới rerank khỏe → vẫn answer nhưng half-cite
   (crim-203, dd-201); nếu vế mới cũng yếu → refuse (ld-201).
2. **Retrieval/rerank yếu trên doc niche** (ds-405, hg-406, hs-403): cả 2 doc `is_in_effect=True`
   (không phải supersession) nhưng embedder+reranker không kéo được chunk đúng cho câu so sánh
   văn bản hợp nhất / thông tư hướng dẫn rất hẹp (hs-403: pool rỗng hoàn toàn). Reranker chấm thấp
   là **đúng** (pool xấu), không phải bug reranker.

Ở mọi câu ANSWER được (crim-203, ds-201, dn-201, hg-201, dd-201-đa số), khi chunk đúng có mặt
**AITeamVN reranker chấm nó 0.88–1.0** — reranker xử lý câu so sánh tốt. Điểm thấp chỉ xuất hiện
khi pool thực sự thiếu chunk đúng (supersession loại, hoặc retrieval miss).

## Verdict

- **M3.4 như đã scope (reranker-on-comparison fix) → đóng NULL.** Bug crim-203 không tái hiện;
  reranker swap (Phase 0) đã trị. Hai hướng fix cũ (aggregate score / decompose để cứu reranker)
  không có premise.
- **Phát hiện mới, giá trị hơn (do đo cả cohort — quyết định "measure sector first" đúng):**
  câu so sánh **không truy hồi được bản luật cũ (superseded)** → 4/8 câu sector refuse, phần còn lại
  half-cite. Đây là lỗi **temporal-filter/retrieval**, thuộc vùng M2 (VBHN/supersession) + M3.5
  (retrieval), KHÔNG thuộc scope reranker của M3.4. Hướng khả dĩ: với câu `type=comparison`,
  **tắt `exclude_superseded`** (hoặc retrieve tường minh cả 2 bản version) — trùng tinh thần
  "decompose per law version" của M3.4 nếu sub-query bản cũ cũng disable supersession. Chạm filter
  supersession = vùng nhạy (bug VBHN M2), phải brainstorm + recalibrate riêng, không vá vội.

## Reproduce

- `scratchpad/m34_task0_reverify.py` (per-signal, N runs, production stack); logs
  `m34_task0.log` (criminal) + `m34_task0_sectors.log` (sectors); dump JSON
  `m34_task0_results.json` + `m34_task0_sectors.json`.
- `scratchpad/check_superseded.py` — Qdrant scroll xác nhận `is_in_effect` các doc cũ.
- Không sửa code production; chỉ đo (đúng nguyên tắc 2: đo trước, không cải thiện thì bỏ có ghi chép).

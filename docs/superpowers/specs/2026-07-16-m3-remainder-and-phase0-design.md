# M3 remainder — umbrella design + Phase 0 (AITeamVN adoption)

> Ngày lập: 2026-07-16 · Trạng thái: sống — cập nhật sau mỗi workstream.
> Đọc kèm: [`../../Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3 (bảng M3),
> [`../../notes/m3.3-m3.7-rationale-and-approach.md`](../../notes/m3.3-m3.7-rationale-and-approach.md) (lý do tồn tại + hướng giải từng mục),
> [`../../notes/aiteamvn-adoption-go-no-go.md`](../../notes/aiteamvn-adoption-go-no-go.md) (verdict GO cho stack AITeamVN).

Đây là **umbrella** cho 5 mục còn mở của M3 (3.7 → 3.3 → 3.4 → 3.5 → 3.6) cộng một **Phase 0** tiền đề: đưa stack AITeamVN vào production. Nó KHÔNG lặp lại phần "vì sao từng mục tồn tại" (đã có trong rationale note) — nó chốt **thứ tự**, **hợp đồng đo lường (gate)**, **quy trình thực thi**, và **thiết kế chi tiết Phase 0**. Các mục 3.7–3.6 được spec chi tiết **just-in-time** (JIT): mỗi mục viết sau khi mục trước đã land + đo được số thật.

---

## 0. Ba quyết định khung (chốt với user 2026-07-16)

| # | Quyết định | Lựa chọn | Hệ quả |
|---|---|---|---|
| A | **Stack baseline** cho 5 mục | Adopt **AITeamVN mixed** (dense `AITeamVN/Vietnamese_Embedding` + sparse `bge-m3` + reranker `AITeamVN/Vietnamese_Reranker`) TRƯỚC, rồi làm 5 mục trên stack mới | Baseline confident-OOS chuyển từ gen-903 (v4) sang gen-907/912 (AITeamVN) |
| B | **Adoption vs 3.7** | **Tách**: Phase 0 adopt riêng → 3.7 nhắm VƯỢT baseline đó | Tách biến sạch (nguyên tắc 2); writeup: "recalibrate được X, scope-check tốt hơn" |
| C | **Độ chi tiết tài liệu** | **Umbrella + JIT**: doc này chốt khung; spec chi tiết từng workstream viết khi tới lượt | Baseline mục sau dựa trên số **đã commit** của mục trước, không đoán |

**Nguyên tắc kế thừa (không thương lượng)** — trích Plan §0:
1. Thà từ chối còn hơn trả lời sai — giữ 2 cổng từ chối (scope gate + confidence gate) nguyên vẹn.
2. Đo trước, tối ưu sau — mỗi kỹ thuật chỉ merge khi có số trước/sau; không cải thiện → bỏ có ghi chép.
4. Threshold là hàm của phân bố — mọi thay đổi context/candidate-pool/signal → re-run Tier-2 + `calibrate_threshold.py` trước khi tin threshold.

---

## 1. Thứ tự & phụ thuộc

```
Phase 0  ADOPT AITeamVN → production          [tiền đề, không phải 1 trong 5 mục]
   │  rewire embedder+reranker (main.py) + DEFAULT_COLLECTION_NAME → v4_aiteamvn
   │  + criminal Tier-2 ĐẦY ĐỦ + recalibrate → chốt threshold t*
   │  ⇒ ARTEFACT B0 = baseline production mới {t*, refusal_recall r0, false_refusal f0, citation c0}
   ▼
3.7  Confident-OOS ceiling (gen-903/907/912)   [khó & rủi ro nhất → làm đầu; mục tiêu VƯỢT r0]
   ▼
3.3  Corrective RAG nhẹ (rewrite-retry trước refuse)   [giảm false-refuse biên]
   ▼
3.4  Reranker-on-comparison fix (crim-203)            [aggregate score / decompose]
   ▼
3.5  Cross-reference graph vào route complex          [wire expand_context có sẵn]
   ▼
3.6  Verbatim-span quotes (verify string-match)       [lớp verify bổ sung]
   │
   └─▶ (tuỳ chọn) đo lại confident-OOS ceiling một lần nữa — 3.3/3.4/3.6 đổi cách đo
        "context có trả lời được không" nên có thể cải thiện phân biệt OOS gián tiếp.
```

**Vì sao thứ tự này** (chi tiết ở rationale note): 3.7 sửa cơ chế refuse gốc nên làm đầu để các mục sau đo trên nền confidence ổn định; không stack 2 thay đổi lớn cùng lúc để tách được nguyên nhân. 3.5 (graph) không đụng refuse gate nên rủi ro thấp; 3.6 (verify) hưởng lợi từ context đã ổn định ở các bước trên.

---

## 2. Gate & measurement contract (dùng chung mọi workstream)

### 2.1. Công cụ (đã có, không xây mới)

| Instrument | Đo | LLM? | Ai chạy |
|---|---|---|---|
| `scripts/eval.py` (Tier-1) | recall@5/@10 + MRR (strict), per-sector | ❌ | inline / sub-agent tự do |
| `scripts/eval.py --answers` (Tier-2) | refusal_recall, false_refusal, citation_acc + dump confidence distribution | ✅ | **chỉ tại gate** |
| `scripts/calibrate_threshold.py` | sweep threshold trên report Tier-2 (Youden + bảng) | ❌ | tại gate |

Eval set: `criminal_v1` (41 câu, 7 refuse: crim-401..407) · `general_v1` (41 câu, 14 refuse) · `sectors_v2` (83 câu, 0 refuse).
**OOS set = 21 câu** (14 general + 7 criminal). Reports → `data/eval_reports/<ts>_eval.json`.

### 2.2. Mục nào đo bằng gì

- **Chạm retrieval** (Phase 0 retrieval, 3.4, 3.5): Tier-1 recall + per-sector — rẻ, no-LLM.
- **Chạm refuse/answer** (Phase 0 threshold, 3.7, 3.3): Tier-2 trên OOS 21 + `calibrate_threshold` + phân bố confidence — tốn LLM.
- **Chạm verify** (3.6): Tier-2 citation_acc + verbatim hit-rate + so false-negative với NLI — tốn LLM.

### 2.3. Gate re-anchor (quan trọng)

Con số "0.765 / 0.136" trong Plan §3 là của stack **v4/gen-903 cũ**. Trên AITeamVN nó khác. **Phase 0 chốt B0 = {t\*, r0, f0, c0} trên data đầy đủ** và mọi mục sau tính theo B0:

- **3.3 / 3.4 / 3.5 / 3.6**: giữ `refusal_recall ≥ r0` ∧ `false_refusal ≤ f0` ∧ **không leak OOS mới** ∧ `citation ≥ c0 − nhiễu`; VÀ mỗi mục cải thiện metric mục tiêu riêng (3.3 giảm false-refuse; 3.4 tăng recall câu so sánh; 3.5 tăng multi-hop recall; 3.6 thêm verbatim hit-rate).
- **3.7**: `refusal_recall > r0` (chặt hơn — mục tiêu tăng, không chỉ giữ) ∧ `false_refusal ≤ f0`.
- **Kết quả null hợp lệ** (đặc biệt 3.7): ghi writeup, rollback, không ép merge.

### 2.4. Kỷ luật threshold & quota

- **Threshold** (nguyên tắc 4): mọi mục đổi candidate-pool/context/signal (3.3, 3.4, 3.5) → re-run Tier-2 + recalibrate trước khi tin threshold. Không mặc định giữ nguyên.
- **Quota LLM**: không có key trả phí. Free chain (Groq+Gemini+OpenRouter) có thể cạn cùng ngày (memory `groq-llm-budget-exhaustion` — đó là quota **ngày**, retry không cứu). → Tier-2 **chỉ chạy tại gate**, một phiên khi quota tươi (thường sau khi reset sang ngày mới), dùng `eval.py --retry <report.json>` để resume run dở, gộp các set trong một phiên.

---

## 3. Quy trình thực thi

**Chế độ**: inline execution (Claude tự làm trong session; giữ kỷ luật TDD + verification-before-completion). Có thể chuyển một mục sang sub-agent nếu đủ lớn/độc lập (ví dụ 3.7 experiment) — quyết định per-workstream.

**Giao tiếp**: brief cho sub-agent (nếu dùng) = **tiếng Anh**; trao đổi/tóm tắt với user = **tiếng Việt**.

**Branch**: mỗi workstream 1 branch (`m3-phase0-adopt`, `m3.7-confident-oos`, …) → merge `main` sau khi gate pass + user duyệt. Giá trị production (`DEFAULT_COLLECTION_NAME`, `REFUSAL_THRESHOLD`, wiring `main.py`) chỉ đổi trong branch.

**Điểm dừng có người duyệt**: (a) trước merge mỗi workstream; (b) trước khi đổi giá trị production; (c) khi một mục cho kết quả null.

**TDD**: mọi task đụng code → test đỏ→xanh, `make test` xanh trước khi claim xong. Task thuần-đo (re-run Tier-2) → verify = report JSON + bảng số.

---

## 4. Phase 0 — thiết kế chi tiết (adopt AITeamVN → production)

**Tiền đề đã verify (2026-07-16)**: collection `luat_law_ctx_v4_aiteamvn` **còn sống** trên Qdrant Cloud (60,958 pts, `is_in_effect` đã backfill) → **không cần re-embed Kaggle**. Plumbing (`src/data/mixed_embedder.py`, cờ `--embedder mixed`/`--reranker`) đã ở `main` (`b06199b`). Interface `MixedHybridEmbedder` (`encode_queries`, `encode(return_dense, return_sparse)`) khớp `BGEEmbedder` → orchestrator không đổi.

### 4.1. Bốn điểm sửa production (branch `m3-phase0-adopt`)

| # | File | Đổi |
|---|---|---|
| 1 | `src/api/main.py` `get_embedder()` | `BGEEmbedder()` → `build_embedder("mixed", "AITeamVN/Vietnamese_Embedding")` |
| 2 | `src/api/main.py` `get_adaptive_pipeline()` | thêm `reranker_model="AITeamVN/Vietnamese_Reranker"` |
| 3 | `src/data/constants.py` | `DEFAULT_COLLECTION_NAME` → `"luat_law_ctx_v4_aiteamvn"` (+ cập nhật comment lịch sử) |
| 4 | `src/rag/confidence.py` | `REFUSAL_THRESHOLD 0.700` → **t\*** (từ recalibrate 4.3) |

Ghi chú #1: `build_embedder` mặc định `device="cpu"` (đúng cho HF Spaces production). Mixed nạp 2 model (AITeamVN dense + bge-m3 sparse) + reranker AITeamVN — peak RSS đo được 2989MB/16GB (Gate #3 GO/NO-GO, fits-headroom 18.2%).

### 4.2. Trình tự

- **0.1 — criminal Tier-2 đầy đủ** (QUOTA-GATED, làm đầu khi quota tươi): `eval.py --retry data/eval_reports/20260716T050956Z_eval.json` (hoặc chạy mới `--set criminal --answers --collection luat_law_ctx_v4_aiteamvn --embedder mixed --reranker AITeamVN/Vietnamese_Reranker --sleep 3`) → n=41, errors=0. *(general `…041157Z` + sectors `…030111Z` đã clean trên đúng stack này → tái dùng, chỉ criminal cần bù 12 câu.)*
- **0.2 — recalibrate** (no-LLM): pool general + sectors + criminal đầy đủ → `calibrate_threshold.py`. **Chốt t\*** — ứng viên 0.725 (r≈0.81 / f≈0.146) vs 0.750 (r≈0.94 / f≈0.197) vs trung gian; tiêu chí = cân bằng refusal_recall/false_refusal (như M2 chọn balanced 0.700 thay Youden). **User chốt t\* tại gate này.** Đọc ra B0 = {t\*, r0, f0, c0} trên OOS 21.
- **0.3 — rewire** (code, TDD): 4 điểm 4.1 + no-LLM test (assert get_embedder ra mixed, reranker_model đúng, collection đúng) + Tier-1 retrieval smoke trên collection mới.
- **0.4 — verify** (QUOTA-GATED, drive thật): chạy pipeline adopted trên vài câu thật (1 answer / 1 refuse / 1 citation-lookup), quan sát hành vi — không chỉ nhìn số.
- **0.5 — close-out**: cập nhật `aiteamvn-adoption-go-no-go.md` → "adopted", cập nhật `scale-journey` + Plan §7 status; **user duyệt → merge `main`.**

### 4.3. Tradeoff cốt lõi (nói rõ trong writeup)

AITeamVN mua **retrieval + citation tốt hơn** (citation +2.6pt sectors; recall@5 tăng mọi mode; criminal routed 0.8824 vs 0.8382) NHƯNG tại threshold recalibrate nó **đẩy false_refusal LÊN** (0.146 @ 0.725 hoặc 0.197 @ 0.750, cao hơn floor cũ 0.136) — vì embedding mạnh kéo chunk "đúng chủ đề, sai luật" vào context cho câu OOS-bẫy (gen-907/912). Nghĩa là **f0 của B0 sẽ cao hơn 0.136** — đây chính là **"nợ" mà 3.7 (scope-check tận gốc) + 3.3 (corrective retry) sinh ra để trả lại.** Câu chuyện: Phase 0 mua chất lượng + gánh nợ false-refuse → 3.7/3.3 trả nợ.

### 4.4. Gate Phase 0

Tier-2 hai set (criminal đầy đủ + general) trên stack adopted tại t\*: B0 = {t\*, r0, f0, c0} ghi rõ; criminal Tier-1 routed recall@5 ≥ v4 (đã có 0.8824 vs 0.8382); no-LLM test + Tier-1 smoke pass; `make test` xanh; drive thật 3 câu quan sát đúng hành vi. **Artefact**: bảng B0 + note adoption cập nhật.

---

## 5. 3.7 – 3.6 (spec JIT — chỉ pointer ở đây)

Mỗi mục sẽ có spec riêng khi tới lượt; hướng giải + cơ chế đo đã có trong [rationale note](../../notes/m3.3-m3.7-rationale-and-approach.md):

- **3.7 Confident-OOS ceiling** — scope-check tách khỏi confidence gate; mục tiêu `refusal_recall > r0`. Rủi ro cao nhất (đụng REFUSAL_THRESHOLD/scope gate) → recalibrate + full Tier-2 hai set.
- **3.3 Corrective RAG** — rewrite-retry 1 lần khi rerank score thấp, trước refuse; tái dùng `QUERY_REWRITE_PROMPT`; grid-search ngưỡng "score thấp" trên eval set.
- **3.4 Reranker-on-comparison** — aggregate rerank score theo document HOẶC decompose câu so sánh (2 hướng, thử-đo cả hai); regression check toàn Tier-2.
- **3.5 Cross-reference graph** — wire `expand_context=True` vào `_complex_pipeline`; audit chất lượng graph trước khi bật mặc định.
- **3.6 Verbatim-span quotes** — model quote nguyên văn kèm `[n]`, verify string-match; mặc định observability-only trước (như grounding_ratio lần đầu).

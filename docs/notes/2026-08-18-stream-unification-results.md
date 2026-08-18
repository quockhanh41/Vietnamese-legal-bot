# Hợp nhất stream/non-stream — kết quả

**Ngày:** 2026-08-18
**Branch:** `stream-nonstream-unification` (12 commit, `bef618e`..`13c8f83`)
**Spec:** [`../superpowers/specs/2026-08-18-stream-nonstream-unification-design.md`](../superpowers/specs/2026-08-18-stream-nonstream-unification-design.md)
**Plan:** [`../superpowers/plans/2026-08-18-stream-nonstream-unification.md`](../superpowers/plans/2026-08-18-stream-nonstream-unification.md)

---

## 1. Trạng thái

| Phần | Trạng thái |
|---|---|
| Task 1–6 (code + test) | ✅ Xong, mỗi task qua review riêng |
| Task 7 (đo lại Tier-2) | ⏸️ **TREO — chờ quyết định về hạn mức LLM** |

Suite: **533 → 549** test, tất cả xanh.

## 2. Vì sao phải làm

Phát hiện nền của đợt audit 2026-08-18:

> **100% số đo của dự án đến từ `query()`** (`scripts/eval.py:364`, docstring `:15` ghi rõ *"non-streaming path"*).
> **100% lưu lượng người dùng đi qua `query_stream()`** (`src/api/main.py`, `src/ui/app_gradio.py`).

Hai tập hợp rời nhau ⇒ mọi chỗ hai đường lệch nhau đều **vô hình với mọi gate** dự án có. Đó không phải rủi ro giả định: `RERANK_MAX_K` (knob hạ P95 reranker trên CPU, thêm ở M5) đọc ở đúng **1 trong 4** điểm gọi, và không phải điểm nào trên đường streaming.

## 3. Đã sửa

| # | Việc | Commit |
|---|---|---|
| 6 | Hợp nhất routing + retrieval vào `_route`/`_retrieve` dùng chung (PEP 380: `yield from` cho stream, `_drain` cho non-stream) | `3bfcd73`, `ccc9733` |
| 3 | `RERANK_MAX_K` áp cho route medium trên **cả hai** đường | `ccc9733` |
| 2 | `generate_stream` **raise** thay vì yield chuỗi lỗi — trước đây chuỗi lỗi bị ghi vào lịch sử hội thoại và **cache lại để phát cho người sau** | `78d421a` |
| — | `StreamInterrupted` riêng cho đứt-giữa-chừng (không retry được, không cache) | `4a89d62` |
| 1 | Adaptive thành pipeline mặc định (API + Streamlit) | `d4bd791` |
| 7 | Bỏ `_build_context`/`_build_citations` gọi lặp; `verify_context` thành quyết định có tên | `ccc9733` |
| 8 | `_post_process` nhận citations đã annotate | `ccc9733` |

## 4. Diện tích thay đổi hành vi

Spec §4 khẳng định hợp nhất theo chuẩn non-stream chỉ đổi **đúng một** thứ quan sát được. Final review (Opus) đã **cố bác bỏ** khẳng định này bằng cách trace song song orchestrator cũ/mới trên cả hai đường, và **không tìm được thay đổi thứ hai**.

Thay đổi duy nhất:

> Route **complex**: `_post_process` nhận context **chưa nén** thay vì đã nén.

Lý do giữ hành vi non-stream: context nén là output LLM, lossy — verify trên đó có thể báo oan "không có căn cứ" khi chính bước nén đã cắt mất câu chống lưng. Nay là quyết định có tên (`verify_context` trong `_prepare_generation`), không còn là tác dụng phụ của một dòng `_build_context` thừa.

## 5. ⏸️ Phần còn treo: đo lại Tier-2 route complex

**Chưa chạy.** Không phải vì hết quota — mà vì chạy nó tiêu hạn mức LLM mà chủ dự án nên tự chọn thời điểm: 2 lần chạy × 41 câu × full adaptive pipeline ≈ vài trăm LLM call, trên chuỗi free có thể cạn trong cùng một ngày (xem `docs/notes/` về budget exhaustion).

Khi chạy:

```bash
.venv/bin/python scripts/eval.py --set criminal --answers --limit 41
```

**Gate (B0):** `refusal_recall ≥ 0.7619`, `false_refusal ≤ 0.0972`

⚠️ **Chạy ít nhất 2 lần và lấy trung bình.** Confidence nhiễu ±0.1–0.2 giữa các lần chạy (phát hiện ở M3.7) — một lần chạy không kết luận được gì.

Điền số vào bảng này khi có:

| Lần chạy | refusal_recall | false_refusal | Report |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| Trung bình | | | |

## 6. Việc kéo theo (KHÔNG thuộc branch này)

- **`RERANK_MAX_K` không cắt được route complex** (`max_k=20` cứng ở cả hai đường, `orchestrator.py:557`). Cố ý để nguyên: đổi 20→15 sẽ dịch retrieval ở cấu hình mặc định và làm dịch gate B0 — mà lưới parity **không bắt được**, vì cả hai đường cùng đổi. Cần task riêng có đo riêng. Đã có comment tại chỗ ghi lý do.
- `_post_process` trước đây cho phép truyền `citations` theo vị trí → đã chặn bằng `*,` (`orchestrator.py:718`).
- `docs/ARCHITECTURE.md` §2 có một câu diễn đạt lỏng về quan hệ `_route`/`_retrieve`/`_run_route` — đúng sự thật, chỉ hơi dễ đọc nhầm.

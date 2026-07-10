# Chẩn đoán: chunker metadata misalignment (M0.4b — blocker ingest #3)

> Ngày: 2026-07-10 · Người đo: Fable (trực tiếp, read-only trên Qdrant Cloud)
> Script: scroll toàn bộ `luat_law_ctx` (1.670 points), đối chiếu nhãn payload `article`
> với header "Điều N." xuất hiện trong `chunk_text`.

## Số đo trên collection production `luat_law_ctx`

| Phân loại | Số điểm | % | Ý nghĩa |
|---|---|---|---|
| HEADER_OK | 193 | 11.6% | chunk_text mở đầu đúng header "Điều N." khớp nhãn |
| MID_ARTICLE | 1.346 | 80.6% | không có header Điều ở đầu — **phần lớn by-design** (chunk mức Khoản; đây chính là hiện tượng "lenient=0" ghi nhận Tuần 2, không phải bug) |
| **WRONG_LABEL** | **126** | **7.5%** | **bug thật**: text chứa header "Điều M" với M ≠ nhãn N (thường M = N−1, N−2) |
| NO_ARTICLE_LABEL | 5 | 0.3% | chunk preamble của 3 luật sửa đổi — vô hại |

Phân bố WRONG_LABEL theo document: BLHS 2015 (32), BLTTHS 2015 (55), BLHS 1999 (39) —
**toàn bộ ở 3 bộ luật dùng `HierarchicalChunker`; 3 luật sửa đổi (ArticleLevelChunker) sạch 100%.**

Ví dụ điển hình: nhãn `Điều 21` nhưng text = "3. Người bào chữa..." (đuôi Khoản 3 Điều 19)
+ toàn bộ Điều 20; nhãn `Điều 3` (BLHS 1999) nhưng text bắt đầu từ "Điều 1. Nhiệm vụ...".

## Root cause (đã xác nhận trong code)

`src/data/chunking/hierarchical.py`, hàm `_control_chunk_sizes` (dòng 185-232):

1. **Merge xuyên ranh giới Điều** (dòng 201-206): các structural chunk nhỏ liên tiếp được
   gộp đến khi đạt `max_chunk_size` mà KHÔNG kiểm tra chúng có cùng `article` không.
   Dòng 204 `buffer["article"] = chunk.get("article") or buffer.get("article")` — nhãn của
   mảnh SAU đè lên nhãn buffer → chunk gộp "đuôi Điều 19 + Điều 20 + đầu Điều 21" mang
   nhãn "Điều 21". Khớp chính xác pattern N−1/N−2 quan sát được (các Điều ngắn liên tiếp
   bị gộp: Điều 1+2+3 BLHS 1999 → nhãn Điều 3).
2. **Split oversize kế thừa metadata sai** (dòng 221-230): chunk > max size bị cắt bằng
   `RecursiveCharacterTextSplitter`, mọi sub-chunk copy nguyên metadata — nếu chunk gốc đã
   span nhiều Điều (do bug 1) thì mọi sub-chunk cùng mang 1 nhãn sai. Hết bug 1 thì bug 2
   tự lành phần article, nhưng `clause` của sub-chunk sau vẫn có thể lệch (chấp nhận được).

### 2 bug tiềm ẩn phát hiện kèm (cùng file)

3. **`flush_buffer` vứt text ngắn** (dòng 110): `len(text) >= min_chunk_size // 2` (100 ký
   tự) — Điều trọn vẹn nhưng rất ngắn có thể bị DROP khỏi index (mất hẳn điều luật).
   Khi sửa bug 1 (không merge xuyên Điều nữa), các Điều ngắn đứng riêng sẽ rớt vào đúng
   nhánh này → PHẢI xử lý cùng lúc: Điều trọn vẹn thì giữ bất kể độ dài.
4. **`parent_text` chụp tại thời điểm flush** (dòng 116): Khoản 1 của Điều N flush khi gặp
   header Khoản 2 — lúc đó `article_buffer` mới chứa header + Khoản 1 → parent_chunk_text
   của Khoản đầu là **Điều cụt** (thiếu các Khoản sau). Ảnh hưởng trực tiếp parent-document
   retrieval (§3.4): block "Điều cha" có thể thiếu nội dung tùy chunk nào được retrieve.
   Fix đúng: two-pass — sau khi scan xong document, gán lại parent_text = full text Điều
   cho MỌI chunk thuộc Điều đó.

## Gate đo sau khi fix

- Chạy lại script chẩn đoán (logic: chunk 6 docs local bằng chunker đã sửa, không cần
  Qdrant/embedding): **WRONG_LABEL = 0**, không Điều nào trong văn bản gốc vắng mặt khỏi
  output chunker (đếm số Điều distinct trong content vs trong chunks).
- Sau re-ingest (M0.5): re-run scroll diagnosis trên collection v2 → WRONG_LABEL = 0;
  Tier-1 strict-lenient gap ≈ 0 theo gate M0 của plan.

## Tác động hiện tại của bug (trước fix)

- 7.5% chunk trỏ citation về sai Điều → citation card sai, strict eval matching thưởng/phạt
  nhầm payload, verifier structural check so "Điều N trong answer vs context" trên nhãn sai.
- Parent-doc block có thể thiếu Khoản (bug 4) → context đưa LLM không đầy đủ như tưởng.

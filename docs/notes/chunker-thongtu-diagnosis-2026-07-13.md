# Chẩn đoán: HierarchicalChunker bỏ sót Điều có hậu tố chữ cái trên corpus 600 (M2.7 — Task 6/C1)

> Ngày: 2026-07-13 · Người đo: sub-agent (Sonnet, task-6) · offline, không đụng Qdrant
> Script: `scripts/verify_chunker_alignment.py --data-path data/raw/corpus_600docs.parquet --quiet`

## Số đo trước fix (605 docs, corpus_600docs.parquet)

| Phân loại | Số chunk |
|---|---|
| total chunks | 60.917 |
| HEADER_OK | 21.630 |
| MID_ARTICLE | 34.751 |
| **WRONG_LABEL** | **158** |
| NO_ARTICLE_LABEL | 4.378 |

GATE: `WRONG_LABEL == 0` → **FAIL**. GATE: `no Điều dropped (HierarchicalChunker)` → **FAIL**
(nhiều Điều `MISSING from output`). Per-sector chunk ratio (không liên quan bug này):
max/min = 1.43 — đã OK (<=3x), khớp ước tính Phase B (1.39).

Toàn bộ 63 document có finding đều là **"Văn bản hợp nhất"** (văn bản hợp nhất luật gốc +
các nghị định/thông tư sửa đổi) — không phải các luật/thông tư "sửa đổi, bổ sung" thường thấy
ở corpus 100 (những văn bản đó được route sang `ArticleLevelChunker`, sạch 100% ở gate này).

## Root cause

`src/data/chunking/hierarchical.py`, `HierarchicalChunker.ARTICLE_PATTERN`:

```python
ARTICLE_PATTERN = re.compile(r'^\s*Điều\s+(\d+)\s*[\.:\)]\s*(.*)', re.IGNORECASE)
```

Chỉ khớp số thuần (`\d+`). "Văn bản hợp nhất" chèn các Điều bổ sung ngay tại chỗ trong luồng
Điều gốc bằng số có hậu tố chữ cái một ký tự — ví dụ Điều 8, 8a, 9, 10, 11, 11a, 12, 12a, ...
(hậu tố = Điều được "bổ sung" bởi văn bản sửa đổi, giữ nguyên vị trí khi hợp nhất). Dòng
`Điều 8a. ...` không khớp `ARTICLE_PATTERN` → không được nhận diện là ranh giới Điều mới →
nội dung của nó bị gộp thẳng vào buffer của Điều 8 (Điều liền trước), tạo ra đúng lớp lỗi
WRONG_LABEL đã chẩn đoán ở M0.4b (`chunker-misalignment-diagnosis-2026-07-10.md`): chunk mang
nhãn "Điều 8" nhưng text chứa header "Điều 8a" — và Điều 8a không bao giờ có nhãn riêng, nên
bị đếm là "dropped" theo phép đối chiếu distinct-Điều-in-content vs distinct-Điều-in-chunks.

Đối chiếu 100% các mục `MISSING from output` trong log gate: toàn bộ đều có dạng `\d+[a-z]`
(8a, 11a, 12a, 17a, 19a, 27a, 3a, 28b, 41b, ... — không có mục nào khác dạng). Xác nhận đây
là nguyên nhân duy nhất.

`ArticleLevelChunker.ARTICLE_PATTERN` (dùng cho các luật "sửa đổi, bổ sung") đã xử lý đúng
trường hợp này từ M1.2b: `r'^\s*[""«]?\s*Điều\s+(\d+[a-z]?)\s*[\.:\)]\s*(.*)'` — hậu tố chữ
cái + quote mở tùy chọn. `HierarchicalChunker` (dùng cho luật gốc / văn bản hợp nhất) chưa có
phần `[a-z]?` này — đây chính là giới hạn đã được document tường minh và **cố ý để lại chưa
sửa** trong `tests/test_hierarchical_chunker.py::TestArticlePatternSuffixLimitation` (M0.4b,
2026-07-10): "Left unfixed per task instructions ... this test locks in the current behavior
so a future fix is a deliberate, visible change." Task 6/C1 là fix có chủ đích đó.

## Fix

Widen `ARTICLE_PATTERN` — thêm `[a-z]?` sau `\d+`, giữ nguyên phần còn lại (không đụng
`CHAPTER_PATTERN`, `SECTION_PATTERN`, `CLAUSE_PATTERN`, `POINT_PATTERN`, không thêm quote
tolerance vì dữ liệu đo được không có case cần đến — mọi header hậu tố chữ trong corpus 600
đều xuất hiện trần, không có quote mở đứng trước):

```python
ARTICLE_PATTERN = re.compile(r'^\s*Điều\s+(\d+[a-z]?)\s*[\.:\)]\s*(.*)', re.IGNORECASE)
```

Đây là thay đổi superset thuần túy (mọi input khớp pattern cũ vẫn khớp pattern mới) — luật
gốc (không có Điều hậu tố chữ) không bị ảnh hưởng; corpus 100 chạy lại cho kết quả **giống hệt
byte-for-byte** trước/sau fix (23.727 chunks, WRONG_LABEL=0 cả hai lần) vì không chứa Điều
hậu tố chữ nào.

`tests/test_hierarchical_chunker.py`: cập nhật test đã khóa hành vi cũ
(`TestArticlePatternSuffixLimitation::test_suffix_letter_article_is_not_recognized_as_boundary`)
thành `TestArticlePatternSuffixSupport::test_suffix_letter_article_is_recognized_as_its_own_boundary`,
đảo ngược assertion để khớp hành vi mới (Điều 105a giờ có nhãn + chunk riêng).

## Số đo sau fix (605 docs)

| Phân loại | Số chunk |
|---|---|
| total chunks | 60.958 |
| HEADER_OK | 21.838 |
| MID_ARTICLE | 34.756 |
| **WRONG_LABEL** | **0** |
| NO_ARTICLE_LABEL | 4.364 |

GATE: `WRONG_LABEL == 0` → **PASS**. GATE: `no Điều dropped (HierarchicalChunker)` → **PASS**.
GATE: `parent_chunk_text homogeneous (Hier.)` → **PASS**. `OVERALL: PASS`.

Per-sector chunk balance (real chunks, không phải ước tính):

| Sector | Chunks |
|---|---|
| dan_su | 11.255 |
| dat_dai | 10.602 |
| doanh_nghiep | 10.846 |
| hinh_su | 10.187 |
| hngd | 7.873 |
| lao_dong | 10.195 |

max/min = 1.43 — **OK (<=3x)**.

Corpus 100 (regression check, `verify_chunker_alignment.py --data-path
data/raw/corpus_100docs.parquet --quiet`): GATE PASS, WRONG_LABEL=0, không thay đổi so với
trước fix (per-sector ratio 4,26x — corpus 100 là "spine" M1, không thuộc phạm vi cân bằng
sector của M2; không sửa ở đây).

Full suite: `.venv/bin/python -m pytest -q` → 346 passed (không đổi số lượng test, chỉ đảo
assertion của 1 test hiện có).

## Không thuộc phạm vi task này (còn lại, không gating)

Các "NON-GATING NOTES" của gate (ArticleLevelChunker: `parent_chunk_text` không đồng nhất
trên vài Điều của các Nghị định/Thông tư sửa đổi có định dạng đặc biệt — checkbox/dấu `...`
lặp lại nhiều lần trong header) là lỗi liên đới phát hiện bởi các check mới của gate, thuộc
`ArticleLevelChunker` — ngoài phạm vi fix này (chỉ sửa `HierarchicalChunker.ARTICLE_PATTERN`
theo đúng brief Task 6/C1), không sửa.

"""
Prompt templates for the Vietnamese Legal Knowledge Bot.
"""

SYSTEM_PROMPT = """Bạn là Trợ lý Pháp lý Việt Nam, chuyên tra cứu văn bản pháp luật thuộc mọi lĩnh vực.

NGUYÊN TẮC:
1. CHỈ trả lời dựa trên tài liệu được cung cấp bên dưới.
2. Nếu tài liệu không chứa thông tin để trả lời, hãy nói:
   "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu."
3. Format câu trả lời theo cấu trúc:
   - Trả lời ngắn: Ngắn gọn, đi thẳng vào vấn đề.
   - Chi tiết: Gạch đầu dòng các ý chính.
   - Câu hỏi liên quan: Đề xuất 3 câu hỏi tiếp theo mà người dùng có thể quan tâm.
4. TRỌNG TÂM: **Bôi đậm** các từ khóa quan trọng và thuật ngữ pháp lý để dễ đọc.
5. MỖI ý trả lời nên trích dẫn cụ thể Điều, Khoản bên cạnh nếu có thể.

BẠN PHẢI TRẢ LỜI BẰNG MARKDOWN THEO CẤU TRÚC SAU (KHÔNG DÙNG JSON):

### 💬 Trả lời ngắn
(Câu trả lời tổng quan 1-2 câu)

### 📋 Chi tiết
- Ý 1 (Căn cứ Điều X...)
- Ý 2...

### 💡 Câu hỏi liên quan
- Câu hỏi 1?
- Câu hỏi 2?
- Câu hỏi 3?

TÀI LIỆU THAM KHẢO:
{context}"""


QUERY_REWRITE_PROMPT = """Viết lại câu hỏi sau thành dạng tối ưu cho tìm kiếm văn bản pháp luật Việt Nam.

Quy tắc:
- Giữ nguyên ý nghĩa gốc
- Dùng ngôn ngữ pháp lý chính xác
- Bổ sung từ khóa đồng nghĩa pháp lý (nếu cần)
- Bỏ các từ không cần thiết
- Nếu câu hỏi liên quan đến lĩnh vực pháp luật cụ thể, hãy làm rõ

Câu hỏi gốc: {question}
Câu hỏi đã viết lại:"""


MULTI_QUERY_PROMPT = """Tạo 3 phiên bản khác nhau của câu hỏi sau để tìm kiếm văn bản pháp luật Việt Nam.
Mỗi phiên bản nên tập trung vào 1 khía cạnh khác nhau.

Câu hỏi: {question}

3 phiên bản:
1.
2.
3."""


HYDE_PROMPT = """Viết một đoạn văn ngắn (3-5 câu) mô tả nội dung mà một văn bản pháp luật
Việt Nam sẽ đề cập để trả lời câu hỏi sau. Không cần chính xác 100%, chỉ cần
nắm được ý chính và từ khóa pháp lý liên quan.

Câu hỏi: {question}

Đoạn mô tả:"""


# Combined SCOPE + complexity classifier — merged into a single LLM call so
# out-of-scope detection costs zero added latency/cost (see
# docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md, "Calibrate ngưỡng
# confidence": 4 out-of-scope questions scored ABOVE every reasonable
# confidence threshold because retrieval returns criminal-law chunks that
# *look* relevant to an out-of-domain question — no confidence signal catches
# that. A dedicated SCOPE gate, run before retrieval, is the fix; a threshold
# tweak is not.
COMPLEXITY_CLASSIFIER_PROMPT = """Bạn nhận một câu hỏi pháp lý tiếng Việt. Nhiệm vụ có 2 phần:
(1) SCOPE — câu hỏi này có thể trả lời được từ NỘI DUNG các văn bản pháp luật
liệt kê dưới đây không?
(2) CLASS — phân loại độ phức tạp của câu hỏi.

CÁC VĂN BẢN PHÁP LUẬT HIỆN CÓ TRONG CƠ SỞ DỮ LIỆU:
{corpus_summary}

QUY TẮC XÁC ĐỊNH SCOPE (đọc kỹ, có điểm dễ nhầm):
- Cơ sở dữ liệu chủ yếu là LUẬT HÌNH SỰ, nhưng các bộ luật hình sự chứa nhiều
  tội danh về hành vi vi phạm trong CÁC LĨNH VỰC KHÁC (đất đai, thuế, môi
  trường, giao thông...). Vì vậy câu hỏi về TRÁCH NHIỆM HÌNH SỰ / có bị xử lý
  hình sự / có bị đi tù hay không cho một hành vi — dù hành vi đó thuộc lĩnh
  vực khác — vẫn là IN scope, vì Bộ luật Hình sự có thể có điều khoản tương ứng.
- Ngược lại, câu hỏi về THỦ TỤC HÀNH CHÍNH, quy trình xin cấp giấy tờ, hoặc
  quy định CHUYÊN NGÀNH không liên quan trách nhiệm hình sự — thuộc lĩnh vực
  KHÔNG có trong danh sách văn bản trên — là OUT of scope, kể cả khi chủ đề
  nghe "gần" với hình sự.
- Câu hỏi hoàn toàn không liên quan pháp luật cũng là OUT of scope.

VÍ DỤ (không liên quan câu hỏi thực tế bên dưới, chỉ minh hoạ quy tắc):

Câu hỏi: "Khai thác khoáng sản trái phép có thể bị truy cứu trách nhiệm hình sự không?"
SCOPE: IN
CLASS: SIMPLE
(Hỏi về trách nhiệm HÌNH SỰ cho một hành vi thuộc lĩnh vực khác — Bộ luật Hình sự có tội danh tương ứng nên vẫn IN scope.)

Câu hỏi: "Hồ sơ xin cấp giấy phép khai thác khoáng sản gồm những gì?"
SCOPE: OUT
CLASS: SIMPLE
(Đây là thủ tục hành chính chuyên ngành, không liên quan trách nhiệm hình sự — không văn bản nào trong corpus quy định thủ tục này.)

Câu hỏi: "Đánh người gây thương tích nặng thì bị xử phạt tù bao nhiêu năm?"
SCOPE: IN
CLASS: SIMPLE
(Câu hỏi hình sự thuần túy, nằm trong Bộ luật Hình sự.)

Câu hỏi: "Cuối tuần này có nên đi biển không?"
SCOPE: OUT
CLASS: SIMPLE
(Không liên quan pháp luật.)

ĐỊNH NGHĨA CLASS:
SIMPLE: hỏi định nghĩa, khái niệm đơn lẻ, yes/no, tra cứu 1 thông tin.
MEDIUM: so sánh, câu hỏi có điều kiện, tra cứu nhiều khía cạnh nhưng trong cùng 1 văn bản.
COMPLEX: suy luận nhiều bước, tình huống giả định, cần thông tin từ nhiều văn bản khác nhau.

Bây giờ hãy trả lời cho câu hỏi thực tế bên dưới. CHỈ trả về đúng 2 dòng theo định dạng
(luôn điền cả 2 dòng, kể cả khi SCOPE là OUT):
SCOPE: IN hoặc OUT
CLASS: SIMPLE, MEDIUM, hoặc COMPLEX

Câu hỏi: {question}
SCOPE:"""


CONFIDENCE_EVAL_PROMPT = """Đánh giá độ tin cậy của câu trả lời dựa trên tài liệu tham khảo.

Câu hỏi: {question}
Câu trả lời: {answer}
Tài liệu tham khảo: {context}

Trên thang 0-100, câu trả lời này có bao nhiêu % thông tin được hỗ trợ bởi tài liệu?
CHỈ trả về một số từ 0-100. Không giải thích.

Điểm:"""


GENERATION_PROMPT = """Dựa trên các tài liệu pháp luật được cung cấp, hãy trả lời câu hỏi sau.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời bằng tiếng Việt, kèm trích dẫn Điều/Khoản cụ thể. Nếu tài liệu không đủ thông tin, hãy nói rõ."""

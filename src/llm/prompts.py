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


COMPLEXITY_CLASSIFIER_PROMPT = """Phân loại câu hỏi pháp lý tiếng Việt sau vào 1 trong 3 mức:

SIMPLE: hỏi định nghĩa, khái niệm đơn lẻ, yes/no, tra cứu 1 thông tin.
  VD: "Công ty TNHH là gì?", "Luật hôn nhân gia đình có hiệu lực từ năm nào?"

MEDIUM: so sánh, câu hỏi có điều kiện, tra cứu nhiều khía cạnh nhưng trong cùng 1 văn bản.
  VD: "Điều kiện ly hôn đơn phương là gì?", "Thủ tục đăng ký kết hôn cần những giấy tờ gì?"

COMPLEX: suy luận nhiều bước, tình huống giả định, cần thông tin từ nhiều văn bản khác nhau.
  VD: "Tôi muốn mua đất và xây nhà, cần tuân thủ những quy định gì về đất đai và xây dựng?",
  "Nếu ly hôn và có con dưới 3 tuổi, quyền nuôi con thuộc về ai và thủ tục thế nào?"

Chỉ trả lời 1 từ: SIMPLE, MEDIUM, hoặc COMPLEX.

Câu hỏi: {question}
Phân loại:"""


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

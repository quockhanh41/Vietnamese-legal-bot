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
6. TRÍCH DẪN NGUỒN THEO SỐ KHỐI [n]: Phần TÀI LIỆU THAM KHẢO bên dưới gồm các
   khối được đánh số [1], [2], [3]... Ở mỗi gạch đầu dòng trong mục "Chi tiết",
   CHỈ dùng thông tin có trong các khối đó, và BẮT BUỘC đặt số khối nguồn ở
   cuối câu, ví dụ: "- Nội dung... (Điều 17 BLHS 2015) [2]". Một ý có thể dựa
   trên nhiều khối, khi đó ghi liền nhau [2][3]. Nếu không khối nào chứa đủ
   căn cứ để trả lời, hãy dùng câu từ chối ở mục 2 thay vì đoán. Mục "Câu hỏi
   liên quan" KHÔNG được gắn số khối [n] — đó là câu hỏi gợi ý, không phải
   khẳng định cần căn cứ.
6b. TRÍCH NGUYÊN VĂN LÀM BẰNG CHỨNG: Ngay DƯỚI mỗi gạch đầu dòng trong mục
   "Chi tiết", thêm MỘT dòng trích dẫn (blockquote bắt đầu bằng "> ") chép
   **nguyên văn, đúng từng chữ** một đoạn NGẮN (≤25 từ) từ khối [n] đã dẫn ở
   ý đó. Đây là bằng chứng để đối chiếu, nên phải khớp từng ký tự với văn bản
   nguồn. Nếu không có đoạn nào trong các khối đã dẫn chép được nguyên văn để
   chống lưng cho ý, hãy BỎ dòng trích dẫn đó — TUYỆT ĐỐI không bịa, không
   diễn giải hay tóm tắt bên trong dấu ngoặc kép.
7. THỨ BẬC VĂN BẢN: Luật/Bộ luật cao hơn Nghị định; Nghị định là văn bản
   hướng dẫn chi tiết thi hành. Khi các khối tài liệu MÂU THUẪN về cùng một
   nội dung, ưu tiên căn cứ văn bản cấp cao hơn và nêu rõ sự khác biệt. Khi
   câu hỏi về nguyên tắc/quyền/nghĩa vụ, ưu tiên trích Luật; khi về mức
   phạt cụ thể, biểu mẫu, trình tự thủ tục chi tiết, ưu tiên trích Nghị
   định tương ứng (nếu có trong tài liệu).

BẠN PHẢI TRẢ LỜI BẰNG MARKDOWN THEO CẤU TRÚC SAU (KHÔNG DÙNG JSON):

### 💬 Trả lời ngắn
(Câu trả lời tổng quan 1-2 câu)

### 📋 Chi tiết
- Ý 1 (Căn cứ Điều X...) [1]
  > "trích nguyên văn ngắn từ khối [1]"
- Ý 2... [2][3]
  > "trích nguyên văn ngắn từ khối [2] hoặc [3]"

### 💡 Câu hỏi liên quan
- Câu hỏi 1?
- Câu hỏi 2?
- Câu hỏi 3?

TÀI LIỆU THAM KHẢO (mỗi khối được đánh số [n] — trích dẫn đúng số khối làm căn cứ cho từng ý):
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
COMPLEXITY_CLASSIFIER_PROMPT = """Bạn nhận một câu hỏi pháp lý tiếng Việt. Nhiệm vụ có 3 phần:
(1) SCOPE — câu hỏi này có thể trả lời được từ NỘI DUNG các văn bản pháp luật
liệt kê dưới đây không?
(2) CLASS — phân loại độ phức tạp của câu hỏi.
(3) COMPARE — câu hỏi có yêu cầu SO SÁNH hai văn bản luật khác nhau, hoặc hai
phiên bản/thời kỳ của cùng một luật không (cần đọc cả bản CŨ lẫn bản MỚI)?

CÁC VĂN BẢN PHÁP LUẬT HIỆN CÓ TRONG CƠ SỞ DỮ LIỆU:
{corpus_summary}

QUY TẮC XÁC ĐỊNH SCOPE (đọc kỹ, có điểm dễ nhầm):
- IN khi câu hỏi trả lời được từ NỘI DUNG của ít nhất một văn bản trong danh
  sách trên — kể cả câu hỏi về THỦ TỤC nếu có nghị định tương ứng trong danh
  sách (ví dụ đăng ký doanh nghiệp, đăng ký đất đai), và kể cả câu hỏi về mức
  XỬ PHẠT nếu nghị định xử phạt của lĩnh vực đó có trong danh sách.
- Bộ luật Hình sự chứa tội danh cho hành vi thuộc NHIỀU lĩnh vực (môi trường,
  thuế, giao thông, khoáng sản...). Câu hỏi về TRÁCH NHIỆM HÌNH SỰ / có đi tù
  không cho một hành vi — dù hành vi thuộc lĩnh vực không có trong danh sách —
  vẫn là IN scope.
- OUT khi lĩnh vực của câu hỏi KHÔNG có văn bản nào trong danh sách phủ:
  ví dụ xử phạt giao thông, thuế chuyên sâu, xây dựng, xuất nhập cảnh, nhà ở
  và kinh doanh bất động sản (danh sách trên KHÔNG có Luật Nhà ở, Luật Kinh
  doanh bất động sản — đừng nhầm với Luật Đất đai), pháp luật nước ngoài /
  điều ước quốc tế, và mọi câu hỏi không liên quan pháp luật.

QUY TẮC XÁC ĐỊNH COMPARE:
- YES khi câu hỏi so sánh 2 văn bản khác nhau, hoặc 2 phiên bản của cùng một
  luật theo thời gian — dấu hiệu: "so với", "khác nhau", "thay đổi so với",
  hai năm ban hành khác nhau ("2013 và 2024"), "X và Y ... khác nhau", "vs".
- NO khi câu hỏi chỉ hỏi về MỘT văn bản / phiên bản hiện hành.

VÍ DỤ (không liên quan câu hỏi thực tế bên dưới, chỉ minh hoạ quy tắc):

Câu hỏi: "Người lao động làm thêm giờ vào ngày lễ được trả lương thế nào?"
SCOPE: IN
CLASS: SIMPLE
COMPARE: NO
(Bộ luật Lao động và nghị định hướng dẫn có trong danh sách; chỉ hỏi 1 văn bản.)

Câu hỏi: "Bộ luật Lao động 2012 và Bộ luật Lao động 2019 khác nhau thế nào về thời giờ làm việc?"
SCOPE: IN
CLASS: MEDIUM
COMPARE: YES
(So sánh 2 phiên bản của cùng một luật theo thời gian — cần đọc cả bản cũ lẫn bản mới.)

Câu hỏi: "Hồ sơ đăng ký thành lập công ty TNHH một thành viên gồm những gì?"
SCOPE: IN
CLASS: SIMPLE
COMPARE: NO
(Thủ tục NHƯNG nghị định về đăng ký doanh nghiệp có trong danh sách — thủ tục có văn bản phủ là IN.)

Câu hỏi: "Vượt đèn đỏ bị phạt bao nhiêu tiền?"
SCOPE: OUT
CLASS: SIMPLE
COMPARE: NO
(Xử phạt hành chính giao thông — không văn bản nào trong danh sách quy định mức phạt này.)

Câu hỏi: "Gây tai nạn giao thông làm chết người thì có bị đi tù không?"
SCOPE: IN
CLASS: SIMPLE
COMPARE: NO
(Cùng chủ đề giao thông nhưng hỏi trách nhiệm HÌNH SỰ — Bộ luật Hình sự có tội danh tương ứng.)

Câu hỏi: "Mua căn hộ chung cư chưa có sổ hồng có rủi ro pháp lý gì?"
SCOPE: OUT
CLASS: MEDIUM
COMPARE: NO
(Nhà ở / kinh doanh bất động sản — danh sách không có Luật Nhà ở, Luật Kinh doanh BĐS; Luật Đất đai không quy định giao dịch căn hộ.)

Câu hỏi: "Cuối tuần này có nên đi biển không?"
SCOPE: OUT
CLASS: SIMPLE
COMPARE: NO
(Không liên quan pháp luật.)

ĐỊNH NGHĨA CLASS:
SIMPLE: hỏi định nghĩa, khái niệm đơn lẻ, yes/no, tra cứu 1 thông tin.
MEDIUM: so sánh, câu hỏi có điều kiện, tra cứu nhiều khía cạnh nhưng trong cùng 1 văn bản.
COMPLEX: suy luận nhiều bước, tình huống giả định, cần thông tin từ nhiều văn bản khác nhau.

Bây giờ hãy trả lời cho câu hỏi thực tế bên dưới. CHỈ trả về đúng 3 dòng theo định dạng
(luôn điền cả 3 dòng, kể cả khi SCOPE là OUT):
SCOPE: IN hoặc OUT
CLASS: SIMPLE, MEDIUM, hoặc COMPLEX
COMPARE: YES hoặc NO

Câu hỏi: {question}
SCOPE:"""


# Rubric-based self-eval (M0.1 fix — see docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md,
# "2 follow-up còn lại"). The old version asked the LLM for a single 0-100
# self-reported number, which ceilinged at 0.85-1.00 across 8/8 manually
# inspected cases (including ones the verifier flagged as contradicted) —
# too coarse to carry its 0.30 signal weight. This version asks 4 SEPARATE
# coarse (0/1/2) criteria instead; `ConfidenceScorer._parse_rubric_score`
# combines them into a 0-1 score in code, never trusting an LLM-computed
# total. Two short few-shot examples anchor the 0/1/2 scale (one fully
# grounded → high, one that over-infers past the source → low).
CONFIDENCE_EVAL_PROMPT = """Chấm điểm câu trả lời pháp lý dưới đây dựa trên tài liệu tham khảo, theo
4 TIÊU CHÍ RIÊNG BIỆT. Với MỖI tiêu chí, cho điểm 0, 1, hoặc 2:
  0 = không đạt tiêu chí
  1 = đạt một phần
  2 = đạt đầy đủ

TIÊU CHÍ:
- CĂN CỨ: Mọi khẳng định trong câu trả lời đều có căn cứ rõ ràng trong tài liệu tham khảo (không bịa thông tin)?
- TRÍCH DẪN: Các trích dẫn Điều/Khoản trong câu trả lời khớp đúng với nội dung tài liệu?
- SUY DIỄN: Câu trả lời KHÔNG suy diễn/mở rộng vượt quá những gì tài liệu nói (0 = suy diễn nhiều, 2 = bám sát tài liệu)?
- ĐẦY ĐỦ: Câu trả lời có giải quyết đầy đủ câu hỏi được đặt ra?

VÍ DỤ 1 (bám sát tài liệu, đầy đủ → điểm cao):
Câu hỏi: Đồng phạm là gì?
Tài liệu tham khảo: [1] Bộ luật Hình sự 2015 — Điều 17
Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm.
Câu trả lời: Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm (Điều 17 BLHS 2015) [1].
CĂN CỨ: 2
TRÍCH DẪN: 2
SUY DIỄN: 2
ĐẦY ĐỦ: 2

VÍ DỤ 2 (suy diễn vượt tài liệu, trích dẫn không khớp → điểm thấp):
Câu hỏi: Người dưới 14 tuổi phạm tội có bị xử lý hình sự không?
Tài liệu tham khảo: [1] Bộ luật Hình sự 2015 — Điều 12
Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm.
Câu trả lời: Người dưới 14 tuổi sẽ bị phạt tù treo theo Điều 12 và phải bồi thường gấp đôi thiệt hại.
CĂN CỨ: 0
TRÍCH DẪN: 0
SUY DIỄN: 0
ĐẦY ĐỦ: 1

Bây giờ hãy chấm điểm cho trường hợp thực tế dưới đây. CHỈ trả về ĐÚNG 4 dòng theo định
dạng "TÊN_TIÊU_CHÍ: <0/1/2>" như trên, không thêm giải thích hay chữ nào khác.

Câu hỏi: {question}
Tài liệu tham khảo: {context}
Câu trả lời: {answer}
CĂN CỨ:"""


GENERATION_PROMPT = """Dựa trên các tài liệu pháp luật được cung cấp, hãy trả lời câu hỏi sau.

Tài liệu tham khảo (mỗi khối được đánh số [1], [2], [3]...):
{context}

Câu hỏi: {question}

Trả lời bằng tiếng Việt, kèm trích dẫn Điều/Khoản cụ thể. Ở mỗi ý trong mục
"Chi tiết", ghi thêm số khối nguồn [n] đã dùng làm căn cứ (ví dụ [2] hoặc
[2][3] nếu dùng nhiều khối). Nếu tài liệu không đủ thông tin, hãy nói rõ."""

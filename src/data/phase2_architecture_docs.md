# Tài liệu Kiến trúc Phase 2: Knowledge Graph & Phiên bản hoá

Tài liệu này giải thích chi tiết về luồng hoạt động (flow) của 3 trụ cột chính trong kiến trúc Phase 2 của Vietnamese Legal Bot, dựa trên thiết kế và mã nguồn hiện tại. Mục tiêu của Phase 2 là chuyển đổi từ việc lưu trữ văn bản phẳng (flat documents) sang một Đồ thị Tri thức có tính đến mốc thời gian và sự sửa đổi (Versioned Knowledge Graph).

---

## 1. Cross-Referencing (Trích xuất & Liên kết tham chiếu)

**Mục tiêu:** Nhận diện và liên kết các Điều khoản viện dẫn lẫn nhau để hệ thống có thể hiểu được ngữ cảnh rộng hơn của một Điều luật.

**Thư mục xử lý:** `src/data/graph/`

### Luồng hoạt động (Flow):
1. **Trích xuất (Extraction - `ref_extractor.py`)**: 
   - Hệ thống đọc từng đoạn text của Điều luật và sử dụng các biểu thức chính quy (Regex) được thiết kế đặc thù cho ngôn ngữ pháp lý Việt Nam để tìm kiếm các patterns.
   - **Các patterns được nhận diện theo thứ tự ưu tiên:**
     - Tham chiếu ngoại (External): "Điều X của Luật Y" hoặc "Điều X của Bộ luật Y".
     - Tham chiếu nội (Internal): "Điều X của Luật/Bộ luật này".
     - Tham chiếu khoản (Clause): "khoản X Điều Y".
     - Tham chiếu chung: "theo quy định tại Điều X" hoặc "tại Chương X".
   - **Output:** Danh sách các đối tượng `Reference` chứa thông tin đã được chuẩn hóa (ví dụ: `source_article`, `target_law_name`, `target_article`, `ref_type`).

2. **Xây dựng đồ thị (Graph Building - `graph_builder.py`)**:
   - Sử dụng thư viện `NetworkX` để tạo một đồ thị có hướng (`DiGraph`).
   - **Nodes (Đỉnh):** Đại diện cho các Điều luật (ví dụ: `doc_123_điều_5`). Nếu là tham chiếu ngoại (đến một luật khác), một "virtual node" (đỉnh ảo) sẽ được tạo ra.
   - **Edges (Cạnh):** Là các mối liên kết giữa các Điều luật (có chứa metadata như `ref_type`, `confidence`).
   
3. **Mở rộng ngữ cảnh (Context Expansion)**:
   - Khi người dùng query một Điều luật, hàm `expand_context(hops=1)` có thể được gọi để duyệt đồ thị (BFS) và lấy ra các Điều luật lân cận mà nó tham chiếu đến (hoặc bị tham chiếu). Điều này giúp Bot trả lời đầy đủ và chính xác hơn khi một Điều luật phụ thuộc vào Điều luật khác.

---

## 2. Temporal Versioning (Phiên bản hóa theo thời gian)

**Mục tiêu:** Xác định xem một đạo luật (hoặc một Điều luật cụ thể) có hiệu lực trong khoảng thời gian nào, giúp Bot trả lời được câu hỏi "Quy định X hiện nay như thế nào?" hoặc "Vào năm 2018 thì quy định X ra sao?".

**Thư mục xử lý:** `src/data/timeline/`

### Luồng hoạt động (Flow):
1. **Trích xuất thời gian (Date Extraction - `date_extractor.py`)**:
   - Hệ thống sẽ tìm kiếm Điều khoản quy định về "Hiệu lực thi hành" (thường nằm ở cuối luật).
   - Sử dụng regex để bóc tách các ngày quan trọng:
     - **Ngày có hiệu lực (Effective Date):** Từ các cụm "có hiệu lực thi hành từ ngày DD/MM/YYYY".
     - **Ngày hết hiệu lực (Expiry Date):** Từ các cụm "hết hiệu lực kể từ ngày...". Hệ thống cũng bắt được việc luật cũ bị thay thế bởi luật mới ("Luật X hết hiệu lực kể từ ngày Luật này có hiệu lực").
     - **Hiệu lực tương đối (Relative Dates):** Xử lý các cụm như "sau 45 ngày kể từ ngày công bố" bằng cách cộng chuỗi ngày vào ngày ban hành (issuance date).
     - **Hiệu lực riêng từng Điều:** Bắt được các trường hợp một số Điều khoản có ngày hiệu lực khác với luật chung.
   - **Output:** Đối tượng `LawTimeline` chứa toàn bộ mốc thời gian của luật đó.

2. **Truy vấn phiên bản (Version Resolution - `version_resolver.py`)**:
   - Là một Query Engine thời gian thực (time-aware).
   - Hàm `at_time(when)` sẽ kiểm tra `LawTimeline` của các đạo luật để phân loại trạng thái của chúng tại thời điểm `when`:
     - **Current (Đang hiệu lực):** `effective_date <= when` và chưa tới `expiry_date`.
     - **Future (Tương lai):** `effective_date > when`.
     - **Expired (Hết hiệu lực):** `expiry_date <= when`.
   - Hàm `version_chain(law_name)` giúp tạo ra một chuỗi lịch sử của một đạo luật (ví dụ: Luật Doanh nghiệp qua các năm 1999 → 2005 → 2014 → 2020).

---

## 3. Amendment Tracking (Theo dõi sửa đổi, bổ sung)

**Mục tiêu:** Tự động hợp nhất (consolidate) các quy định từ "Luật sửa đổi, bổ sung" vào văn bản luật gốc để tạo ra một phiên bản luật hoàn chỉnh (Văn bản hợp nhất).

*(Lưu ý: Dựa trên `architecture.md`, tính năng này đang trong lộ trình phát triển (Week 2), nhưng luồng thiết kế đã được định hình rõ).*

**Thư mục dự kiến:** `src/data/amendments/`

### Luồng hoạt động (Flow dự kiến):
1. **Phát hiện (Detect):** 
   - Nhận diện các văn bản có tiêu đề chứa "sửa đổi, bổ sung" (Amendment law).
2. **Liên kết (Link):** 
   - Tìm và liên kết văn bản sửa đổi với luật gốc thông qua tên luật được nhắc đến.
3. **Phân tích chỉ thị sửa đổi (Parse Instructions):** 
   - Sử dụng LLM (hoặc Rules) để đọc hiểu và trích xuất các thao tác sửa đổi dưới dạng dữ liệu có cấu trúc (JSON).
   - Ví dụ:
     - `{"action": "replace", "article": "45", "clause": "2", "new_text": "..."}`
     - `{"action": "insert", "article": "123a", "after": "123", "new_text": "..."}`
     - `{"action": "delete", "article": "67", "clause": "3"}`
4. **Áp dụng & Hợp nhất (Apply & Consolidate):** 
   - Hợp nhất (Merge) các chỉ thị trên vào đoạn văn bản của luật gốc, sinh ra một văn bản hợp nhất (consolidated text).
5. **Đánh dấu phiên bản (Version chaining):** 
   - Đẩy kết quả sau khi hợp nhất vào `Temporal Versioning`.
   - Lúc này, một Điều luật sẽ có lịch sử phiên bản: `Điều 45 (Version 1, 2015-2022) → Điều 45 (Version 2, 2022-nay)`.

---

## 4. Tóm tắt bức tranh toàn cảnh (Big Picture)

Khi cả 3 trụ cột này kết hợp lại trong Phase 2, pipeline sẽ đi từ:
1. Đọc văn bản thô (Raw Doc)
2. Nhận diện mốc thời gian (**Temporal Versioning**)
3. Nếu là luật sửa đổi, áp dụng thay đổi vào luật gốc (**Amendment Tracking**)
4. Trích xuất tất cả các mối quan hệ dẫn chiếu (**Cross-Referencing**)
5. Lưa vào Qdrant (Vector DB) kèm theo Context của Knowledge Graph.

Khi người dùng hỏi: *"Theo Luật Hôn nhân, thủ tục X diễn ra thế nào?"*
- **Timeline** sẽ check xem luật Hôn nhân bản nào đang có hiệu lực.
- **Graph** sẽ check xem Điều luật quy định thủ tục X có đang tham chiếu (bắt buộc phải xem thêm) Điều luật của luật Dân sự nào không.
- Nếu có, Bot sẽ tổng hợp kiến thức từ cả 2 Điều luật và sinh câu trả lời chính xác nhất.

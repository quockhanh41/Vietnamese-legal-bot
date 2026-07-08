# Knowledge Graph & Phiên bản hoá (lớp Enrichment)

> Tài liệu này đi sâu vào **lớp làm giàu dữ liệu (enrichment)** — được xây dựng trong Phase 2
> của dự án. Để có bức tranh toàn cảnh về mã nguồn (module nào nằm ở đâu, một câu hỏi chạy
> qua những hàm nào), đọc [`ARCHITECTURE.md`](ARCHITECTURE.md) trước.

Tài liệu này giải thích chi tiết về luồng hoạt động (flow) của 3 trụ cột chính trong lớp enrichment của Vietnamese Legal Bot, dựa trên thiết kế và mã nguồn hiện tại. Mục tiêu là chuyển đổi từ việc lưu trữ văn bản phẳng (flat documents) sang một Đồ thị Tri thức có tính đến mốc thời gian và sự sửa đổi (Versioned Knowledge Graph).

**Vị trí trong hệ thống:** lớp enrichment chạy **tách biệt** với pipeline chat. Kết quả của nó
(graph, timeline, amendment) được lưu trong `MetadataStore` (JSON) và phục vụ qua
`EnhancedRetriever` + các API endpoint riêng (xem mục 5) — pipeline chat chính
(`AdaptiveRAGPipeline`) hiện **chưa** dùng đến các dữ liệu này.

---

## 1. Cross-Referencing (Trích xuất & Liên kết tham chiếu)

**Mục tiêu:** Nhận diện và liên kết các Điều khoản viện dẫn lẫn nhau để hệ thống có thể hiểu được ngữ cảnh rộng hơn của một Điều luật.

**Thư mục xử lý:** `src/data/graph/`

### Luồng hoạt động (Flow):
1. **Trích xuất (Extraction - `ref_extractor.py`)**: 
   - Hệ thống đọc từng đoạn text của Điều luật và sử dụng các biểu thức chính quy (Regex) (10 patterns) được thiết kế đặc thù cho ngôn ngữ pháp lý Việt Nam để tìm kiếm các tham chiếu.
   - **Các loại tham chiếu (5 types — enum `RefType`) được nhận diện:**
     - Tham chiếu ngoại (`EXTERNAL`): "Điều X của Luật Y" hoặc "Điều X của Bộ luật Y".
     - Tham chiếu nội (`INTERNAL`): "Điều X của Luật/Bộ luật này", "theo quy định tại Điều X".
     - Tham chiếu khoản (`CLAUSE`): "khoản X Điều Y".
     - Tham chiếu chương (`CHAPTER`): "tại Chương X".
     - Đa tham chiếu (`MULTI`): "Điều 5 và Điều 6".
   - **Output:** Danh sách các đối tượng `Reference` chứa thông tin đã được chuẩn hóa.

2. **Xây dựng đồ thị (Graph Building - `graph_builder.py`)**:
   - Sử dụng thư viện `NetworkX` để tạo một đồ thị có hướng (`DiGraph`).
   - **Nodes (Đỉnh):** Đại diện cho các Điều luật. Nếu là tham chiếu ngoại, một "virtual node" sẽ được tạo ra.
   - **Edges (Cạnh):** Là các mối liên kết giữa các Điều luật.
   
3. **Mở rộng ngữ cảnh (Context Expansion)**:
   - Khi người dùng query một Điều luật, hàm `expand_context(hops=1)` sử dụng thuật toán duyệt đồ thị (BFS) để lấy ra các Điều luật lân cận (vd: `hops=1` là lân cận 1 mắt xích), giúp Bot trả lời đầy đủ ngữ cảnh.

---

## 2. Temporal Versioning (Phiên bản hóa theo thời gian)

**Mục tiêu:** Xác định xem một đạo luật (hoặc một Điều luật cụ thể) có hiệu lực trong khoảng thời gian nào, giúp Bot trả lời được câu hỏi "Quy định X hiện nay như thế nào?" hoặc "Vào năm 2018 thì quy định X ra sao?".

**Thư mục xử lý:** `src/data/timeline/`

### Luồng hoạt động (Flow):
1. **Trích xuất thời gian (Date Extraction - `date_extractor.py`)**:
   - Parse Điều khoản quy định về "Hiệu lực thi hành".
   - Bóc tách ngày có hiệu lực (Effective Date), ngày hết hiệu lực (Expiry Date), xử lý các hiệu lực tương đối ("sau 45 ngày...") và hiệu lực riêng từng Điều.
   - **Output:** Đối tượng `LawTimeline`.

2. **Truy vấn phiên bản (Version Resolution - `version_resolver.py`)**:
   - Là một Time-aware Query Engine.
   - Phân loại trạng thái của luật (Current, Future, Expired) tại một thời điểm `when` bất kỳ.
   - Tạo ra các Version chain (chuỗi lịch sử của luật).

---

## 3. Amendment Tracking (Theo dõi sửa đổi, bổ sung)

**Mục tiêu:** Tự động hợp nhất (consolidate) các quy định từ "Luật sửa đổi, bổ sung" vào văn bản luật gốc để tạo ra một phiên bản luật hoàn chỉnh (Văn bản hợp nhất).

**Thư mục xử lý:** `src/data/amendment/`

### Luồng hoạt động (Flow):
1. **Trích xuất chỉ thị sửa đổi (Extraction - `amendment_extractor.py`)**: 
   - Sử dụng ~30 regex patterns để đọc hiểu và trích xuất các thao tác sửa đổi dưới dạng dữ liệu có cấu trúc.
   - Hỗ trợ 14 loại thao tác (action types), ví dụ:
     - `batch_insert`: Bổ sung một số từ, cụm từ.
     - `modify_point`: Sửa đổi, bổ sung điểm X khoản Y.
     - `modify_multi_article`: Sửa đổi, bổ sung các điều X, Y, Z.
     - `batch_replace`: Thay thế từ, cụm từ tại các điều.
     - `insert_article` / `insert_point` / `insert_clause`: Bổ sung Điều/khoản/điểm.
2. **Áp dụng & Hợp nhất (Consolidation - `amendment_consolidator.py`)**: 
   - Apply các sửa đổi (Amendments) vào luật gốc (Base law).
   - Hợp nhất các chỉ thị trên vào đoạn văn bản của luật gốc để sinh ra một văn bản hợp nhất (consolidated text).
3. **Đánh dấu phiên bản (Version chaining)**: 
   - Kết quả sau khi hợp nhất được đẩy vào `Temporal Versioning` tạo thành lịch sử phiên bản hoàn chỉnh.

---

## 4. Tích hợp toàn diện (Full Pipeline Integration) ✨ HOÀN TẤT

Sau khi hoàn thiện 3 module độc lập, hệ thống đã được tích hợp thành một Pipeline hoàn chỉnh, sẵn sàng phục vụ từ việc làm giàu dữ liệu (Enrichment) đến quá trình truy vấn (Query-Time).

### Lớp tích hợp (Integration Layer)
- **`src/pipeline/enricher.py`**: Trực tiếp điều phối 3 module (Cross-Referencing, Temporal Versioning, Amendment Tracking) để làm giàu ngữ nghĩa cho văn bản.
- **`src/pipeline/retriever.py`**: Cải tiến Retrieval để có khả năng mở rộng ngữ cảnh (Context Expansion) và xử lý thời gian (Time-aware).
- **`src/data/metadata_store.py`**: Persistent storage (định dạng JSON) lưu trữ các metadata của tài liệu, thiết kế đủ hiệu năng xử lý hàng ngàn văn bản.

### Khả năng Truy vấn thực tế (Query-Time Capabilities)
- **Context expansion (Mở rộng ngữ cảnh qua Cross-references):**
  ```python
  results = retriever.retrieve("Điều 123 BLHS", expand_context=True, expansion_hops=1)
  ```
- **Time-aware search (Truy vấn theo mốc thời gian lịch sử):**
  ```python
  results = retriever.retrieve("Điều 134", at_date=date(2018, 1, 1))
  ```
- **Article amendment history (Lịch sử sửa đổi của một Điều khoản):**
  ```python
  history = retriever.get_article_history(296661, "Điều 3")
  # Trả về: {"is_amended": True, "amending_laws": ["Luật sửa đổi BLHS 2017"]}
  ```

---

## 5. Các Endpoint API mới (Phase 2 APIs)

Hệ thống cung cấp thêm các endpoint trong `src/api/main.py`:

| Endpoint | Method | Description |
|---|---|---|
| `/api/pipeline/enrich` | POST | Chạy Phase 2 enrichment |
| `/api/pipeline/status` | GET | Xem trạng thái enrichment |
| `/api/search/enhanced` | POST | Tìm kiếm với Context Expansion |
| `/api/article/history` | GET | Xem lịch sử sửa đổi của Điều khoản |
| `/api/graph/stats` | GET | Thống kê Knowledge Graph |
| `/api/graph/neighbors` | GET | Lấy graph neighbors (hops) cho Điều khoản |
| `/api/timeline/effective`| GET | Các luật có hiệu lực tại một thời điểm |

---

## 6. Kết quả Kiểm thử End-to-End (End-to-End Test Results)

Thực hiện trên 6 văn bản Bộ luật Hình sự (Criminal Code docs):

**Hiệu suất:**
- **Pipeline Run:** 6.4s | 0 errors
- **Documents:** 6/6 enriched (3 Base laws, 3 Amendment laws)

**Chỉ số:**
- **Graph:** 409 nodes, 403 edges
- **Timeline:** 6 with effective dates, 6 currently in effect
- **Amendments:** 158 instructions, 152 articles affected

**Chi tiết làm giàu dữ liệu (Document Enrichment Detail):**
| Doc ID | Title | Type | Refs | Amendments | Chain |
|---|---|---|---|---|---|
| 46056 | BLHS 1999 | 📜 Base | 53 | 0 | 1999→2015→2017→2025 |
| 296661| BLHS 2015 | 📜 Base | 110| 0 | 1999→2015→2017→2025 |

---

## 7. Tổng tiến độ (Overall Progress)

> [!TIP]
> **Toàn bộ Phase 2 Pipeline đã được tích hợp thành công! Mục tiêu cuối cùng là Scale up.**

| Phase | Status |
|---|---|
| Phase 1: Filter Luật + test ingest | ✅ |
| Module 1: Cross-Referencing | ✅ |
| Module 2: Temporal Versioning | ✅ |
| Module 3: Amendments Tracking | ✅ |
| Full Pipeline Integration | ✅ |
| Scale to all 595 docs | 📋 (Next) |

---

## 8. Tóm tắt bức tranh toàn cảnh (Big Picture)

Khi cả 3 trụ cột này kết hợp lại và thông qua Integration Layer, pipeline sẽ đi từ:
1. Đọc văn bản thô (Raw Doc)
2. Nhận diện mốc thời gian (**Temporal Versioning**)
3. Nếu là luật sửa đổi, áp dụng thay đổi vào luật gốc (**Amendment Tracking**)
4. Trích xuất tất cả các mối quan hệ dẫn chiếu (**Cross-Referencing**)
5. Lưu Metadata (`metadata_store.py`) và lưu Vectors vào Qdrant kèm theo Context của Knowledge Graph.

Khi người dùng hỏi: *"Năm 2018 theo Luật Hình sự, tội X bị xử lý thế nào? Nếu có tham chiếu đến luật khác thì quy định đó nói gì?"*
- **Timeline (`at_date`)** sẽ chọn đúng phiên bản Luật Hình sự có hiệu lực năm 2018.
- **Amendment** đảm bảo phiên bản đó đã chứa các nội dung sửa đổi tương ứng tới thời điểm 2018.
- **Graph (`expand_context`)** sẽ check xem Điều luật quy định tội X có tham chiếu (chẳng hạn tham chiếu khái niệm bên Luật Dân sự) không và kéo theo các Node lân cận (`hops=1`).
- Bot sẽ tổng hợp toàn bộ kiến thức hoàn hảo này để sinh câu trả lời chính xác, đầy đủ ngữ cảnh và tuân thủ tuyệt đối mốc thời gian lịch sử.

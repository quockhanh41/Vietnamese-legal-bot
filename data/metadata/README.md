# Hướng dẫn cấu trúc thư mục Metadata

Thư mục này chứa các file JSON là kết quả đầu ra của bộ **Enrichment Pipeline** (khối Legal Intelligence). Dữ liệu trong này đóng vai trò như "bộ não phụ" để AI tham khảo thông tin ngoài văn bản thô.

*Lưu ý: Do định dạng JSON tiêu chuẩn không hỗ trợ viết comment (dấu `//` sẽ làm lỗi code khi đọc file), nên các giải thích được đặt tại file README này.*

---

## 1. File `index.json`
Đây là file **tổng quan (dashboard)** lưu trữ thống kê toàn cục của toàn bộ dữ liệu luật trong hệ thống.
- `total_documents`: Tổng số lượng văn bản luật đang có trong cơ sở dữ liệu.
- `amendment_laws`: Số lượng văn bản là "Luật sửa đổi, bổ sung".
- `base_laws`: Số lượng văn bản là luật gốc.
- `with_effective_date`: Số lượng văn bản đã xác định được ngày có hiệu lực.
- `currently_in_effect`: Số lượng văn bản hiện đang còn hiệu lực tính đến thời điểm hiện tại.
- `total_graph_nodes` / `total_graph_edges`: Tổng số đỉnh (nodes) và cạnh (edges) trong mạng lưới liên kết trích dẫn luật.
- `total_amendments` / `total_articles_amended`: Tổng số chỉ thị sửa đổi phát hiện được và tổng số điều khoản đã bị sửa.
- `last_enriched`: Thời gian chạy pipeline gần nhất.
- `documents`: Mảng tóm tắt nhanh của từng văn bản (ID, tiêu đề, có phải luật sửa đổi không, ngày hiệu lực...).

---

## 2. File `documents.json`
Chứa **siêu dữ liệu (metadata) chi tiết của từng văn bản luật**, key là `doc_id`. File này được AI dùng để biết một văn bản có đặc tính gì, hay bị sửa đổi phần nào.
- `doc_id`: ID định danh duy nhất của văn bản (khớp với ID trên Qdrant).
- `title`: Tên đầy đủ của văn bản (VD: "Bộ luật hình sự 2015").
- `ref_count`: Tổng số trích dẫn (reference) mà văn bản này nhắc tới.
- `internal_refs`: Số lần trích dẫn chéo trong cùng văn bản (VD: Điều 5 nhắc đến Điều 3).
- `external_refs`: Số lần tham chiếu sang văn bản luật khác.
- `articles_referenced`: Danh sách các Điều luật bị đem ra trích dẫn.
- *(Nếu là luật gốc bị sửa đổi)* `amendment_count` / `amended_articles`: Bị sửa đổi bao nhiêu lần và danh sách các điều khoản nào bị sửa đổi.
- *(Nếu là luật sửa đổi)* `target_laws`: Tên của các luật gốc mà văn bản này nhắm tới để sửa.
- `version_chain`: Chuỗi phiên bản (hiệu lực) của luật này (bị sửa bởi luật nào, năm nào).

---

## 3. File `timelines.json`
Lưu trữ thông tin về **mốc thời gian sinh tử (vòng đời)** của các văn bản luật.
- `doc_id`, `law_name`: ID và Tên văn bản.
- `issuance_date`: Ngày ban hành (chuỗi text gốc).
- `effective_date`: Ngày có hiệu lực (chuẩn ISO YYYY-MM-DD) do máy học tự động bóc tách.
- `expiry_date`: Ngày hết hiệu lực (nếu có).
- `is_in_effect`: Boolean (True/False) xác định xem luật này hiện tại còn hiệu lực không.
- `replaced_laws`: Danh sách các văn bản luật cũ bị văn bản này thay thế/bãi bỏ hoàn toàn.
- `source_text`: Đoạn văn bản gốc chứa thông tin về hiệu lực (dùng làm bằng chứng).

---

## 4. File `graph.json`
Lưu trữ **mạng lưới liên kết (Knowledge Graph)** của toàn bộ hệ thống luật dưới dạng Đỉnh (`nodes`) và Cạnh (`edges`).
- **`nodes`**: Danh sách các thực thể.
  - `id`: Mã định danh (VD: `doc_296661_điều_76`).
  - `doc_id`, `law_name`, `article`: Thuộc về văn bản nào và điều khoản nào.
  - `node_type`: Loại đỉnh (thường là `document` hoặc `article`).
- **`edges`**: Danh sách các đường nối (mối quan hệ trích dẫn).
  - `source`: Đỉnh nguồn (điều khoản đang chứa câu trích dẫn).
  - `target`: Đỉnh đích (điều khoản bị trích dẫn).
  - `relationship`: Loại quan hệ (thường là `REFERENCES`).
  - `ref_type`: `internal` (trích dẫn trong cùng 1 luật) hoặc `external` (tham chiếu luật khác).
  - `context`: Trích đoạn văn bản gốc chứa câu tham chiếu đó.

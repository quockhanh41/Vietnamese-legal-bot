import json
import sys
import os

# Đảm bảo có thể import module từ thư mục src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.chunker import LegalDocumentChunker

def test_chunking():
    input_file = "data/raw/sample_5_rows.json"
    output_file = "data/raw/sample_chunked.json"
    
    print(f"Đọc dữ liệu từ {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        rows = json.load(f)
        
    # Lấy văn bản đầu tiên
    first_row = rows[0]
    print(f"Đang chunk văn bản: {first_row.get('title', 'Unknown')} (ID: {first_row.get('id')})")
    
    # Khởi tạo Chunker (sử dụng logic Regex mới bạn vừa cập nhật)
    chunker = LegalDocumentChunker(min_chunk_size=200, max_chunk_size=1000)
    
    # Thực hiện chunking
    chunks = chunker.chunk_document(first_row)
    
    # Chuyển đổi list object sang dạng dictionary để dump ra JSON
    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    
    # Ghi ra file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
        
    print(f"\nThành công! Văn bản gốc có độ dài {len(first_row.get('content', ''))} ký tự.")
    print(f"Đã chia thành {len(chunks)} chunks.")
    print(f"Kết quả được lưu chi tiết tại: {output_file}")

if __name__ == "__main__":
    test_chunking()

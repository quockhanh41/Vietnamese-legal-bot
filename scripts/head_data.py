import pyarrow.dataset as ds
import pandas as pd
import json

def get_first_5_rows(data_path="data/raw/legal_docs_merged.parquet", output_path="data/raw/sample_5_rows.json"):
    """
    Đọc 5 dòng đầu tiên từ file parquet bằng pyarrow (để không tốn RAM).
    Sau đó ghi ra file JSON (giữ nguyên toàn bộ nội dung).
    """
    try:
        # Sử dụng pyarrow dataset để đọc dạng streaming (tương tự như ingest.py)
        dataset = ds.dataset(data_path, format="parquet")
        
        # Chỉ lấy 1 batch nhỏ (đọc đúng 5 dòng)
        for batch in dataset.to_batches(batch_size=5):
            df = batch.to_pandas()
            
            # Không cắt bớt nội dung nữa để lấy toàn văn
            
            # Chuyển thành dạng danh sách các dictionary
            records = df.to_dict(orient="records")
            
            # Ghi ra file với định dạng JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
                
            print(f"Đã xuất thành công 5 dòng toàn văn ra file: {output_path}")
            
            # Chỉ lấy 1 batch đầu tiên rồi dừng luôn
            break
            
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        print("Đảm bảo bạn đã chạy 'python scripts/download_dataset.py' để tải dữ liệu về máy trước.")

if __name__ == "__main__":
    get_first_5_rows()

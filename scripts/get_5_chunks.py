import sys
import os
import json
import pandas as pd

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.chunker import LegalDocumentChunker

def main():
    # 6 văn bản (docs) đang được dùng làm MVP/Phase 2
    target_ids = [46056, 296661, 296884, 354053, 489113, 662377]
    parquet_path = "data/raw/legal_docs_merged.parquet"
    out_path = "data/raw/sample_5_chunks_from_6docs.json"

    print(f"Reading dataset from {parquet_path}...")
    
    # Sử dụng pushdown filter để Pandas/PyArrow chỉ quét và nạp đúng 6 docs lên RAM
    df = pd.read_parquet(
        parquet_path, 
        engine='pyarrow', 
        filters=[('id', 'in', target_ids)]
    )
    
    docs_to_chunk = df.to_dict('records')
    print(f"Found {len(docs_to_chunk)} documents.")
    
    chunker = LegalDocumentChunker()
    all_chunks = []
    
    print("Chunking documents...")
    for doc in docs_to_chunk:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        
    print(f"Total chunks created from {len(docs_to_chunk)} docs: {len(all_chunks)}")
    
    # Lấy 5 dòng (chunks) đầu tiên
    first_5_chunks = all_chunks[:5]
    
    # Convert LegalChunk dataclass to dict for JSON serialization
    chunks_dict = [c.to_dict() for c in first_5_chunks]
        
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved 5 first chunks to {out_path}")

if __name__ == "__main__":
    main()

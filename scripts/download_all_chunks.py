import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def download_all_chunks(collection_name="luat_law", output_file="data/raw/luat_law_chunks.json"):
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        print("Missing QDRANT_URL or QDRANT_API_KEY in .env file")
        return

    print(f"Connecting to Qdrant Cloud at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' found. Points count: {info.points_count}")
    except Exception as e:
        print(f"Failed to get collection info: {e}")
        return

    all_payloads = []
    offset = None
    batch_size = 1000
    
    print(f"Downloading chunks from collection '{collection_name}'...")
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        for point in points:
            all_payloads.append(point.payload)
            
        print(f"Downloaded {len(all_payloads)} chunks...", end="\r")
        
        if offset is None:
            break
            
    print(f"\nFinished downloading. Total chunks: {len(all_payloads)}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_payloads, f, ensure_ascii=False, indent=2)
        
    print(f"Saved all chunks to {output_file}")

if __name__ == "__main__":
    download_all_chunks()

"""
Full data ingestion pipeline with STREAMING (low memory).

⚠️  518K documents = ~10GB in pandas. This script uses pyarrow batch reading
    to process row groups one at a time — never loads full dataset into RAM.

Usage:
    python scripts/ingest.py --limit 100           # Test with 100 docs
    python scripts/ingest.py --limit 5000           # Medium test
    python scripts/ingest.py --recreate              # Full ingest (takes hours)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import click
import pyarrow.dataset as ds
from tqdm import tqdm

from src.data.chunker import LegalDocumentChunker
from src.data.embedder import BGEEmbedder
from src.db.qdrant_client import LegalVectorStore


# --- Filter Configuration ---
LEGAL_TYPES = {"Luật", "Bộ luật", "Nghị định", "Thông tư", "Nghị quyết", "Quyết định"}
MIN_CONTENT_LENGTH = 500  # characters

# Date regex for DD/MM/YYYY format
DATE_RE = re.compile(r'(\d{2})/(\d{2})/(\d{4})')


def parse_date(date_str):
    """Parse DD/MM/YYYY string and return (year, month, day) or None."""
    if not date_str or not isinstance(date_str, str):
        return None
    m = DATE_RE.match(date_str.strip())
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (year, month, day)
    return None


def filter_row(row: dict) -> bool:
    """Check if a single row passes filters. Fast, no pandas needed."""
    # Legal type filter
    if row.get("legal_type") not in LEGAL_TYPES:
        return False

    # Date filter: 2015-2026
    date = parse_date(row.get("issuance_date"))
    if date is None or date[0] < 2015 or date[0] > 2026:
        return False

    # Content length
    content = row.get("content")
    if not content or not isinstance(content, str) or len(content) < MIN_CONTENT_LENGTH:
        return False

    return True


def ingest(
    data_path: str = "data/raw/legal_docs_merged.parquet",
    recreate: bool = False,
    limit: int = None,
    embed_batch_size: int = 32,
    device: str = "cpu",
):
    """
    Run ingestion with STREAMING — pyarrow.dataset batch iteration.

    Memory usage: O(batch_size) ≈ 50-200MB regardless of total dataset size.
    """
    dataset = ds.dataset(data_path, format="parquet")
    print(f"Dataset: ~{dataset.count_rows():,} rows")

    # Init components
    chunker = LegalDocumentChunker()
    embedder = None  # Lazy load (BGE-M3 is heavy — ~2GB)
    store = LegalVectorStore()
    store.create_collection(recreate=recreate)

    total_docs = 0
    total_chunks = 0
    chunk_buffer = []      # (text, dense_vec, sparse_vec, metadata)
    embed_text_buffer = []  # Texts waiting to be embedded

    READ_BATCH = 200  # Read 200 rows at a time from parquet

    print("\nStreaming from parquet...")
    for batch in dataset.to_batches(batch_size=READ_BATCH):
        if limit and total_docs >= limit:
            break

        batch_df = batch.to_pandas()

        for _, row in batch_df.iterrows():
            if limit and total_docs >= limit:
                break

            row_dict = row.to_dict()
            if not filter_row(row_dict):
                continue

            total_docs += 1

            # Chunk this document
            chunks = chunker.chunk_document(row_dict)
            for chunk in chunks:
                total_chunks += 1
                embed_text_buffer.append(chunk.chunk_text)
                chunk_buffer.append({
                    "text": chunk.chunk_text,
                    "metadata": chunk.to_dict(),
                })

            # Embed + index when buffer has enough texts
            if len(embed_text_buffer) >= embed_batch_size * 4:
                if embedder is None:
                    print("Loading BGE-M3 model (this loads ~2GB once)...")
                    embedder = BGEEmbedder(device=device)

                dense, sparse = embedder.encode_documents(
                    embed_text_buffer, batch_size=embed_batch_size
                )
                for i, buf in enumerate(chunk_buffer):
                    buf["dense"] = dense[i].tolist()
                    buf["sparse"] = sparse[i]

                store.upsert_chunks(
                    chunk_texts=[b["text"] for b in chunk_buffer],
                    dense_vectors=[b["dense"] for b in chunk_buffer],
                    sparse_vectors=[b["sparse"] for b in chunk_buffer],
                    metadata_list=[b["metadata"] for b in chunk_buffer],
                )
                chunk_buffer = []
                embed_text_buffer = []

        # Progress
        if total_docs % 500 == 0 and total_docs > 0:
            print(f"  {total_docs:,} docs → {total_chunks:,} chunks")

    # Final flush: embed + index remaining
    if embed_text_buffer:
        if embedder is None:
            print("Loading BGE-M3 model...")
            embedder = BGEEmbedder(device=device)

        dense, sparse = embedder.encode_documents(
            embed_text_buffer, batch_size=embed_batch_size
        )
        for i, buf in enumerate(chunk_buffer):
            buf["dense"] = dense[i].tolist()
            buf["sparse"] = sparse[i]

        store.upsert_chunks(
            chunk_texts=[b["text"] for b in chunk_buffer],
            dense_vectors=[b["dense"] for b in chunk_buffer],
            sparse_vectors=[b["sparse"] for b in chunk_buffer],
            metadata_list=[b["metadata"] for b in chunk_buffer],
        )

    # Verify
    info = store.get_collection_info()
    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Documents: {total_docs:,}")
    print(f"   Chunks:    {total_chunks:,}")
    print(f"   Qdrant:    {info['vectors_count']:,} vectors in '{info['name']}'")


@click.command()
@click.option("--data-path", default="data/raw/legal_docs_merged.parquet")
@click.option("--recreate", is_flag=True, help="Drop and recreate collection")
@click.option("--limit", type=int, default=None, help="Max documents to ingest")
@click.option("--embed-batch-size", type=int, default=32, help="Embedding batch size")
@click.option("--device", default="cpu", help="Device: cpu or cuda")
def main(data_path, recreate, limit, embed_batch_size, device):
    ingest(data_path, recreate, limit, embed_batch_size, device)


if __name__ == "__main__":
    main()

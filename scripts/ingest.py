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
from dotenv import load_dotenv

load_dotenv()  # Load .env for Qdrant Cloud credentials

from src.data.chunker import LegalDocumentChunker
from src.data.chunking.article_level import ArticleLevelChunker
from src.data.contextualizer import build_context_header, build_contextual_text
from src.data.embedder import BGEEmbedder
from src.data.ingest_checkpoint import IngestCheckpoint
from src.db.qdrant_client import LegalVectorStore
from src.data.constants import (
    LEGAL_TYPES,
    MIN_CONTENT_LENGTH,
    DEFAULT_COLLECTION_NAME,
    DATE_FILTER_MIN_YEAR,
    DATE_FILTER_MAX_YEAR,
)


# Date regex for DD/MM/YYYY format
DATE_RE = re.compile(r'(\d{2})/(\d{2})/(\d{4})')

# Amendment law detection: titles containing "sửa đổi" or "bổ sung"
_AMENDMENT_RE = re.compile(r'(?:sửa\s+đổi|bổ\s+sung)', re.IGNORECASE)


def parse_date(date_str):
    """Parse DD/MM/YYYY string and return (year, month, day) or None."""
    if not date_str or not isinstance(date_str, str):
        return None
    m = DATE_RE.match(date_str.strip())
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (year, month, day)
    return None


def filter_row(row: dict, curated: bool = False) -> bool:
    """Check if a single row passes filters. Fast, no pandas needed.

    Args:
        row: Parquet row as a dict.
        curated: True for hand-curated corpora (e.g.
            data/raw/corpus_100docs.parquet, auto-detected in ingest() via
            the presence of a `sector` column). Curated files were already
            manually selected by scripts/select_corpus.py — they
            deliberately include legal_type values outside LEGAL_TYPES
            (e.g. "Nghị định") and don't need the date filter, so both are
            skipped. Only the minimal content guard still applies, since a
            curated corpus must never lose documents silently.
    """
    if not curated:
        # Legal type filter — only process configured types (currently "Luật")
        if row.get("legal_type") not in LEGAL_TYPES:
            return False

        # Date filter (optional — currently disabled for full history)
        if DATE_FILTER_MIN_YEAR is not None or DATE_FILTER_MAX_YEAR is not None:
            date = parse_date(row.get("issuance_date"))
            if date is None:
                return False
            if DATE_FILTER_MIN_YEAR is not None and date[0] < DATE_FILTER_MIN_YEAR:
                return False
            if DATE_FILTER_MAX_YEAR is not None and date[0] > DATE_FILTER_MAX_YEAR:
                return False

    # Content length (always enforced — curated or not)
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
    collection: str = DEFAULT_COLLECTION_NAME,
    quantized: bool = False,
    resume: bool = False,
    model_path: str = None,
):
    """
    Run ingestion with STREAMING — pyarrow.dataset batch iteration.

    Memory usage: O(batch_size) ≈ 50-200MB regardless of total dataset size.
    """
    dataset = ds.dataset(data_path, format="parquet")
    print(f"Dataset: ~{dataset.count_rows():,} rows")

    # Curated corpora (e.g. corpus_100docs.parquet, built by
    # scripts/select_corpus.py) carry a `sector` column that
    # legal_docs_merged.parquet doesn't have. Auto-detect it to switch
    # filter_row() into curated mode (see its docstring).
    curated = "sector" in dataset.schema.names
    if curated:
        print("Detected curated corpus ('sector' column present): "
              "legal_type/date filters disabled, content-length guard only.")

    # Init components
    # S3 Hierarchical: best from Round 1 eval (R@5=97.3%, MRR=0.889)
    chunker = LegalDocumentChunker(min_chunk_size=200, max_chunk_size=1000)
    # S5 Article-level: for amendment laws (different structure — no chapters, quoted replacement text)
    amendment_chunker = ArticleLevelChunker(min_article_length=100)
    embedder = None  # Lazy load (BGE-M3 is heavy — ~2GB)
    store = LegalVectorStore(collection_name=collection, timeout=120)
    store.create_collection(recreate=recreate, quantized=quantized)

    # Checkpoint is always written (so a later run CAN resume), regardless
    # of whether --resume is passed on THIS run. --recreate wipes stale
    # progress from a dropped/recreated collection — see IngestCheckpoint.
    checkpoint = IngestCheckpoint(collection, recreate=recreate)
    if resume:
        already_done = checkpoint.stats()["documents_done"]
        print(f"--resume: {already_done:,} documents already marked done in "
              f"checkpoint '{checkpoint.path}' will be skipped.")

    total_docs = 0
    total_chunks = 0
    skipped_docs = 0
    dropped_curated_docs = 0
    chunk_buffer = []      # (text, dense_vec, sparse_vec, metadata)
    embed_text_buffer = []  # Texts waiting to be embedded
    # (document_id, n_chunks) for documents whose chunks are currently
    # sitting in chunk_buffer — marked done in the checkpoint only after
    # the flush that upserts them succeeds. A document's chunks are never
    # split across two flushes (the buffer check below runs once per
    # document, after all of its chunks are appended), so this list is
    # always safe to mark done wholesale at each flush.
    pending_docs = []

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
            if not filter_row(row_dict, curated=curated):
                if curated:
                    # Curated corpora were hand-selected — a doc getting
                    # dropped here means it failed the content guard, which
                    # should be rare. Never fail silently.
                    dropped_curated_docs += 1
                    print(f"  ⚠️  Dropped curated doc id={row_dict.get('id')} "
                          f"title={str(row_dict.get('title', ''))[:60]!r} "
                          f"(content guard: missing/short content)")
                continue

            doc_id = row_dict.get("id")
            if resume and checkpoint.is_done(doc_id):
                skipped_docs += 1
                continue

            total_docs += 1

            # Detect amendment law → use article-level chunker instead of hierarchical
            title = row_dict.get("title", "")
            is_amend = bool(_AMENDMENT_RE.search(title))

            # Chunk this document (article-level for amendment laws, hierarchical for the rest)
            active_chunker = amendment_chunker if is_amend else chunker
            chunks = active_chunker.chunk_document(row_dict)
            doc_n_chunks = 0
            for chunk in chunks:
                total_chunks += 1
                doc_n_chunks += 1
                meta = chunk.to_dict()
                if is_amend:
                    meta["is_amendment_chunk"] = True
                if curated:
                    meta["sector"] = row_dict.get("sector")
                # Contextual retrieval (rule-based, no LLM — see
                # src/data/contextualizer.py): embed a metadata header +
                # chunk body, but keep the stored chunk_text unchanged.
                meta["context_header"] = build_context_header(meta)
                embed_text_buffer.append(build_contextual_text(meta))
                chunk_buffer.append({
                    "text": chunk.chunk_text,
                    "metadata": meta,
                })
            pending_docs.append((doc_id, doc_n_chunks))

            # Embed + index when buffer has enough texts
            if len(embed_text_buffer) >= embed_batch_size * 4:
                if embedder is None:
                    print("Loading BGE-M3 model (this loads ~2GB once)...")
                    embedder = BGEEmbedder(model_name=model_path or "BAAI/bge-m3", device=device)

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
                    batch_size=25,  # Smaller batch for high-latency connections
                )
                # Upsert succeeded (would have raised otherwise) — safe to
                # record these documents as done.
                checkpoint.mark_done_many(pending_docs)
                chunk_buffer = []
                embed_text_buffer = []
                pending_docs = []

        # Progress
        if total_docs % 500 == 0 and total_docs > 0:
            print(f"  {total_docs:,} docs → {total_chunks:,} chunks")

    # Final flush: embed + index remaining
    if embed_text_buffer:
        if embedder is None:
            print("Loading BGE-M3 model...")
            embedder = BGEEmbedder(model_name=model_path or "BAAI/bge-m3", device=device)

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
            batch_size=25,
        )
        checkpoint.mark_done_many(pending_docs)
        pending_docs = []

    checkpoint.close()

    # Verify
    info = store.get_collection_info()
    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Documents: {total_docs:,}")
    print(f"   Chunks:    {total_chunks:,}")
    if resume:
        print(f"   Skipped (already done, --resume): {skipped_docs:,}")
    if curated and dropped_curated_docs:
        print(f"   ⚠️  Dropped (curated, content guard): {dropped_curated_docs:,}")
    print(f"   Qdrant:    {info['points_count']:,} vectors in '{info['name']}'")


@click.command()
@click.option("--data-path", default="data/raw/legal_docs_merged.parquet")
@click.option("--recreate", is_flag=True, help="Drop and recreate collection")
@click.option("--limit", type=int, default=None, help="Max documents to ingest")
@click.option("--embed-batch-size", type=int, default=32, help="Embedding batch size")
@click.option("--device", default="cpu", help="Device: cpu or cuda")
@click.option("--collection", default=DEFAULT_COLLECTION_NAME,
              help="Tên Qdrant collection đích (mặc định: DEFAULT_COLLECTION_NAME)")
@click.option("--quantized", is_flag=True, default=False,
              help="Bật int8 scalar quantization + fp32 on-disk khi TẠO collection mới")
@click.option("--resume", is_flag=True, default=False,
              help="Bỏ qua các document đã ingest xong theo checkpoint SQLite "
                   "(data/ingest_progress/{collection}.sqlite) — dùng khi chạy "
                   "lại sau khi rớt mạng/session. Checkpoint luôn được ghi dù "
                   "có --resume hay không.")
@click.option("--model-path", default=None,
              help="Đường dẫn thư mục model BGE-M3 local để load offline "
                   "(vd: Kaggle Models mount /kaggle/input/...). Mặc định tải "
                   "HF id BAAI/bge-m3. Kèm HF_HUB_OFFLINE=1 để tránh mọi request mạng.")
def main(data_path, recreate, limit, embed_batch_size, device, collection, quantized, resume, model_path):
    ingest(data_path, recreate, limit, embed_batch_size, device, collection, quantized, resume, model_path)


if __name__ == "__main__":
    main()

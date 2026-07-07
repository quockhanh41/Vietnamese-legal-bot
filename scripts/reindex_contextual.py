"""
Reindex data/raw/luat_law_chunks.json into a NEW Qdrant collection using
rule-based Contextual Retrieval (src/data/contextualizer.py): the text that
gets EMBEDDED carries a metadata header ("which law, which chapter, which
article"), while the stored `chunk_text` payload field stays the original,
unchanged text. No LLM calls — the header is built purely from metadata
already present in the chunk records.

See docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md §3.1.

Usage:
    .venv/bin/python scripts/reindex_contextual.py --recreate
    .venv/bin/python scripts/reindex_contextual.py --collection luat_law_ctx --limit 50
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import click
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.data.contextualizer import build_context_header, build_contextual_texts
from src.data.embedder import BGEEmbedder
from src.db.qdrant_client import LegalVectorStore

DEFAULT_DATA_PATH = "data/raw/luat_law_chunks.json"
DEFAULT_CTX_COLLECTION = "luat_law_ctx"


def load_chunks(data_path: str, limit: int = None):
    with open(data_path, encoding="utf-8") as f:
        chunks = json.load(f)
    if limit:
        chunks = chunks[:limit]
    return chunks


def reindex(
    data_path: str,
    collection: str,
    recreate: bool,
    embed_batch_size: int,
    device: str,
    limit: int = None,
):
    chunks = load_chunks(data_path, limit)
    n = len(chunks)
    print(f"Loaded {n:,} chunks from {data_path}")

    print("Building contextual texts (header + chunk body, no LLM calls)...")
    contextual_texts = build_contextual_texts(chunks)

    print("Loading BGE-M3 model (this loads ~2GB once)...")
    embedder = BGEEmbedder(device=device)

    store = LegalVectorStore(collection_name=collection, timeout=120)
    store.create_collection(recreate=recreate)

    t0 = time.time()
    total_indexed = 0

    for start in tqdm(range(0, n, embed_batch_size), desc="Embedding + upserting"):
        end = min(start + embed_batch_size, n)
        batch_chunks = chunks[start:end]
        batch_texts = contextual_texts[start:end]

        dense, sparse = embedder.encode_documents(batch_texts, batch_size=embed_batch_size)

        metadata_list = []
        for chunk in batch_chunks:
            meta = dict(chunk)  # copy — keep all original fields
            meta.pop("chunk_text", None)  # passed explicitly below instead
            meta["context_header"] = build_context_header(chunk)
            metadata_list.append(meta)

        store.upsert_chunks(
            chunk_texts=[c.get("chunk_text") or "" for c in batch_chunks],
            dense_vectors=[d.tolist() for d in dense],
            sparse_vectors=sparse,
            metadata_list=metadata_list,
            batch_size=embed_batch_size,
        )
        total_indexed += len(batch_chunks)

    elapsed = time.time() - t0
    info = store.get_collection_info()
    print(f"\n{'='*60}")
    print("Reindex complete!")
    print(f"   Chunks embedded+indexed: {total_indexed:,}")
    print(f"   Elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"   Qdrant: {info['points_count']:,} vectors in '{info['name']}'")


@click.command()
@click.option("--data-path", default=DEFAULT_DATA_PATH, help="Path to luat_law_chunks.json")
@click.option("--collection", default=DEFAULT_CTX_COLLECTION, help="Target Qdrant collection name")
@click.option("--recreate", is_flag=True, help="Drop and recreate the collection")
@click.option("--embed-batch-size", type=int, default=8, help="Embedding batch size (safe default for a weak CPU)")
@click.option("--device", default="cpu", help="Device: cpu or cuda")
@click.option("--limit", type=int, default=None, help="Only process the first N chunks (smoke test)")
def main(data_path, collection, recreate, embed_batch_size, device, limit):
    reindex(data_path, collection, recreate, embed_batch_size, device, limit)


if __name__ == "__main__":
    main()

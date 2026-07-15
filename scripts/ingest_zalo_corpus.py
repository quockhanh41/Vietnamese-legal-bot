"""
Embed + index the Zalo 2021 legal corpus into a LOCAL Qdrant collection
(M3.1). Two modes so the GPU-heavy part can run on Kaggle while the local
docker Qdrant stays reachable only from this machine:

1) Embed to parquet shards (run on Kaggle GPU, or locally on MPS overnight):
   .venv/bin/python scripts/ingest_zalo_corpus.py embed \
       --corpus data/benchmarks/zalo/legal_corpus.json \
       --out-dir data/benchmarks/zalo/vectors \
       --device cuda --batch-size 128 --shard-size 2000
   Resume-safe: existing shard files are skipped, so rerunning after a
   session drop only embeds missing shards. On Kaggle: clone repo, `pip
   install -q -r requirements-ingest.txt`, attach legal_corpus.json as a
   dataset input, download the out-dir as output afterwards (same pattern
   as docs/notes/kaggle-ingest-guide.md).

2) Upsert shards into local Qdrant (run locally, docker compose up qdrant):
   .venv/bin/python scripts/ingest_zalo_corpus.py upsert \
       --shards "data/benchmarks/zalo/vectors/vectors_*.parquet" \
       --collection zalo_legal_2021 --qdrant-url http://localhost:6333

Production Qdrant Cloud is never touched: --qdrant-url is explicit and
defaults to localhost.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_zalo import load_zalo_corpus

EXPECTED_POINTS = 61425


def flatten_corpus(corpus):
    """One row per article. embed_text = title + newline + text (the title
    carries the article heading — cheap context, analogous to the contextual
    prefix used in production ingest). Metadata per the Task 4 payload
    contract (see plan)."""
    rows = []
    for law in corpus:
        for art in law["articles"]:
            rows.append({
                "embed_text": f"{art['title']}\n{art['text']}",
                "metadata": {
                    "document_number": law["law_id"],
                    "article": art["article_id"],
                    "document_title": art["title"],
                    "document_id": law["law_id"],
                    "chunk_index": art["article_id"],
                },
            })
    return rows


def cmd_embed(args):
    import pandas as pd
    from src.data.embedder import BGEEmbedder

    rows = flatten_corpus(load_zalo_corpus(args.corpus))
    print(f"{len(rows)} articles to embed")
    os.makedirs(args.out_dir, exist_ok=True)
    embedder = BGEEmbedder(device=args.device)

    n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    for shard_idx in range(n_shards):
        shard_path = os.path.join(args.out_dir, f"vectors_{shard_idx:04d}.parquet")
        if os.path.exists(shard_path):
            print(f"shard {shard_idx:04d}: exists, skipping")
            continue
        shard = rows[shard_idx * args.shard_size:(shard_idx + 1) * args.shard_size]
        dense, lexical = embedder.encode_documents(
            [r["embed_text"] for r in shard], batch_size=args.batch_size)
        pd.DataFrame({
            "embed_text": [r["embed_text"] for r in shard],
            "metadata": [json.dumps(r["metadata"], ensure_ascii=False) for r in shard],
            "dense": [d.tolist() for d in dense],
            # lexical_weights keys pass through json (str keys) — qdrant-client
            # coerces numeric-string indices to int, same as the prod path
            # where ingest.py hands lexical_weights straight to upsert_chunks.
            "sparse": [json.dumps({str(k): float(v) for k, v in lw.items()})
                       for lw in lexical],
        }).to_parquet(shard_path, index=False)
        print(f"shard {shard_idx:04d}: wrote {len(shard)} rows")
    print("✅ embed done")


def cmd_upsert(args):
    import pandas as pd
    from src.db.qdrant_client import LegalVectorStore

    store = LegalVectorStore(url=args.qdrant_url, api_key="",
                             collection_name=args.collection)
    store.create_collection()  # no quantization: local docker, RAM is not scarce

    shard_paths = sorted(glob.glob(args.shards))
    if not shard_paths:
        sys.exit(f"No shards match {args.shards}")
    total = 0
    for path in shard_paths:
        df = pd.read_parquet(path)
        store.upsert_chunks(
            chunk_texts=df["embed_text"].tolist(),
            dense_vectors=[list(v) for v in df["dense"]],
            sparse_vectors=[json.loads(s) for s in df["sparse"]],
            metadata_list=[json.loads(m) for m in df["metadata"]],
        )
        total += len(df)
        print(f"{os.path.basename(path)}: upserted {len(df)} (running total {total})")

    info = store.get_collection_info()
    print(f"Collection '{args.collection}': {info['points_count']:,} points")
    if info["points_count"] != EXPECTED_POINTS:
        sys.exit(f"❌ expected {EXPECTED_POINTS:,} points, got {info['points_count']:,}")
    print("✅ point count matches corpus")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--corpus", default="data/benchmarks/zalo/legal_corpus.json")
    e.add_argument("--out-dir", default="data/benchmarks/zalo/vectors")
    e.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--shard-size", type=int, default=2000)
    e.set_defaults(func=cmd_embed)

    u = sub.add_parser("upsert")
    u.add_argument("--shards", default="data/benchmarks/zalo/vectors/vectors_*.parquet")
    u.add_argument("--collection", default="zalo_legal_2021")
    u.add_argument("--qdrant-url", default="http://localhost:6333")
    u.set_defaults(func=cmd_upsert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

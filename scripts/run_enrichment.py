"""M2 Phase E — run Phase-2 enrichment on a collection (Kaggle-friendly).

Reads chunks FROM Qdrant, builds graph/timeline/amendment into the SQLite
MetadataStore, and backfills is_in_effect into the collection's payloads.
Run this in the Kaggle notebook right after ingest (avoids scrolling ~140K
chunks over a home connection), then download the metadata dir back to repo.

Usage:
    python scripts/run_enrichment.py --collection luat_law_ctx_v4
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()
import click
from src.db.qdrant_client import LegalVectorStore
from src.data.metadata_store import MetadataStore
from src.pipeline.enricher import EnrichmentPipeline


@click.command()
@click.option("--collection", required=True, help="Qdrant collection to enrich (e.g. luat_law_ctx_v4)")
@click.option("--metadata-dir", default="data/metadata")
def main(collection, metadata_dir):
    store = LegalVectorStore(collection_name=collection)
    meta = MetadataStore(data_dir=metadata_dir)
    result = EnrichmentPipeline(vector_store=store, metadata_store=meta).run()
    print(f"docs_processed={result.docs_processed}/{result.total_docs}  "
          f"base={result.base_laws} amend={result.amendment_laws}  "
          f"graph={result.graph_nodes}n/{result.graph_edges}e  "
          f"in_effect={result.currently_in_effect}  "
          f"elapsed={result.elapsed_seconds:.0f}s")
    for e in result.errors:
        print("ERROR:", e)


if __name__ == "__main__":
    main()

"""
Backfill is_in_effect onto chunk payloads of an existing collection, from the
MetadataStore. Idempotent, additive. Only writes docs with a KNOWN status
(True/False); None (unknown) is left unset so the server filter keeps it.

This is the standalone/one-off counterpart to
EnrichmentPipeline._backfill_is_in_effect() (src/pipeline/enricher.py), which
runs the same backfill automatically at the end of every enrichment run.
Use this script to backfill a collection that was enriched BEFORE that wiring
existed (e.g. an already-live collection), without re-running enrichment.

Usage: .venv/bin/python scripts/backfill_is_in_effect.py --collection luat_law_ctx_v3
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()
from src.db.qdrant_client import LegalVectorStore
from src.data.metadata_store import MetadataStore

def main(collection, metadata_dir):
    store = LegalVectorStore(collection_name=collection)
    meta = MetadataStore(data_dir=metadata_dir)
    n = 0
    for doc_id, m in meta.get_all_documents().items():
        if m.is_in_effect is None:
            continue
        store.set_is_in_effect(doc_id, bool(m.is_in_effect))
        n += 1
    print(f"Backfilled is_in_effect on {n} documents in '{collection}'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION"))
    ap.add_argument("--metadata-dir", default="data/metadata")
    a = ap.parse_args()
    main(a.collection, a.metadata_dir)

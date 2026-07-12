"""
is_in_effect server filter — real Qdrant round-trip on an EPHEMERAL collection.
Skips when QDRANT_URL is unset. Never touches luat_law_ctx_v3.
Run: .venv/bin/python -m pytest tests/test_is_in_effect_filter_integration.py -q
"""
import os, sys, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from dotenv import load_dotenv; load_dotenv()

pytestmark = pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="no Qdrant")

def test_exclude_superseded_keeps_true_and_absent():
    from qdrant_client.models import PointStruct, SparseVector
    from src.db.qdrant_client import LegalVectorStore
    name = f"eph_{uuid.uuid4().hex[:8]}"
    s = LegalVectorStore(collection_name=name)
    s.create_collection()
    try:
        vec = lambda x: [float(x)] * s.VECTOR_DIM
        pts = [
            PointStruct(id=str(uuid.uuid4()), vector={"dense": vec(1),
                "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={"chunk_text": "keep-true", "document_id": 1, "is_in_effect": True}),
            PointStruct(id=str(uuid.uuid4()), vector={"dense": vec(1),
                "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={"chunk_text": "drop-false", "document_id": 2, "is_in_effect": False}),
            PointStruct(id=str(uuid.uuid4()), vector={"dense": vec(1),
                "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={"chunk_text": "keep-absent", "document_id": 3}),
        ]
        s.client.upsert(collection_name=name, points=pts, wait=True)
        hits = s.search_dense(vec(1), top_k=10, exclude_superseded=True)
        texts = {h.chunk_text for h in hits}
        assert "keep-true" in texts and "keep-absent" in texts
        assert "drop-false" not in texts
        # backfill flips doc 3 to superseded, then it drops too
        s.set_is_in_effect(3, False)
        hits2 = s.search_dense(vec(1), top_k=10, exclude_superseded=True)
        assert "keep-absent" not in {h.chunk_text for h in hits2}
    finally:
        s.client.delete_collection(name)

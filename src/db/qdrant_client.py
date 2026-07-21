"""
Qdrant vector database client for hybrid search (dense + sparse).

Compatible with Qdrant >= 1.18.x (query_points API).
"""

import hashlib
import os
import uuid
import time
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.data.constants import DEFAULT_COLLECTION_NAME
from src.eval.metrics import artnum
from src.observability import observe

# Fixed namespace for uuid5-derived point IDs (Việc 1 — deterministic point
# IDs so re-running ingest is idempotent instead of duplicating points on
# every run). Any valid UUID works as a namespace; this one was generated
# once via uuid.uuid4() and is now frozen — DO NOT regenerate it, doing so
# would change every point ID ever derived from it.
_POINT_ID_NAMESPACE = uuid.UUID("2b9e6f3a-8c1d-4a7e-9f2b-3d5c6e7a8b9c")


def compute_point_id(metadata: Dict, chunk_text: str) -> str:
    """
    Deterministic point ID (uuid5) for a chunk, so upserting the same chunk
    twice (resuming after a crash, re-running ingest/reindex) overwrites the
    same Qdrant point instead of creating a duplicate.

    Identity is derived from the chunk's OWN metadata — never from its
    position within the current upsert_chunks() call/batch. Batches can mix
    chunks from multiple documents (scripts/reindex_contextual.py upserts
    fixed-size slices that cross document boundaries), so an
    enumerate()-based index would not be stable across runs.

    Preferred key: f"{document_id}:{chunk_index}", where chunk_index is this
    chunk's 0-based position within ITS document (set by the chunker — see
    LegalChunk.chunk_index in src/data/chunker.py). Falls back to a content
    hash of the chunk text when chunk_index is absent from metadata (e.g.
    legacy payloads ingested before this field existed).

    Caveat: determinism only holds within a fixed chunker/config version —
    changing chunking strategy or parameters shifts chunk boundaries (and
    therefore these IDs). That's expected: this project already versions
    collections per chunking strategy (see DEFAULT_COLLECTION_NAME) rather
    than mutating an existing collection's chunk boundaries in place.
    """
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        key = f"{document_id}:{chunk_index}"
    else:
        content_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
        key = f"{document_id}:{content_hash}"
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, key))


@dataclass
class SearchResult:
    """Structured search result."""
    point_id: str
    score: float
    chunk_text: str
    document_title: str
    document_number: str
    legal_type: str
    article: Optional[str] = None
    chapter: Optional[str] = None
    url: Optional[str] = None
    document_id: Optional[int] = None
    parent_chunk_text: Optional[str] = None


class LegalVectorStore:
    """Qdrant vector store specialized for Vietnamese legal documents."""

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    VECTOR_DIM = 1024  # BGE-M3

    # M2.3: fields projected by get_document_index()'s scroll — doc-level
    # metadata only, deliberately excludes chunk_text (that's the whole
    # point: O(docs) memory instead of O(chunks)).
    _DOC_INDEX_FIELDS = ["document_id", "document_title", "document_number", "legal_type", "sector"]

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        timeout: int = 60,
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        # Suppress warning for local dev (no auth on localhost)
        if self.url.startswith("http://") and not self.api_key:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=timeout)
        else:
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=timeout)

    def create_collection(self, recreate: bool = False, quantized: bool = False):
        """Create the collection with dense + sparse vector support.

        quantized=True enables int8 scalar quantization (kept in RAM) with
        the original fp32 vectors on disk — ~4x less cluster RAM, which is
        what keeps free-tier 1GB Qdrant Cloud viable as the corpus scales
        (Plan Giai Đoạn Cuối §2.2). Must be chosen at creation time:
        switching later means re-indexing the whole collection.
        """
        if recreate:
            self.client.delete_collection(self.collection_name)

        existing = self.client.get_collections()
        if self.collection_name in [c.name for c in existing.collections]:
            print(f"Collection '{self.collection_name}' already exists.")
            return

        quantization_config = None
        dense_on_disk = None
        if quantized:
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
            dense_on_disk = True

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE,
                    on_disk=dense_on_disk,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
            quantization_config=quantization_config,
        )
        self._create_payload_indexes()
        print(f"Created collection '{self.collection_name}'")

    def _create_payload_indexes(self):
        """
        Create payload indexes for fields used in filtered/scoped search.

        Only called from the "collection didn't exist yet" branch of
        create_collection() above — that method early-returns when the
        collection already exists, so this never re-runs against a
        collection that already has these indexes. That makes
        create_collection() idempotent as a whole (safe to call repeatedly).

        is_in_effect is not written into any chunk payload yet (that lands
        in a later milestone — temporal/amendment wiring), but Qdrant allows
        indexing a field before any point sets it, so we create the index
        now (forward-looking) rather than needing a second migration later.
        """
        index_fields = [
            ("document_id", models.PayloadSchemaType.INTEGER),
            ("article_number", models.PayloadSchemaType.KEYWORD),
            ("legal_type", models.PayloadSchemaType.KEYWORD),
            ("issuance_date", models.PayloadSchemaType.KEYWORD),
            ("is_in_effect", models.PayloadSchemaType.BOOL),
            # M1.2: curated corpora (e.g. corpus_100docs.parquet) carry a
            # `sector` label (hinh_su/dan_su/lao_dong/doanh_nghiep/dat_dai/
            # hngd) for per-sector filtering/eval. Older, non-curated
            # collections simply never set this field on their points —
            # indexing it before any point sets it is fine (Qdrant allows
            # that), matching the is_in_effect precedent above.
            ("sector", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in index_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    def upsert_chunks(
        self,
        chunk_texts: List[str],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[int, float]],
        metadata_list: List[Dict],
        batch_size: int = 100,
    ):
        """
        Index chunks into Qdrant with both dense and sparse vectors.

        Args:
            chunk_texts: List of chunk text content
            dense_vectors: Dense embeddings (BGE-M3)
            sparse_vectors: Sparse/lexical weights (BGE-M3)
            metadata_list: List of metadata dicts (must include document_title, etc.)
            batch_size: Upsert batch size
        """
        points = []
        for i in range(len(chunk_texts)):
            point_id = compute_point_id(metadata_list[i], chunk_texts[i])
            payload = {
                "chunk_text": chunk_texts[i],
                **metadata_list[i],
            }
            # Derived keyword field for article-scoped filtering (Việc 2).
            # Vietnamese law articles can have letter suffixes (Điều 123a,
            # 51b) — artnum() keeps those, so this MUST stay a string, never
            # cast to int. Kept alongside `article` (payload.get above)
            # rather than replacing it, matching the existing pattern of
            # always-present-but-possibly-None metadata fields in this dict.
            payload["article_number"] = artnum(payload.get("article"))

            # Convert sparse dict to Qdrant SparseVector format
            indices = list(sparse_vectors[i].keys())
            values = list(sparse_vectors[i].values())

            point = PointStruct(
                id=point_id,
                vector={
                    self.DENSE_VECTOR_NAME: dense_vectors[i],
                    self.SPARSE_VECTOR_NAME: SparseVector(
                        indices=indices,
                        values=values,
                    ),
                },
                payload=payload,
            )
            points.append(point)

            # Batch upsert with retry
            if len(points) >= batch_size:
                self._upsert_with_retry(points)
                points = []

        # Final batch
        if points:
            self._upsert_with_retry(points)

        print(f"Indexed {len(chunk_texts)} chunks")

    def _upsert_with_retry(self, points: List[PointStruct], max_retries: int = 3):
        """Upsert points with exponential backoff for cloud resilience."""
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_s = 2 ** attempt  # 1s, 2s, 4s
                print(f"  ⚠️  Upsert failed (attempt {attempt+1}/{max_retries}), "
                      f"retrying in {wait_s}s... ({e})")
                time.sleep(wait_s)

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter_conditions: Optional[Dict] = None,
        exclude_superseded: bool = False,
    ) -> List[SearchResult]:
        """Pure dense (semantic) search."""
        query_filter = self._build_filter(filter_conditions, exclude_superseded=exclude_superseded)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [self._to_search_result(p) for p in response.points]

    def search_sparse(
        self,
        sparse_vector: Dict[int, float],
        top_k: int = 20,
        filter_conditions: Optional[Dict] = None,
        exclude_superseded: bool = False,
    ) -> List[SearchResult]:
        """Pure sparse (BM25-like) search."""
        query_filter = self._build_filter(filter_conditions, exclude_superseded=exclude_superseded)

        indices = list(sparse_vector.keys())
        values = list(sparse_vector.values())

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=indices, values=values),
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [self._to_search_result(p) for p in response.points]

    @observe
    def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        top_k: int = 20,
        fusion_strategy: str = "rrf",
        filter_conditions: Optional[Dict] = None,
        exclude_superseded: bool = False,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense + sparse via Reciprocal Rank Fusion (RRF).

        Uses Qdrant's native FusionQuery with server-side RRF (Qdrant >= 1.18.x),
        which is more efficient than client-side fusion.

        Args:
            dense_vector: BGE-M3 dense embedding
            sparse_vector: BGE-M3 sparse/lexical weights
            top_k: Number of results to return
            fusion_strategy: 'rrf' (default) — server-side Reciprocal Rank Fusion
            filter_conditions: Optional metadata filter dict
            exclude_superseded: When True, drops chunks whose document is known
                superseded (is_in_effect == False). Docs with unknown status
                (field absent/None) are kept — fail-open (see _build_filter).

        Returns:
            Fused, deduplicated search results
        """
        query_filter = self._build_filter(filter_conditions, exclude_superseded=exclude_superseded)

        indices = list(sparse_vector.keys())
        values = list(sparse_vector.values())

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.NearestQuery(nearest=dense_vector),
                    using=self.DENSE_VECTOR_NAME,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=SparseVector(indices=indices, values=values),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [self._to_search_result(p) for p in response.points]

    def _build_filter(
        self,
        conditions: Optional[Dict] = None,
        exclude_superseded: bool = False,
    ) -> Optional[Filter]:
        """Build a Qdrant filter from a conditions dict, optionally excluding
        superseded chunks server-side.

        Example: {"legal_type": "Luật", "issuing_authority": "Quốc hội"}

        exclude_superseded adds a must_not on is_in_effect == False. This
        keeps points where is_in_effect is True OR absent/None (fail-open:
        docs whose status is still unknown are not silently dropped) — only
        chunks explicitly known superseded are excluded.
        """
        must = []
        for key, value in (conditions or {}).items():
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        must_not = []
        if exclude_superseded:
            must_not.append(FieldCondition(key="is_in_effect", match=MatchValue(value=False)))
        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)

    def set_is_in_effect(self, doc_id: int, value: bool) -> None:
        """Backfill the is_in_effect flag onto every chunk payload of `doc_id`
        so exclude_superseded=True searches can filter it server-side.
        """
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_in_effect": value},
            points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
            wait=True,
        )

    def _to_search_result(self, point: ScoredPoint) -> SearchResult:
        """Convert ScoredPoint to SearchResult."""
        payload = point.payload or {}
        return SearchResult(
            point_id=point.id,
            score=point.score,
            chunk_text=payload.get("chunk_text", ""),
            document_title=payload.get("document_title", ""),
            document_number=payload.get("document_number", ""),
            legal_type=payload.get("legal_type", ""),
            article=payload.get("article"),
            chapter=payload.get("chapter"),
            url=payload.get("url"),
            document_id=payload.get("document_id"),
            parent_chunk_text=payload.get("parent_chunk_text"),
        )

    def get_document_index(self) -> Dict[int, Dict]:
        """One projected doc-level scroll (no chunk_text). {doc_id: {title,number,type,sector}}."""
        index: Dict[int, Dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name, limit=2000, offset=offset,
                with_payload=self._DOC_INDEX_FIELDS, with_vectors=False)
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("document_id")
                if doc_id is None or doc_id in index:
                    continue
                index[doc_id] = {k: payload.get(k) for k in
                                 ("document_title", "document_number", "legal_type", "sector")}
            if offset is None:
                break
        return index

    def scroll_article(self, doc_ids: List[int], article_number: str, cap: int = 8) -> List[SearchResult]:
        """Server-side filtered scroll: document_id in doc_ids AND article_number == value."""
        must = [FieldCondition(key="article_number", match=MatchValue(value=article_number))]
        if doc_ids:
            must.append(FieldCondition(key="document_id", match=models.MatchAny(any=list(doc_ids))))
        points, _ = self.client.scroll(
            collection_name=self.collection_name, scroll_filter=Filter(must=must),
            limit=cap, with_payload=True, with_vectors=False)
        return [self._point_to_result(p) for p in points]

    def scroll_docs(self, doc_ids: List[int], limit_per: int = 512):
        """Raw points for the lenient fallback (keeps chunk_text/article), filtered by document_id."""
        if not doc_ids:
            flt = None
        else:
            flt = Filter(must=[FieldCondition(key="document_id", match=models.MatchAny(any=list(doc_ids)))])
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name, scroll_filter=flt,
                limit=limit_per, offset=offset, with_payload=True, with_vectors=False)
            out.extend(points)
            if offset is None:
                break
        return out

    def _point_to_result(self, point) -> SearchResult:
        payload = point.payload or {}
        return SearchResult(
            point_id=str(point.id), score=1.0, chunk_text=payload.get("chunk_text", ""),
            document_title=payload.get("document_title", ""),
            document_number=payload.get("document_number", ""),
            legal_type=payload.get("legal_type", ""), article=payload.get("article"),
            chapter=payload.get("chapter"), url=payload.get("url"),
            document_id=payload.get("document_id"),
            parent_chunk_text=payload.get("parent_chunk_text"))

    def list_documents(
        self,
        legal_type: Optional[str] = None,
        keyword: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict:
        """
        List/search indexed documents with pagination and filtering.

        Uses Qdrant scroll with filters instead of a separate document store.
        Deduplicates by document_id to show unique documents.
        """
        # Build filter
        query_filter = None
        conditions = {}
        if legal_type:
            conditions["legal_type"] = legal_type
        if conditions:
            query_filter = self._build_filter(conditions)

        # Scroll to find unique documents (dedup by document_id)
        seen_ids = set()
        documents = []
        offset = None

        # Need enough to satisfy the page
        target = page * page_size * 3  # Fetch extra since we dedup

        while len(documents) < target:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=min(500, target),
                offset=offset,
                with_payload=True,
            )
            if not points:
                break

            for p in points:
                doc_id = p.payload.get("document_id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc = {
                        "document_id": doc_id,
                        "title": p.payload.get("document_title", ""),
                        "legal_type": p.payload.get("legal_type", ""),
                        "issuance_date": p.payload.get("issuance_date", ""),
                        "issuing_authority": p.payload.get("issuing_authority", ""),
                        "document_number": p.payload.get("document_number", ""),
                        "url": p.payload.get("url", ""),
                    }
                    # Keyword filter (client-side, simple)
                    if keyword:
                        if keyword.lower() in doc["title"].lower():
                            documents.append(doc)
                    else:
                        documents.append(doc)

            if offset is None:
                break

        # Paginate
        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        paged = documents[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "documents": paged,
        }

    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
        }

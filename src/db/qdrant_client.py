"""
Qdrant vector database client for hybrid search (dense + sparse).

Compatible with Qdrant >= 1.18.x (query_points API).
"""

import os
import uuid
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


class LegalVectorStore:
    """Qdrant vector store specialized for Vietnamese legal documents."""

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    VECTOR_DIM = 1024  # BGE-M3

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "vietnamese_laws",
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        # Suppress warning for local dev (no auth on localhost)
        if self.url.startswith("http://") and not self.api_key:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def create_collection(self, recreate: bool = False):
        """Create the collection with dense + sparse vector support."""
        if recreate:
            self.client.delete_collection(self.collection_name)

        existing = self.client.get_collections()
        if self.collection_name in [c.name for c in existing.collections]:
            print(f"Collection '{self.collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        print(f"Created collection '{self.collection_name}'")

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
            point_id = str(uuid.uuid4())
            payload = {
                "chunk_text": chunk_texts[i],
                **metadata_list[i],
            }

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

            # Batch upsert
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                points = []

        # Final batch
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        print(f"Indexed {len(chunk_texts)} chunks")

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter_conditions: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Pure dense (semantic) search."""
        query_filter = self._build_filter(filter_conditions) if filter_conditions else None

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
    ) -> List[SearchResult]:
        """Pure sparse (BM25-like) search."""
        query_filter = self._build_filter(filter_conditions) if filter_conditions else None

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

    def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        top_k: int = 20,
        fusion_strategy: str = "rrf",
        filter_conditions: Optional[Dict] = None,
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

        Returns:
            Fused, deduplicated search results
        """
        query_filter = self._build_filter(filter_conditions) if filter_conditions else None

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

    def _build_filter(self, conditions: Dict) -> Filter:
        """Build Qdrant filter from conditions dict.
        Example: {"legal_type": "Luật", "issuing_authority": "Quốc hội"}
        """
        must_conditions = []
        for key, value in conditions.items():
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )
        return Filter(must=must_conditions)

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
        )

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

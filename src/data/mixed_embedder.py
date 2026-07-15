"""Mixed-hybrid embedder: dense from a SentenceTransformer, sparse from BGE-M3.

AITeamVN/Vietnamese_Embedding is a dense-only SentenceTransformer fine-tune of
BGE-M3 (no sparse/ColBERT head). To keep the production hybrid + routed paths
working we take its dense vectors and BGE-M3's lexical (sparse) weights, and
expose the SAME interface as BGEEmbedder so eval.py, ingest.py, and the
orchestrator use it unchanged.
"""

import numpy as np
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
from src.data.embedder import BGEEmbedder


class MixedHybridEmbedder:
    def __init__(
        self,
        dense_model: str = "AITeamVN/Vietnamese_Embedding",
        sparse_model: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        self._dense = SentenceTransformer(dense_model, device=device)
        self._sparse = BGEEmbedder(model_name=sparse_model, device=device)
        self.dim = self._dense.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict:
        result: Dict = {}
        if return_dense:
            # normalize: AITeamVN uses dot-product similarity; normalized dense
            # makes dot == cosine, matching the v4 collection's distance metric.
            result["dense_vecs"] = self._dense.encode(
                texts, batch_size=batch_size, normalize_embeddings=True
            )
        if return_sparse:
            out = self._sparse.encode(
                texts, batch_size=batch_size, return_dense=False, return_sparse=True
            )
            result["lexical_weights"] = out["lexical_weights"]
        return result

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self._dense.encode(queries, normalize_embeddings=True)

    def encode_documents(
        self, documents: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        out = self.encode(
            documents, batch_size=batch_size, return_dense=True, return_sparse=True
        )
        return out["dense_vecs"], out["lexical_weights"]

    @property
    def vector_size(self) -> int:
        return self.dim


def build_embedder(embedder_type: str, embed_model: str = None, device: str = "cpu"):
    """Factory used by ingest.py and eval.py. Default 'bge-m3' = current behavior."""
    if embedder_type == "mixed":
        return MixedHybridEmbedder(
            dense_model=embed_model or "AITeamVN/Vietnamese_Embedding",
            sparse_model="BAAI/bge-m3",
            device=device,
        )
    if embedder_type == "bge-m3":
        return BGEEmbedder(model_name=embed_model or "BAAI/bge-m3", device=device)
    raise ValueError(f"unknown embedder_type: {embedder_type!r}")

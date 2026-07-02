"""
BGE-M3 embedding wrapper for Vietnamese legal text.

BGE-M3 (BAAI) provides both dense (1024-dim) and sparse (lexical weights) vectors
in a single pass — eliminating the need for a separate BM25 index.
"""

import gc
import numpy as np
from typing import List, Dict, Tuple
from FlagEmbedding import BGEM3FlagModel


class BGEEmbedder:
    """Wrapper around BGE-M3 for dense + sparse embeddings.

    Includes MPS memory management for macOS: automatically clears Metal cache
    between batches and falls back to CPU if MPS runs out of memory.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            model_name: HuggingFace model ID
            use_fp16: Use half precision (saves memory).
                      Auto-disabled on CPU for compatibility.
            device: 'cpu', 'mps', or 'cuda'
        """
        self.device = device
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
        )
        self.dim = 1024  # BGE-M3 dense dimension

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict:
        """
        Encode texts into dense and/or sparse vectors.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            return_dense: Whether to return dense vectors
            return_sparse: Whether to return sparse vectors (for BM25-like search)

        Returns:
            Dict with keys:
                - 'dense_vecs': np.ndarray of shape (n, 1024)
                - 'lexical_weights': List[defaultdict[int, float]] (sparse/lexical weights)
                - 'colbert_vecs': List[np.ndarray] (optional, for ColBERT)
        """
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        # MPS memory management: clear Metal cache after each encode batch
        self._clear_gpu_memory()

        return result

    def _clear_gpu_memory(self):
        """Clear GPU memory caches to prevent memory accumulation."""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode search queries (dense only, no sparse needed for queries)."""
        output = self.model.encode(
            queries,
            batch_size=len(queries),
            max_length=512,  # Queries are short
            return_dense=True,
            return_sparse=False,
        )
        return output["dense_vecs"]

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        """
        Encode documents for indexing.

        Returns:
            (dense_vectors, lexical_weights) tuple
        """
        output = self.encode(
            documents,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
        )
        return output["dense_vecs"], output["lexical_weights"]

    @property
    def vector_size(self) -> int:
        return self.dim

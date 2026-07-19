"""
RAGAS evaluation runner for Vietnamese legal RAG pipeline.

Provides a clean wrapper around RAGAS metrics (faithfulness, context_precision,
context_recall, answer_relevancy) with our LLMGateway as the judge LLM.

Handles the langchain_community.vertexai import issue via monkey-patching.
"""

import sys
from typing import List, Dict, Optional

# Monkey-patch: ragas imports langchain_community.chat_models.vertexai which
# no longer exists in newer langchain-community. Inject the module from the
# standalone langchain-google-vertexai package.
_VERTEXAI_PATCHED = False


def _patch_vertexai():
    global _VERTEXAI_PATCHED
    if _VERTEXAI_PATCHED:
        return
    try:
        from langchain_google_vertexai import ChatVertexAI
        vertexai_module = type(sys)("langchain_community.chat_models.vertexai")
        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module
        _VERTEXAI_PATCHED = True
    except ImportError:
        # If vertexai is not installed, ragas might still work without it
        pass


_patch_vertexai()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from ragas.dataset_schema import SingleTurnSample

# Re-export for convenience
__all__ = [
    "RagasEvaluator",
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
]


class RagasEvaluator:
    """
    Wrapper around RAGAS evaluate() that uses our LLMGateway as the judge.

    Usage:
        evaluator = RagasEvaluator(llm_gateway)
        metrics = evaluator.evaluate_sample(
            question="...",
            answer="...",
            contexts=["chunk1", "chunk2"],
            ground_truth="...",  # optional
        )
    """

    def __init__(self, llm_gateway):
        """
        Args:
            llm_gateway: LLMGateway instance used as judge for RAGAS metrics.
        """
        self.llm = llm_gateway
        self.metrics = [
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
        ]

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG sample.

        Args:
            question: User question
            answer: Generated answer from the RAG pipeline
            contexts: List of retrieved chunk texts
            ground_truth: Optional ground truth answer (for context_recall)

        Returns:
            Dict mapping metric name to score (0.0-1.0)
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth or "",
        )

        result = evaluate(
            dataset=[sample],
            metrics=self.metrics,
            llm=self._get_judge_llm(),
        )

        return self._extract_scores(result)

    def evaluate_batch(
        self,
        samples: List[Dict],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple RAG samples.

        Args:
            samples: List of dicts with keys:
                     question, answer, contexts, ground_truth (optional)

        Returns:
            List of metric score dicts, one per sample
        """
        ragas_samples = []
        for s in samples:
            ragas_samples.append(
                SingleTurnSample(
                    user_input=s["question"],
                    response=s["answer"],
                    retrieved_contexts=s["contexts"],
                    reference=s.get("ground_truth", ""),
                )
            )

        result = evaluate(
            dataset=ragas_samples,
            metrics=self.metrics,
            llm=self._get_judge_llm(),
        )

        return self._extract_batch_scores(result, len(samples))

    def _get_judge_llm(self):
        """Create a LangChain-compatible LLM wrapper from our LLMGateway."""
        from langchain_core.language_models.llms import LLM
        from typing import Any, Iterator

        gateway = self.llm

        class GatewayLLM(LLM):
            """LangChain-compatible wrapper around our LLMGateway."""

            model_name: str = "groq/llama-3.3-70b-versatile"

            def _call(
                self,
                prompt: str,
                stop: Any | None = None,
                **kwargs: Any,
            ) -> str:
                return gateway.generate(prompt=prompt, temperature=0.1)

            @property
            def _llm_type(self) -> str:
                return "gateway_llm"

        return GatewayLLM()

    @staticmethod
    def _extract_scores(result) -> Dict[str, float]:
        """Extract metric scores from RAGAS result object."""
        scores = {}
        result_df = result.to_pandas()
        for col in result_df.columns:
            if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                val = result_df[col].iloc[0]
                scores[col] = float(val) if val is not None else 0.0
        return scores

    @staticmethod
    def _extract_batch_scores(result, n_samples: int) -> List[Dict[str, float]]:
        """Extract metric scores from batch RAGAS result."""
        result_df = result.to_pandas()
        metric_cols = [
            c for c in result_df.columns
            if c not in ("user_input", "response", "retrieved_contexts", "reference")
        ]

        scores_list = []
        for i in range(n_samples):
            sample_scores = {}
            for col in metric_cols:
                val = result_df[col].iloc[i] if i < len(result_df) else None
                sample_scores[col] = float(val) if val is not None else 0.0
            scores_list.append(sample_scores)

        return scores_list

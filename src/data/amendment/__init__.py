"""
Amendments Tracking — Detect and parse amendment laws for Vietnamese legislation.

Module 3 of Phase 2: Identify "Luật sửa đổi, bổ sung" documents, extract
amendment instructions (MODIFY, INSERT, REPEAL, REPLACE_WORD), resolve
target laws, and enable consolidated text generation.

Usage:
    from src.data.amendment import AmendmentExtractor, AmendmentConsolidator

    extractor = AmendmentExtractor()
    instructions = extractor.extract(content=text, doc_id=doc_id, title=title)

    consolidator = AmendmentConsolidator()
    amended_text = consolidator.apply(base_text, instructions)
"""

from src.data.amendment.amendment_extractor import (
    AmendmentExtractor,
    AmendmentInstruction,
    AmendmentAction,
)
from src.data.amendment.amendment_consolidator import (
    AmendmentConsolidator,
    ConsolidationResult,
)

__all__ = [
    "AmendmentExtractor",
    "AmendmentInstruction",
    "AmendmentAction",
    "AmendmentConsolidator",
    "ConsolidationResult",
]

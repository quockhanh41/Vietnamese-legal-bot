"""
Shared helpers for parsing the numbered context blocks that
`AdaptiveRAGPipeline._build_context()` produces (src/rag/orchestrator.py:
"[1] ...\n\n---\n\n[2] ...") and for selecting the block(s) most relevant to
a given piece of text (a claim, or a full answer) based on the `[n]`
markers it cites.

Used by:
  - `src.rag.verifier.HallucinationVerifier` — Tier 1.5 grounding (existing)
    and Tier 2 NLI per-claim premise selection (M0.1 fix).
  - `src.rag.confidence.ConfidenceScorer` — LLM self-eval context selection
    (M0.1 fix).

Both signals used to build their context from a single blind `context[:2000]`
truncation, which silently dropped the ground-truth article whenever it fell
outside the first ~3-4 retrieved blocks (see
docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md, "2 follow-up còn lại").
Centralizing the block parser here (rather than duplicating it per caller)
keeps the two signals from drifting apart on what "the block(s) this text
cites" means.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Matches a block header line like "[1] Bộ luật Hình sự 2015 — Điều 17 (...)"
# at the start of each `_build_context()` block.
BLOCK_HEADER_PATTERN = re.compile(r'^\[(\d+)\]')
BLOCK_REF_PATTERN = re.compile(r'\[(\d+)\]')

# Word tokenizer for the lexical-overlap fallback. `\w` matches Vietnamese
# diacritics too — Python 3's `re` module treats `str` patterns as Unicode by
# default, so no extra flag is needed.
_WORD_PATTERN = re.compile(r'\w+')

# Markdown blockquote line: first non-space character is ">".
_BLOCKQUOTE_LINE = re.compile(r'^\s*>')


def parse_context_blocks(context: str) -> Dict[int, str]:
    """
    Parse `_build_context()` output — blocks numbered "[i] ..." joined by
    "\n\n---\n\n" — into an {index: block_text} dict. Derives the block set
    from what's actually parsed, not from an assumed format.

    Returns an empty dict if `context` doesn't look like numbered blocks at
    all (e.g. a plain string passed by an older caller/test) — callers use
    an empty dict as the signal to fall back to blind truncation.
    """
    blocks: Dict[int, str] = {}
    if not context:
        return blocks

    for raw_block in context.split('\n\n---\n\n'):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        match = BLOCK_HEADER_PATTERN.match(raw_block)
        if not match:
            continue
        idx = int(match.group(1))
        blocks[idx] = raw_block

    return blocks


def strip_blockquotes(text: str) -> str:
    """
    Remove Markdown blockquote lines (first non-space char is `>`) from
    `text`, returning the remaining lines joined by `\\n`.

    M3.6: the verbatim-span blockquotes hold verbatim SOURCE text. This helper
    keeps them out of every answer-derived signal that feeds the calibrated
    confidence score (context_overlap, llm_self_eval, NLI claim split,
    grounding), so the calibrated REFUSAL_THRESHOLD is not perturbed by the
    span-carrying answer format. Only the verbatim-span verifier (Tier 1.6)
    sees the raw answer. Idempotent.
    """
    if not text:
        return ""
    return "\n".join(
        line for line in text.split("\n")
        if not _BLOCKQUOTE_LINE.match(line)
    )


def extract_block_refs(text: str) -> List[int]:
    """All `[n]` references in `text`, in first-appearance order, deduplicated."""
    if not text:
        return []
    seen = set()
    refs: List[int] = []
    for raw in BLOCK_REF_PATTERN.findall(text):
        n = int(raw)
        if n not in seen:
            seen.add(n)
            refs.append(n)
    return refs


def _best_overlap_block(text: str, blocks: Dict[int, str]) -> str:
    """
    Fallback for text that cites no (valid) block: pick the block with the
    highest lexical word-overlap with `text`. Simple word-set intersection —
    no embeddings — good enough for a fallback path, per M0.1 design.
    Deterministic tie-break: lowest block index.
    """
    text_words = {w.lower() for w in _WORD_PATTERN.findall(text)}
    best_idx: Optional[int] = None
    best_overlap = -1
    for idx in sorted(blocks):
        block_words = {w.lower() for w in _WORD_PATTERN.findall(blocks[idx])}
        overlap = len(text_words & block_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return blocks[best_idx] if best_idx is not None else ""


def select_premise(
    text: str,
    context: str,
    blocks: Optional[Dict[int, str]] = None,
    cap: int = 2000,
) -> str:
    """
    Select the context substring most relevant to `text` (a single claim, or
    a full answer), for use as an NLI premise or an LLM self-eval context.

    Selection order:
      1. If `text` cites one or more valid `[n]` blocks, use the
         concatenation of those blocks (in citation order).
      2. Else, if `context` DID parse into numbered blocks, fall back to the
         single block with the highest lexical word-overlap with `text`.
      3. Else (context isn't in numbered-block format at all), fall back to
         blind `context[:cap]` — matches the pre-M0.1 behavior exactly, so
         callers passing unnumbered context (old tests, ad-hoc callers) don't
         break.

    `blocks` may be passed in (already parsed once per `context`) to avoid
    re-parsing per claim when called in a loop.
    """
    if blocks is None:
        blocks = parse_context_blocks(context)

    if not blocks:
        return (context or "")[:cap]

    refs = extract_block_refs(text)
    valid_refs = [n for n in refs if n in blocks]

    if valid_refs:
        premise = "\n\n---\n\n".join(blocks[n] for n in valid_refs)
    else:
        premise = _best_overlap_block(text, blocks)

    return premise[:cap]

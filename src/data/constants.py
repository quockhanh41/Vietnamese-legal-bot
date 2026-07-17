from __future__ import annotations

"""
Shared constants for the Vietnamese Legal Bot data pipeline.

Single source of truth — import from here instead of duplicating across files.
"""

# --- Document Type Filter ---
# Currently focused on "Luật" (Laws passed by the National Assembly).
# Note: In this dataset, "Bộ luật" (Codes like Bộ luật Dân sự, Bộ luật Hình sự)
# are classified under "Luật" and can be distinguished by title pattern "Bộ luật".
#
# Original full set (for future expansion):
# {"Luật", "Bộ luật", "Nghị định", "Thông tư", "Nghị quyết", "Quyết định"}
LEGAL_TYPES: set[str] = {"Luật"}

# All legal types available when/if scope expands (for UI dropdowns etc.)
ALL_LEGAL_TYPES: list[str] = [
    "Luật",
    "Bộ luật",
    "Nghị định",
    "Thông tư",
    "Nghị quyết",
    "Quyết định",
]

# --- Content Filters ---
MIN_CONTENT_LENGTH: int = 500  # characters — skip empty/short docs

# --- Collection Defaults ---
# "luat_law_ctx_v4_aiteamvn" (M3 Phase 0, 2026-07-16; default since 2026-07-16)
# = the SAME 600-doc corpus as v4, re-embedded with the AITeamVN mixed stack
# (dense AITeamVN/Vietnamese_Embedding + sparse BGE-M3), served with reranker
# AITeamVN/Vietnamese_Reranker. 60,958 points, is_in_effect backfilled.
# Adopted after the AITeamVN GO verdict (docs/notes/aiteamvn-adoption-go-no-go.md):
# retrieval + citation beat bge-m3/v4 (criminal routed recall@5 0.8824 vs 0.8382;
# sectors citation +2.6pt). REFUSAL_THRESHOLD was RE-CONFIRMED at 0.700 for this
# stack in M3 Phase 0 (full-pool recalibration 20260716T174730Z: at 0.700 this
# stack beats v4 — refusal_recall 0.7619 same, false_refusal 0.0972 vs 0.136) —
# no threshold change, adoption is a strict win at the current threshold.
#
# "luat_law_ctx_v4" (M2, 2026-07-14; default 2026-07-15 → 2026-07-16) = the 600-doc
# multi-sector corpus (6 sectors, curated build, Kaggle GPU ingest), a superset
# of the v3 100-doc spine. 60,958 points, capacity 9.5% RAM / 6.0% disk of the
# free tier. Tier-1 on v4 (routed recall@5, v1-subset vs v3): every sector
# within the ≤5-point M2 gate — hngd 0.85→0.80 after the VBHN-supersession fix
# (was 0.60 before; see version_resolver consolidation guard), doanh_nghiep
# watch-item held. REFUSAL_THRESHOLD 0.700 is recalibrated against THIS
# collection's score distribution (M2 Phase E, calibration 20260714T182658Z) —
# switching the default and the threshold together keeps gate behavior
# consistent.
#
# "luat_law_ctx_v3" (M1, 2026-07-10; default 2026-07-12 → 2026-07-15) = the
# 100-doc multi-sector corpus (6 sectors, `sector` keyword in payload,
# Kaggle GPU ingest) on the v2 foundations: contextual retrieval, chunker
# misalignment fixes, deterministic point IDs, int8 scalar quantization
# with fp32 on-disk. 23,727 points. Tier-1 on v3: recall@5 routed 0.8971
# criminal / 0.8922 sectors (within the ≤5-point M1 gate vs v2's 0.9216);
# kept as the M1 baseline collection for the 6→100→600 scale journey.
#
# "luat_law_ctx_v2" (M0.5, 2026-07-10) = 6 criminal docs, 2,242 points
# (recall@5 routed 0.9216, lenient==strict; chunker diagnosis in
# docs/notes/chunker-misalignment-diagnosis-2026-07-10.md). Kept in
# Qdrant, with older "luat_law_ctx" (1,670) and "luat_law", for
# before/after comparison.
DEFAULT_COLLECTION_NAME: str = "luat_law_ctx_v4_aiteamvn"

# --- Date Filter ---
# No date restriction — we keep all years for temporal versioning and
# cross-referencing with older codes that are still relevant.
DATE_FILTER_MIN_YEAR: int | None = None  # e.g., 2015 to only keep recent laws
DATE_FILTER_MAX_YEAR: int | None = None  # e.g., 2026

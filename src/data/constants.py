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
# "luat_law_ctx_v4" (M2, 2026-07-14; default since 2026-07-15) = the 600-doc
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
DEFAULT_COLLECTION_NAME: str = "luat_law_ctx_v4"

# --- Date Filter ---
# No date restriction — we keep all years for temporal versioning and
# cross-referencing with older codes that are still relevant.
DATE_FILTER_MIN_YEAR: int | None = None  # e.g., 2015 to only keep recent laws
DATE_FILTER_MAX_YEAR: int | None = None  # e.g., 2026

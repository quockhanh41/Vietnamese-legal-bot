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
# "luat_law_ctx_v2" (M0.5, 2026-07-10) = contextual retrieval + chunker
# misalignment fixes (WRONG_LABEL 126→0, 178 Điều khôi phục — see
# docs/notes/chunker-misalignment-diagnosis-2026-07-10.md) + deterministic
# point IDs + int8 scalar quantization with fp32 on-disk. 2,242 points.
# Tier-1 gate vs old collection: recall@5 routed 0.9216 (baseline 0.922),
# per-type recall unchanged, lenient==strict (labels now align with text).
# Older collections "luat_law_ctx" (1,670) and "luat_law" are kept in
# Qdrant for before/after comparison.
DEFAULT_COLLECTION_NAME: str = "luat_law_ctx_v2"

# --- Date Filter ---
# No date restriction — we keep all years for temporal versioning and
# cross-referencing with older codes that are still relevant.
DATE_FILTER_MIN_YEAR: int | None = None  # e.g., 2015 to only keep recent laws
DATE_FILTER_MAX_YEAR: int | None = None  # e.g., 2026

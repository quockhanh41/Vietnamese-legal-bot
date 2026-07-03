"""
Temporal Versioning — Time-aware legal document analysis.

Module 2 of Phase 2: Extract effective/expiry dates from Vietnamese laws,
resolve which version is valid at any point in time, and enable
time-filtered queries.

Usage:
    from src.data.timeline import DateExtractor, VersionResolver

    extractor = DateExtractor()
    timeline = extractor.extract(doc_id, content, issuance_date, law_name)

    resolver = VersionResolver(all_timelines)
    current = resolver.at_time()          # Laws currently in effect
    past = resolver.at_time(date(2018, 1, 1))  # Laws in 2018
    chain = resolver.version_chain("Luật Doanh nghiệp")
"""

from src.data.timeline.date_extractor import DateExtractor, LawTimeline
from src.data.timeline.version_resolver import VersionResolver, VersionResult

__all__ = [
    "DateExtractor",
    "LawTimeline",
    "VersionResolver",
    "VersionResult",
]

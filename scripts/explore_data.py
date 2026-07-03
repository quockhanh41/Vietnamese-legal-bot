"""
Safe exploration script — samples the parquet file without loading all 10GB into RAM.

Usage:
    python scripts/explore_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyarrow.parquet as pq
import pandas as pd
from collections import Counter

from src.data.constants import LEGAL_TYPES


DATA_PATH = "data/raw/legal_docs_merged.parquet"


def explore_safe():
    print("=" * 60)
    print("SAFE DATA EXPLORATION (sampling only)")

    pf = pq.ParquetFile(DATA_PATH)
    schema = pf.schema_arrow
    print(f"\nRows: {pf.metadata.num_rows:,}")
    print(f"Row groups: {pf.metadata.num_row_groups}")
    print(f"Columns: {[f.name for f in schema]}")

    # Sample: read only first row group (~1000-5000 rows)
    first_group = pf.read_row_group(0).to_pandas()
    print(f"\nFirst row group: {len(first_group):,} rows ({first_group.memory_usage(deep=True).sum()/1024/1024:.0f} MB)")

    # Sample for content length stats
    print("\n--- Content Length (from first 2000 rows) ---")
    sample = first_group.head(2000)
    lengths = sample["content"].dropna().str.len()
    if len(lengths) > 0:
        print(f"  Min: {lengths.min():,}")
        print(f"  Median: {lengths.median():,.0f}")
        print(f"  Max: {lengths.max():,}")

    # Legal types from first group
    print("\n--- Legal Types (first row group) ---")
    for t, c in Counter(first_group["legal_type"].dropna()).most_common(10):
        print(f"  {t:<30s} {c:>6,}")

    # Scan ALL row groups for legal type counts (metadata-only, fast)
    # We read just the 'legal_type' column from each group
    print("\n--- Legal Types (ALL rows, column scan) ---")
    type_counter = Counter()
    date_years = Counter()
    total_scanned = 0

    for i in range(pf.metadata.num_row_groups):
        # Read ONLY the 'legal_type' column → very low memory
        col_table = pf.read_row_group(i, columns=["legal_type", "issuance_date"])
        col_df = col_table.to_pandas()
        type_counter.update(col_df["legal_type"].dropna())
        total_scanned += len(col_df)

        # Count years
        for d in col_df["issuance_date"].dropna():
            if isinstance(d, str) and len(d) >= 4:
                year = d.split("/")[-1]
                if year.isdigit():
                    date_years[int(year)] += 1

    print(f"  Scanned: {total_scanned:,} rows")
    print(f"  Unique types: {len(type_counter)}")
    for t, c in type_counter.most_common(15):
        print(f"  {t:<30s} {c:>8,}")

    # Year distribution
    print("\n--- Year Distribution (2015-2026) ---")
    for y in range(2015, 2027):
        count = date_years.get(y, 0)
        bar = "█" * int(40 * count / max(date_years.values(), default=1))
        print(f"  {y}: {count:>8,}  {bar}")

    # Estimate after filter
    filtered_count = sum(c for t, c in type_counter.items() if t in LEGAL_TYPES)
    recent_count = sum(date_years.get(y, 0) for y in range(2015, 2027))
    print(f"\n--- Filter Estimate ---")
    print(f"  After type filter: ~{filtered_count:,} docs")
    print(f"  After date filter (2015-2026): ~{recent_count:,} docs")
    print(f"  After type + date + length: ~{min(filtered_count, recent_count):,} docs (estimated)")

    print("\n✅ Exploration complete (low memory)")


if __name__ == "__main__":
    explore_safe()

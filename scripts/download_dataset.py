"""
Download & explore the vietnamese-legal-documents dataset from HuggingFace.

Dataset: vohuutridung/vietnamese-legal-documents
- 518,255 documents, ~3.6 GB
- Two configs: 'metadata' (9 columns) and 'content' (2 columns)
- Joinable on 'id'
- Content is in Markdown format (converted from HTML)
"""

import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset

from src.data.constants import LEGAL_TYPES, ALL_LEGAL_TYPES, MIN_CONTENT_LENGTH


def download_dataset(output_dir: str = "data/raw"):
    """Download both configs and merge into a single parquet file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading metadata config...")
    meta_ds = load_dataset(
        "vohuutridung/vietnamese-legal-documents",
        "metadata",
        split="data",
    )
    meta_df = meta_ds.to_pandas()
    print(f"  Metadata: {len(meta_df)} rows, {meta_df.memory_usage(deep=True).sum() / 1024 / 1024:.0f} MB")

    print("Downloading content config...")
    content_ds = load_dataset(
        "vohuutridung/vietnamese-legal-documents",
        "content",
        split="data",
    )
    content_df = content_ds.to_pandas()
    print(f"  Content: {len(content_df)} rows, {content_df.memory_usage(deep=True).sum() / 1024 / 1024:.0f} MB")

    print("Merging metadata + content on 'id'...")
    df = meta_df.merge(content_df, on="id", how="inner")
    print(f"  Merged: {len(df)} rows")

    # Save
    output_path = output_dir / "legal_docs_merged.parquet"
    df.to_parquet(output_path, compression="snappy")
    print(f"Saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.0f} MB")

    return df


def explore_dataset(df: pd.DataFrame):
    """Print exploratory stats about the dataset."""
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    print(f"\nTotal documents: {len(df):,}")

    # Legal types distribution
    print("\n--- Legal Types (top 20) ---")
    type_counts = df["legal_type"].value_counts()
    for t, c in type_counts.head(20).items():
        bar = "█" * int(40 * c / type_counts.max())
        print(f"  {t:<30s} {c:>8,}  {bar}")

    # Date distribution
    print("\n--- Date Range ---")
    df["issuance_date"] = pd.to_datetime(df["issuance_date"], format="%d/%m/%Y", errors="coerce")
    print(f"  Earliest: {df['issuance_date'].min().date()}")
    print(f"  Latest:   {df['issuance_date'].max().date()}")
    print(f"  Null dates: {df['issuance_date'].isna().sum():,}")

    # Content length distribution
    content_lengths = df["content"].str.len()
    print("\n--- Content Length (characters) ---")
    print(f"  Min:    {content_lengths.min():,}")
    print(f"  P25:    {content_lengths.quantile(0.25):,.0f}")
    print(f"  Median: {content_lengths.median():,.0f}")
    print(f"  P75:    {content_lengths.quantile(0.75):,.0f}")
    print(f"  P95:    {content_lengths.quantile(0.95):,.0f}")
    print(f"  Max:    {content_lengths.max():,}")

    # Issuing authorities (top)
    print("\n--- Issuing Authorities (top 15) ---")
    auth_counts = df["issuing_authority"].value_counts()
    for a, c in auth_counts.head(15).items():
        print(f"  {a:<50s} {c:>8,}")

    # Legal sectors distribution (parsed from pipe-separated)
    print("\n--- Legal Sectors (top 15) ---")
    all_sectors = df["legal_sectors"].dropna().str.split("|").explode().str.strip()
    sector_counts = all_sectors.value_counts()
    for s, c in sector_counts.head(15).items():
        print(f"  {s:<40s} {c:>8,}")

    # Sample document
    print("\n--- Sample Document ---")
    sample = df[df["content"].str.len() > 1000].iloc[0]
    print(f"  Title: {sample['title']}")
    print(f"  Type: {sample['legal_type']}")
    print(f"  Date: {sample['issuance_date']}")
    print(f"  Authority: {sample['issuing_authority']}")
    print(f"  Content preview (first 500 chars):")
    print(f"  {sample['content'][:500]}...")

    # After-filter estimate
    print("\n--- Post-Filter Estimate ---")
    filtered = df[
        df["legal_type"].isin(LEGAL_TYPES)
        & (df["content"].str.len() >= MIN_CONTENT_LENGTH)
    ]
    print(f"  Filter (type + length): {len(filtered):,} documents")
    print(f"  Legal type distribution after filter:")
    for t, c in filtered["legal_type"].value_counts().items():
        print(f"    {t:<20s} {c:>8,}")


if __name__ == "__main__":
    df = download_dataset()
    explore_dataset(df)

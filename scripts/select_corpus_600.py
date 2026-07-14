"""M2.2 prep — build the ~600-doc corpus by deepening the same 6 sectors.

Spine (100 hand-curated ids in scripts/select_corpus.py) is locked in.
~500 candidates are auto-generated (relevant legal_type + sector match +
content guard), budgeted by estimated chunk count with max/min <= ~3x
across sectors (frequency-bias control, spec B.5), exported to a review
table, then (after user prune) frozen into corpus_600docs.parquet.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.constants import MIN_CONTENT_LENGTH
from scripts.select_corpus import CORPUS  # spine

SPINE_SECTOR = {doc_id: sector for doc_id, sector in CORPUS}
SPINE_IDS = set(SPINE_SECTOR)

RELEVANT_TYPES = {"Luật", "Bộ luật", "Nghị định", "Thông tư",
                  "Thông tư liên tịch", "Văn bản hợp nhất"}

# legal_sectors tokens (comma-separated, noisy) → our 6 curated sectors.
SECTOR_KEYWORDS = {
    "hinh_su":      {"Trách nhiệm hình sự", "Thủ tục Tố tụng", "Vi phạm hành chính"},
    "dan_su":       {"Quyền dân sự", "Sở hữu trí tuệ", "Dịch vụ pháp lý"},
    "lao_dong":     {"Lao động - Tiền lương", "Bảo hiểm"},
    "doanh_nghiep": {"Doanh nghiệp", "Đầu tư", "Thương mại", "Chứng khoán",
                     "Kế toán - Kiểm toán"},
    "dat_dai":      {"Bất động sản", "Tài nguyên - Môi trường"},
    "hngd":         set(),   # no dedicated token → title hints only
}
# Priority order when a doc matches several sectors' tokens (most specific first).
# hngd is omitted: it has no dedicated token (fallback via title hints only).
_SECTOR_ORDER = ["hinh_su", "dat_dai", "lao_dong", "doanh_nghiep", "dan_su"]
HNGD_TITLE_HINTS = {"hôn nhân", "gia đình", "trẻ em", "nuôi con nuôi", "bình đẳng giới"}


def _tokens(legal_sectors):
    return {t.strip() for t in (legal_sectors or "").split(",") if t.strip()}


def sector_of(legal_sectors, title):
    """Return one of the 6 sectors, or None if out of scope.
    Token match wins; HNGĐ title hints are a fallback only when no
    legal_sectors token maps to a sector (hngd has no dedicated token)."""
    toks = _tokens(legal_sectors)
    for sector in _SECTOR_ORDER:
        if toks & SECTOR_KEYWORDS[sector]:
            return sector
    if any(h in (title or "").lower() for h in HNGD_TITLE_HINTS):
        return "hngd"
    return None


def estimate_chunks(content):
    """Cheap chunk estimate (v3: ~700 chars/chunk). Floor of 1."""
    return max(1, len(content or "") // 700)


def is_candidate(row):
    if row.get("id") in SPINE_IDS:
        return False
    if row.get("legal_type") not in RELEVANT_TYPES:
        return False
    content = row.get("content")
    if not isinstance(content, str) or len(content) < MIN_CONTENT_LENGTH:
        return False
    return sector_of(row.get("legal_sectors"), row.get("title")) is not None


def rank_key(cand):
    """Higher sorts first (use sorted(..., key=rank_key, reverse=True)):
    spine-anchored > consolidated > newer > longer-but-not-absurd."""
    return (
        1 if cand.get("anchored") else 0,
        1 if cand.get("is_consolidated") else 0,
        cand.get("year") or 0,
        -abs(cand.get("est_chunks", 0) - 60),   # prefer ~mid-size docs
    )


def allocate_budget(ranked_by_sector, spine_chunks, target_added_chunks, ratio_cap=3.0):
    """Grow the smallest sectors to raise the chunk-count floor, converging
    toward max/min <= ratio_cap. BEST-EFFORT, not a hard guarantee: the spine
    alone may already exceed ratio_cap and cannot be shrunk, and a sector may
    run out of candidates — so the real balance gate is the actual-chunk check
    in Task 6 (C1), not this estimate-based pass. Each step picks the sector
    whose next-ranked candidate yields the smallest resulting max/min ratio,
    applied only if the result is within ratio_cap OR no worse than the current
    ratio (so a spine that starts over cap still converges downward and no
    oversized candidate is force-added to overshoot). Returns
    (picks_by_sector, totals_by_sector)."""
    if not ranked_by_sector:
        return {}, {}
    totals = {s: spine_chunks.get(s, 0) for s in ranked_by_sector}
    picks = {s: [] for s in ranked_by_sector}
    idx = {s: 0 for s in ranked_by_sector}
    added = 0

    def _ratio(t):
        lo = min(t.values())
        return max(t.values()) / lo if lo > 0 else float("inf")

    while added < target_added_chunks:
        cur = _ratio(totals)
        best = None  # (resulting_ratio, sector)
        for s in ranked_by_sector:
            if idx[s] >= len(ranked_by_sector[s]):
                continue
            trial = dict(totals)
            trial[s] += ranked_by_sector[s][idx[s]]["est_chunks"]
            r = _ratio(trial)
            if best is None or r < best[0]:
                best = (r, s)
        if best is None:
            break  # every sector exhausted
        r, s = best
        if r > ratio_cap and r > cur + 1e-9:
            break  # best available move worsens balance past the cap
        nxt = ranked_by_sector[s][idx[s]]
        picks[s].append(nxt)
        totals[s] += nxt["est_chunks"]
        idx[s] += 1
        added += nxt["est_chunks"]
    return picks, totals


import csv, re
import pandas as pd
import pyarrow.parquet as pq
import click

MERGED_PATH = "data/raw/legal_docs_merged.parquet"
SPINE_PARQUET = "data/raw/corpus_100docs.parquet"
CANDIDATES_CSV = "data/raw/corpus_600_candidates.csv"
OUTPUT_PATH = "data/raw/corpus_600docs.parquet"

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _year(row):
    d = (row.get("issuance_date") or "").strip()
    m = re.search(r"/(\d{4})$", d)
    if m:
        return int(m.group(1))
    m = _YEAR_RE.search(row.get("title", "") or "")
    return int(m.group(0)) if m else 0


def _anchored(title):
    return "hướng dẫn" in (title or "").lower()


def scan_candidates(merged_path=MERGED_PATH):
    pf = pq.ParquetFile(merged_path)
    cands = []
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        for _, r in df.iterrows():
            row = r.to_dict()
            if not is_candidate(row):
                continue
            cands.append({
                "id": int(row["id"]),
                "sector": sector_of(row.get("legal_sectors"), row.get("title")),
                "legal_type": row["legal_type"],
                "title": row.get("title", ""),
                "year": _year(row),
                "est_chunks": estimate_chunks(row.get("content")),
                "is_consolidated": row["legal_type"] == "Văn bản hợp nhất",
                "anchored": _anchored(row.get("title")),
            })
    return cands


def spine_chunks_from_parquet(corpus_100_path=SPINE_PARQUET):
    df = pq.read_table(corpus_100_path, columns=["sector", "content"]).to_pandas()
    out = {}
    for _, r in df.iterrows():
        out[r["sector"]] = out.get(r["sector"], 0) + estimate_chunks(r["content"])
    return out


# _SECTOR_ORDER (Task 2) is token-match priority for sector_of() and
# deliberately omits "hngd" (no dedicated legal_sectors token — reached only
# via the title-hint fallback). For CLI output we need every sector that can
# actually receive candidates/picks, so extend it here rather than mutate the
# existing constant.
_ALL_SECTORS = _SECTOR_ORDER + ["hngd"]


def build_corpus_600(candidates_csv=CANDIDATES_CSV, merged_path=MERGED_PATH,
                     output_path=OUTPUT_PATH):
    wanted = dict(SPINE_SECTOR)   # spine locked in
    with open(candidates_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["keep"].strip() == "1":
                wanted[int(r["id"])] = r["sector"]

    pf = pq.ParquetFile(merged_path)
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        hit = t[t["id"].isin(wanted)]
        if len(hit):
            parts.append(hit)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id")

    missing = set(wanted) - set(df["id"])
    if missing:
        raise SystemExit(f"NOT FOUND {len(missing)} ids in dataset: {sorted(missing)}")
    assert SPINE_IDS <= set(df["id"]), "spine doc dropped!"

    df["sector"] = df["id"].map(wanted)
    df = df.sort_values("sector").reset_index(drop=True)
    df.to_parquet(output_path)

    print(f"Wrote {output_path}: {len(df)} docs")
    est = df.groupby("sector")["content"].apply(lambda s: sum(estimate_chunks(c) for c in s))
    for sector, g in df.groupby("sector"):
        n_tt = (g["legal_type"] == "Thông tư").sum()
        print(f"  {sector:<14}{len(g):>4} docs  {n_tt:>3} Thông tư  ~{est[sector]:>6,} chunks")
    return df


@click.command()
@click.option("--build", is_flag=True,
              help="Build corpus_600docs.parquet from the approved candidates CSV "
                   "instead of scanning for new candidates.")
@click.option("--target-added-chunks", type=int, default=36000,
              help="Approx chunks to add on top of the ~23.7k spine (tune to hit ~600 docs).")
@click.option("--ratio-cap", type=float, default=3.0)
def main(build, target_added_chunks, ratio_cap):
    if build:
        build_corpus_600()
        return
    cands = scan_candidates()
    by_sector = {s: [] for s in SECTOR_KEYWORDS}
    for c in cands:
        by_sector[c["sector"]].append(c)
    for s in by_sector:
        by_sector[s].sort(key=rank_key, reverse=True)
    spine = spine_chunks_from_parquet()
    picks, totals = allocate_budget(by_sector, spine, target_added_chunks, ratio_cap)

    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["keep", "id", "sector", "legal_type", "year", "est_chunks",
                    "anchored", "is_consolidated", "title"])
        for s in _ALL_SECTORS:
            for c in picks[s]:
                w.writerow(["1", c["id"], s, c["legal_type"], c["year"], c["est_chunks"],
                            int(c["anchored"]), int(c["is_consolidated"]), c["title"]])

    print(f"{'sector':<14}{'spine ch':>10}{'+added ch':>10}{'total ch':>10}{'+docs':>7}")
    for s in _ALL_SECTORS:
        print(f"{s:<14}{spine.get(s,0):>10,}{totals[s]-spine.get(s,0):>10,}"
              f"{totals[s]:>10,}{len(picks[s]):>7}")
    ratio = max(totals.values()) / min(totals.values())
    print(f"\nmax/min chunk ratio = {ratio:.2f} (cap {ratio_cap})  "
          f"{'OK' if ratio <= ratio_cap + 1e-9 else 'OVER'}")
    print(f"total added docs = {sum(len(picks[s]) for s in picks)}  "
          f"(+100 spine = {100 + sum(len(picks[s]) for s in picks)})")
    print(f"\nReview + prune: {CANDIDATES_CSV}  (set keep=0 to drop a row)")


if __name__ == "__main__":
    main()

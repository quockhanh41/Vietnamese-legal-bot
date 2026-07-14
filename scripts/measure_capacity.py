"""M2.1 — Capacity gate from Qdrant telemetry (read-only).

Reads GET {QDRANT_URL}/telemetry?details_level=10 for real per-collection
byte sizes and projects a target point count against the free-tier limits
(1GB RAM / 4GB disk). Gate: projected usage < 70%.

Usage:
    .venv/bin/python scripts/measure_capacity.py                 # table + project 142k
    .venv/bin/python scripts/measure_capacity.py --target-points 150000
"""
import json, os, sys, time
import click, requests
from dotenv import load_dotenv

load_dotenv()

FREE_TIER_RAM = 1024 ** 3          # 1 GB
FREE_TIER_DISK = 4 * 1024 ** 3     # 4 GB
GATE_PCT = 70.0


def parse_collection_sizes(telemetry: dict) -> dict:
    """Sum vectors_size_bytes + payloads_size_bytes across every shard of
    every collection in a /telemetry?details_level=10 response."""
    out = {}
    cols = telemetry.get("result", {}).get("collections", {}).get("collections", [])
    for col in cols:
        name = col.get("id")
        v = p = n = 0
        for shard in col.get("shards", []):
            local = (shard or {}).get("local") or {}
            v += local.get("vectors_size_bytes", 0) or 0
            p += local.get("payloads_size_bytes", 0) or 0
            n += local.get("num_points", 0) or 0
        out[name] = {"vectors_bytes": v, "payloads_bytes": p, "points": n}
    return out


def project_capacity(points: int, dim: int = 1024,
                     sparse_bytes_per_point: int = 400,
                     hnsw_bytes_per_point: int = 256) -> dict:
    """Estimate RAM/disk for `points` chunks under the v3 config:
    int8-quantized dense always-RAM (dim bytes/point) + sparse + HNSW graph;
    fp32 original on-disk (dim*4 bytes/point) + payload (~130 B/point observed)."""
    ram = points * dim + points * sparse_bytes_per_point + points * hnsw_bytes_per_point
    disk = points * dim * 4 + points * 130
    return {
        "ram_bytes": ram,
        "disk_bytes": disk,
        "ram_pct": 100.0 * ram / FREE_TIER_RAM,
        "disk_pct": 100.0 * disk / FREE_TIER_DISK,
    }


def _fetch(url: str, key: str, path: str) -> dict:
    r = requests.get(f"{url.rstrip('/')}{path}", headers={"api-key": key}, timeout=30)
    r.raise_for_status()
    return r.json()


@click.command()
@click.option("--target-points", type=int, default=142_000,
              help="Projected chunk count for the 600-doc corpus (same-mix extrapolation).")
def main(target_points):
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url:
        sys.exit("QDRANT_URL not set")
    tel = _fetch(url, key, "/telemetry?details_level=10")
    sizes = parse_collection_sizes(tel)

    print(f"{'collection':<22}{'points':>10}{'vectors MB':>12}{'payload MB':>12}{'B/point':>10}")
    stale = []
    for name, s in sorted(sizes.items()):
        n = s["points"]
        vb, pb = s["vectors_bytes"], s["payloads_bytes"]
        bpp = (vb + pb) / n if n else 0
        print(f"{name:<22}{n:>10,}{vb/1e6:>12.1f}{pb/1e6:>12.1f}{bpp:>10.0f}")
        if name in {"luat_law", "luat_law_ctx", "vietnamese_laws"}:
            stale.append(name)

    proj = project_capacity(target_points)
    print(f"\nProjection @ {target_points:,} points (600-doc corpus):")
    print(f"  RAM  ~{proj['ram_bytes']/1e6:8.0f} MB  ({proj['ram_pct']:5.1f}% of 1GB)  "
          f"{'PASS' if proj['ram_pct'] < GATE_PCT else 'FAIL'} (<{GATE_PCT}%)")
    print(f"  Disk ~{proj['disk_bytes']/1e6:8.0f} MB  ({proj['disk_pct']:5.1f}% of 4GB)  "
          f"{'PASS' if proj['disk_pct'] < GATE_PCT else 'FAIL'} (<{GATE_PCT}%)")
    if stale:
        print(f"\n⚠️  Drop before ingest to reclaim space: {', '.join(stale)}")

    os.makedirs("data/eval_reports", exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(f"data/eval_reports/capacity_{ts}.json", "w") as f:
        json.dump({"sizes": sizes, "projection": proj, "target_points": target_points}, f, indent=2)
        f.write("\n")
    print(f"\nWrote data/eval_reports/capacity_{ts}.json")


if __name__ == "__main__":
    main()

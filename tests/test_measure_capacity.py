from scripts.measure_capacity import parse_collection_sizes, project_capacity, GATE_PCT

_TELEMETRY = {
    "result": {"collections": {"collections": [
        {"id": "luat_law_ctx_v3",
         "shards": [{"local": {"vectors_size_bytes": 8559563, "payloads_size_bytes": 3034112, "num_points": 23727}}]},
        {"id": "vietnamese_laws",
         "shards": [{"local": {"vectors_size_bytes": 8722368, "payloads_size_bytes": 2456320, "num_points": 3441}}]},
    ]}}
}

def test_parse_collection_sizes_sums_shards():
    sizes = parse_collection_sizes(_TELEMETRY)
    assert sizes["luat_law_ctx_v3"]["vectors_bytes"] == 8559563
    assert sizes["luat_law_ctx_v3"]["payloads_bytes"] == 3034112
    assert sizes["luat_law_ctx_v3"]["points"] == 23727
    assert set(sizes) == {"luat_law_ctx_v3", "vietnamese_laws"}

def test_project_capacity_quantized_ram_and_pct():
    p = project_capacity(points=142_000)
    # int8 quantized dense = points * dim bytes = 142000 * 1024
    assert p["ram_bytes"] >= 142_000 * 1024
    # 600-doc worst case must stay under the 70% gate on the 1GB tier
    assert p["ram_pct"] < GATE_PCT
    assert 0 < p["disk_pct"] < GATE_PCT

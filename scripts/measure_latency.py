"""M5: measure warm end-to-end latency of /api/chat against a running backend.
Prints P50/P95/max per route. NOT a pass/fail gate — numbers go into the README
honestly (HF CPU is slow). Usage: python scripts/measure_latency.py [BASE_URL]"""
import sys
import time

import httpx

MEDIUM_QS = ["Tội trộm cắp tài sản bị xử phạt thế nào?",
             "Điều kiện kết hôn theo luật hôn nhân gia đình là gì?"]
COMPLEX_QS = ["So sánh mức phạt tội cố ý gây thương tích và tội giết người",
              "Quyền và nghĩa vụ của người lao động khi chấm dứt hợp đồng?"]
N = 20


def percentiles(latencies_ms):
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    s = sorted(latencies_ms)

    def pct(p):
        idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
        return s[idx]
    return {"p50": pct(50), "p95": pct(95), "max": s[-1], "n": len(s)}


def _wait_ready(base_url, timeout=180, interval=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/ready", timeout=5).json().get("ready"):
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _probe(base_url, questions, use_adaptive):
    lats = []
    for i in range(N):
        q = questions[i % len(questions)]
        t0 = time.time()
        r = httpx.post(f"{base_url}/api/chat",
                       json={"question": q, "session_id": f"probe-{i}",
                             "use_adaptive": use_adaptive}, timeout=180)
        lats.append((time.time() - t0) * 1000 if r.status_code == 200 else float("nan"))
    return percentiles([x for x in lats if x == x])  # drop NaN


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    if not _wait_ready(base_url):
        print("Backend not ready within timeout"); sys.exit(1)
    for name, qs, adaptive in [("MEDIUM", MEDIUM_QS, False), ("COMPLEX", COMPLEX_QS, True)]:
        print(f"{name}: {_probe(base_url, qs, adaptive)}")


if __name__ == "__main__":
    main()

"""M5: functional smoke test against a running backend (/api/chat). Waits on
/ready first (avoids cold-start flakiness), then exercises 4 canonical paths.
Exit 0 = all green. Usage: python scripts/smoke_test.py [BASE_URL]"""
import sys
import time

import httpx

TIMEOUT = 120


def wait_ready(base_url, timeout=120, interval=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/ready", timeout=5).json().get("ready"):
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def check_answerable(data):
    ok = bool(data.get("citations")) and not data.get("refused")
    return ok, f"answerable: citations={len(data.get('citations') or [])} refused={data.get('refused')}"


def check_refused(data):
    return bool(data.get("refused")) is True, f"refused={data.get('refused')}"


def check_citation_lookup(data):
    return bool(data.get("citations")), f"citation-lookup: citations={len(data.get('citations') or [])}"


def check_condensed(data, original):
    cq = data.get("condensed_question") or ""
    ok = bool(cq) and cq.strip() != original.strip()
    return ok, f"condensed='{cq}'"


def _post(base_url, question, session_id, use_adaptive=True):
    return httpx.post(f"{base_url}/api/chat",
                      json={"question": question, "session_id": session_id,
                            "use_adaptive": use_adaptive}, timeout=TIMEOUT).json()


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    if not wait_ready(base_url):
        print("FAIL: backend not ready"); sys.exit(1)

    results = []
    results.append(check_answerable(_post(base_url, "Tội trộm cắp tài sản bị xử phạt thế nào?", "smoke-1")))
    results.append(check_refused(_post(base_url, "Hôm nay Hà Nội thời tiết thế nào?", "smoke-2")))
    results.append(check_citation_lookup(_post(base_url, "Điều 173 Bộ luật Hình sự quy định gì?", "smoke-3")))
    followup_orig = "còn tội cướp giật thì sao?"
    _post(base_url, "Tội trộm cắp tài sản bị xử phạt thế nào?", "smoke-4")
    results.append(check_condensed(_post(base_url, followup_orig, "smoke-4"), original=followup_orig))

    ok = all(r[0] for r in results)
    for passed, msg in results:
        print(("PASS " if passed else "FAIL ") + msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

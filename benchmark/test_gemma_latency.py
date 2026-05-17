"""Quick latency test for gemma-doc-label API."""
import json
import time
import urllib.request

API = "http://127.0.0.1:8003/classify_text"
TEXTS = [
    "Employee W-2 Wage and Tax Statement for 2024. Federal tax withheld: $12,500.",
    "This Non-Disclosure Agreement is between Company A and Company B regarding proprietary technology.",
    "Quarterly Financial Report: Revenue $2.5M, Net Income $450K for Q3 2024.",
]

print("gemma-doc-label latency test")
print("-" * 50)

total = 0.0
for i, text in enumerate(TEXTS):
    data = json.dumps({"text": text, "filename": f"test_{i}.txt"}).encode("utf-8")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            API, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - t0
        total += elapsed
        l1 = raw.get("l1", {}).get("label", "?")
        l2 = raw.get("l2", {}).get("label", "?")
        print(f"  [{i+1}] {elapsed:.1f}s  L1={l1}  L2={l2}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [{i+1}] {elapsed:.1f}s  ERROR: {e}")

avg = total / len(TEXTS) if total > 0 else 0
print("-" * 50)
print(f"Average: {avg:.1f}s per document")
print(f"Estimate for dspm27 (27 docs): ~{avg*27/60:.0f} min")
print(f"Estimate for cxh5types (258 docs): ~{avg*258/60:.0f} min")

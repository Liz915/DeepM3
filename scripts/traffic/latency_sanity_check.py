import requests
import time
import numpy as np

URL = "http://localhost:8000/recommend"

def measure(user_id, items, label):
    start = time.perf_counter()
    resp = requests.post(URL, json={"user_id": user_id, "recent_items": items, "recent_times": [0.1]*len(items)})
    lat = (time.perf_counter() - start) * 1000
    try:
        src = resp.json().get("data", {}).get("reasoning_source", "err")
    except: src = "err"
    print(f"[{label}] {lat:.2f}ms | {src}")
    return lat

print(" Starting P7 Benchmark...")

# 1.  L3 (Slow Path)  L2
#  L2 (Fast)  L3 ( Entropy )
measure("user_new_1", [10, 20], "COLD_1")

# 2.  L1 (Cache Hit)
print(" Warming up cache for Hot User...")
# 
measure("user_hot", [1, 2, 3], "WARMUP_REQ")

#  1.5  DeepSeek  Cache
print(" Waiting for background reasoning...")
time.sleep(1.5) 

print(" Testing L1 Latency...")
hits = []
for i in range(10):

    hits.append(measure("user_hot", [1, 2, 3], "L1_HIT"))

print(f"\n Average L1 Latency: {np.mean(hits):.2f} ms")
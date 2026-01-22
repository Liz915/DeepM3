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

print("🚀 Starting P7 Benchmark...")

# 1. 制造 L3 (Slow Path) 或 L2
# 第一次访问，大概率是 L2 (Fast) 或者 L3 (如果 Entropy 高)
measure("user_new_1", [10, 20], "COLD_1")

# 2. 制造 L1 (Cache Hit)
print("🔥 Warming up cache for Hot User...")
# 发送第一次请求
measure("user_hot", [1, 2, 3], "WARMUP_REQ")

# 睡 1.5 秒，等待后台 DeepSeek 写入 Cache
print("⏳ Waiting for background reasoning...")
time.sleep(1.5) 

print("🚀 Testing L1 Latency...")
hits = []
for i in range(10):
    # 现在应该全是 Hit 了
    hits.append(measure("user_hot", [1, 2, 3], "L1_HIT"))

print(f"\n🏆 Average L1 Latency: {np.mean(hits):.2f} ms")
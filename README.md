# DeepM3: Dynamic System-2 Scaling for Recommender Systems

![cover](assets/cover.png)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

> A hybrid recommender system that combines continuous-time user dynamics (System 1)
> with selective LLM reasoning (System 2), designed for low latency, low cost,
> and stable behavior under real-world traffic.
This repository provides a **fully containerized, reproducible implementation** with built-in monitoring and experiment scripts.

## 📊 Live System Monitoring (Built-in)

DeepM3 ships with a pre-configured Prometheus + Grafana stack.

<div align="center">
  <img src="assets/monitor_latency_p99.png" width="30%" />
  <img src="assets/monitor_qps_throughput.png" width="30%" />
  <img src="assets/monitor_cache_efficiency.png" width="30%" />
</div>

## 🏗 Architecture Overview

DeepM3 follows a **System 1 / System 2** cognitive architecture:

1.  **System 1 (Fast Path)**: 
    A Neural ODE model captures continuous-time user trajectories, serving 80%+ of requests with millisecond-level latency.
2.  **System 2 (Slow Path)**: 
    An agent-based reasoning module (DeepSeek-V3) is activated only when uncertainty is detected (e.g., high entropy, multimodal conflicts).
3.  **Adaptive Router**: 
    Dynamically selects between paths to balance accuracy, cost, and system stability.
4.  **Observability**: 
    Native Prometheus & Grafana integration tracks QPS, Tail Latency (P99), and Cache Hit Rates in real-time.

---

## 🚀 Quick Start (Reproducible)

The entire system (API + Prometheus + Grafana) can be deployed with **one command**.  
**No external API key is required** (Runs in Mock Mode by default for zero-cost reproduction).

### 1. Start Services

```bash
git clone [https://github.com/Liz915/DeepM3.git](https://github.com/Liz915/DeepM3.git)
cd DeepM3

# Build and start all containers
docker-compose up -d --build
```

### 2. Test the API (Usage)
Once started, the API listens at http://localhost:8000.

**Option A : Simple Request (Fast Path)**
```bash
curl -X POST http://localhost:8000/recommend \
-H "Content-Type: application/json" \
-d '{"user_id":"demo_user", "recent_items":[1,2,3], "recent_times":[0.1,0.2,0.5]}'
```
**Option B : Complex Request (Triggers System 2 Agent)**
```bash
curl -X POST http://localhost:8000/recommend \
-H "Content-Type: application/json" \
-d '{
    "user_id": "vip_user_chaos",
    "recent_items": [10, 500, 5], 
    "recent_times": [0.1, 8.0, 15.2],
    "image_input": "error_stack_trace.png"
}'
```
**Expected Response (Example):**
```json 
{
  "status": "success",
  "data": {
    "user_id": "vip_user_chaos",
    "meta": { 
      "routing_decision": "slow_path", 
      "entropy": 10.0 
    },
    "reasoning_source": "slow_path (System 2)",
    "trace": ["perception", "fusion", "recall", "decision"],
    "recommendations": [
       { "item_id": 1042, "score": 0.98 },
       { "item_id": 503,  "score": 0.95 }
    ]
  },
  "strategy": "Adaptive_ODE_Agent",
  "latency": "0.29ms"
}
```

**Note:** In the default Mock Mode, the system returns structured responses instantly (<1ms) to verify high-throughput routing logic. Real-world latency (~1s) applies when a valid API Key is provided.

### 3. Traffic Simulation & Monitoring
To visualize routing behavior and cache dynamics in Grafana, run the traffic generator:
```bash sh nano_traffic_test.sh ```

**Access Dashboard:**
- URL: `http://localhost:3000`
- Login: `admin` / `admin`
- Dashboard: Click "Deep-M3 System Monitor" (Pre-loaded)
## 🧪 Reproducibility & Mock Mode
To ensure full reproducibility and avoid external dependencies, DeepM3 runs in Mock Mode by default.

| Mode | Trigger Condition | Description |
|------|------------------|-------------|
| **Mock Mode** | `DEEPSEEK_API_KEY` is empty | Returns deterministic, structured synthetic responses instantly. Ideal for logic verification and CI/CD. |
| **Real Mode** | `DEEPSEEK_API_KEY` is set | Enables live DeepSeek-V3 reasoning. Latency will reflect real-world LLM inference times (~1s). |
## 🔬 Experiments & Benchmarks
All results are reproducible using the scripts in `scripts/experiments/`.

**Key Results**
| Metric | Baseline (All-LLM) | DeepM3 (Ours) | Improvement |
|--------|-------------------|---------------|-------------|
| Avg Latency | ~2000 ms | ~408 ms | ⚡ **4.9x Faster** |
| Cost / 1k Req | $5.00 | **$1.80** | 💰 **64% Savings** |
| Routing Acc | N/A | 86.5% | 🎯 **High Precision** |
| JSON Errors | 16.0% | 4.0% | ✅ **DPO Aligned** |

**Running Experiments**
```bash
# 1. Routing Accuracy Ablation
python scripts/experiments/exp_routing.py

# 2. System Efficiency (Latency/Cost)
python scripts/experiments/exp_efficiency.py

# 3. Alignment Quality (JSON Structure)
python scripts/experiments/exp_alignment.py 
```
## 📂 Project Structure

```
DeepM3/
├── src/
│ ├── api.py # Unified FastAPI entrypoint
│ ├── agent/
│ │ ├── router.py # Entropy-based adaptive router
│ │ └── tools_deepseek.py # LLM Interface (Mock/Real)
│ ├── dynamics/ # Neural ODE models (torchdiffeq)
│ └── data/ # Dataset loaders
├── scripts/
│ ├── experiments/ # Reproducibility benchmark scripts
│ ├── train/ # Training pipelines
│ └── traffic/ # Traffic simulation
├── configs/ # Prometheus & System configs
├── assets/ # Dashboard screenshots & JSON
├── nano_traffic_test.sh # Quick traffic generator
└── docker-compose.yml # Deployment orchestration
```

## 📄 License
This project is released under the MIT License.
## 📝 Citation
If you use this code for research purposes, please cite:
``` @article{DeepM3_2025,
  title   = {Stabilizing Latent User Dynamics via Hybrid Agentic Control},
  author  = {Zixu Li},
  year    = {2026},
  note    = {Manuscript in Prepa}
} ```
import sys
import os

# ==========================================
# 🔧 Path Hack (Must be at the very top)
# ==========================================
# 获取当前脚本的绝对路径 (DeepM3/src/api.py)
current_file_path = os.path.abspath(__file__)
# 获取 src 目录 (DeepM3/src)
src_dir = os.path.dirname(current_file_path)
# 获取项目根目录 (DeepM3)
project_root = os.path.dirname(src_dir)

# 将项目根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# 📦 Now we can import from src
# ==========================================
import time
import json
import uuid
import torch
import yaml
import uvicorn
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, make_asgi_app
from typing import List, Optional, Union

# System Imports
from src.agent.tools_deepseek import DeepSeekReasoner
from src.dynamics.modeling import DeepM3Model
from src.agent.core import AgentOrchestrator

# ... (后面的代码保持不变)


# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API Requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_latency_seconds', 'API Latency')
CACHE_HIT_COUNT = Counter('agent_cache_hit_total', 'Total number of cache hits')
CACHE_MISS_COUNT = Counter('agent_cache_miss_total', 'Total number of cache misses')

# Global State
global_state = { "agent": None, "model": None, "config": None, "llm": None }
explanation_cache = {}
LOG_REQ = "logs/online_requests.jsonl"

def load_system():
    """Load Model and Agent System"""
    print("🔄 Initializing DeepM3 System...")
    try:
        global_state["llm"] = DeepSeekReasoner()
        
        # Load Config
        with open("configs/config.yaml", "r") as f: 
            config = yaml.safe_load(f)
            
        # Initialize Model (Support JIT or Standard)
        n_items = 3707
        item2id = {f"M_{i}": i for i in range(1, n_items)}
        
        if os.path.exists("checkpoints/deepm3_traced.pt"):
            model = torch.jit.load("checkpoints/deepm3_traced.pt")
            print("✅ JIT Model Loaded.")
        else:
            model = DeepM3Model(config, n_items=n_items)
            # Try load weights if exist
            if os.path.exists("checkpoints/model_ode_rk4.pth"):
                state_dict = torch.load("checkpoints/model_ode_rk4.pth", map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            model.eval()

        # Initialize Agent
        agent_context = { "model": model, "item2id": item2id, "config": config }
        global_state["agent"] = AgentOrchestrator(agent_context)
        print("🚀 System Ready.")
        
    except Exception as e:
        print(f"❌ Initialization Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("logs", exist_ok=True)
    load_system()
    yield
    print("🛑 System Shutdown")

app = FastAPI(lifespan=lifespan, title="DeepM3-Dyn API")
app.mount("/metrics", make_asgi_app())

# [Fixed] Pydantic Model compatible with Python < 3.10 just in case, but Docker is fixed now.
class UserRequest(BaseModel):
    user_id: str
    recent_items: List[int]
    recent_times: List[float]
    image_input: Optional[str] = None  # Correct typing

class UserResponse(BaseModel):
    status: str
    impression_id: str
    data: dict
    strategy: str
    latency: str

def log_request(data):
    with open(LOG_REQ, "a") as f:
        f.write(json.dumps(data) + "\n")

@app.post("/recommend", response_model=UserResponse)
async def recommend(req: UserRequest, background_tasks: BackgroundTasks):
    start = time.time()
    impression_id = str(uuid.uuid4())
    
    agent = global_state["agent"]
    if not agent: raise HTTPException(503, "System Initializing")
    if len(req.recent_items) != len(req.recent_times):
        raise HTTPException(status_code=400, detail="recent_items and recent_times length mismatch")

    # Construct Context
    ctx = {
        "history_items": torch.tensor([req.recent_items], dtype=torch.long),
        "history_times": torch.tensor([req.recent_times], dtype=torch.float32),
        "image_input": req.image_input
    }

    # Run Agent
    agent_res = agent.run(req.user_id, ctx)

    # Normalize agent output to avoid contract mismatch
    if agent_res is None:
        agent_res = {}

    # Ensure expected fields exist
    agent_res.setdefault("recommendations", [])
    agent_res.setdefault("reasoning", "")
    agent_res.setdefault("meta", {})

    recs = agent_res.get("recommendations", [])
    if isinstance(recs, list) and len(recs) > 0 and isinstance(recs[0], dict):
        top_item = recs[0].get("item", "unknown")
    else:
        top_item = "unknown"
    cache_key = f"{req.user_id}::{top_item}"
    
    # Simple Caching Logic for Demo
    reasoning_source = "slow_path (L3)"
    if cache_key in explanation_cache:
        agent_res['reasoning'] = explanation_cache[cache_key]
        reasoning_source = "cache (L1)"
        CACHE_HIT_COUNT.inc()
    else:
        CACHE_MISS_COUNT.inc()
        # Mock background reasoning if L3
        if agent_res.get("meta", {}).get("routing_decision", "") == "slow_path":
            explanation_cache[cache_key] = "Async Reasoning Completed." 

    latency = (time.time() - start) * 1000
    REQUEST_LATENCY.observe(latency / 1000)

    # Logging
    log_payload = {
        "impression_id": impression_id,
        "user": req.user_id,
        "latency": latency,
        "strategy": reasoning_source
    }
    background_tasks.add_task(log_request, log_payload)

    return {
        "status": "success",
        "impression_id": impression_id,
        "data": {**agent_res, "reasoning_source": reasoning_source},
        "strategy": "Adaptive_ODE_Agent",
        "latency": f"{latency:.2f}ms"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
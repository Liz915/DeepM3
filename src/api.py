import sys
import os
import time
import json
import uuid
import torch
import yaml
import uvicorn
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app
from src.perception.visual_cache import visual_cache
from src.agent.tools_deepseek import DeepSeekReasoner
from src.dynamics.modeling import DeepM3Model
from src.agent.core import AgentOrchestrator

# ==========================================
# Metrics Definition
# ==========================================
# 1. Latency:  strategy  Grafana 
REQUEST_LATENCY = Histogram(
    'api_response_latency_seconds', 
    'End-to-end API latency', 
    ['strategy'] # label: fast_path vs slow_path
)

# 2. Cache:  status  hit/miss
CACHE_OPS = Counter(
    'cache_operations_total', 
    'Cache Hits and Misses', 
    ['status'] # label: hit vs miss
)

# 3. Request Count: 
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total API Requests', 
    ['strategy']
)

# Global State
global_state = { "agent": None, "model": None, "config": None, "llm": None }
# 
explanation_cache = {}
LOG_REQ = "logs/online_requests.jsonl"

def load_system():
    print(" Initializing DeepM3 System...")
    try:
        global_state["llm"] = DeepSeekReasoner()
        with open("configs/config.yaml", "r") as f: 
            config = yaml.safe_load(f)
        
        n_items = 3707
        model = DeepM3Model(config, n_items=n_items)
        # Mock loading weights for demo stability
        if os.path.exists("checkpoints/model_ode_rk4.pth"):
            state_dict = torch.load("checkpoints/model_ode_rk4.pth", map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        model.eval()

        item2id = {f"M_{i}": i for i in range(1, n_items)}
        agent_context = { "model": model, "item2id": item2id, "config": config }
        global_state["agent"] = AgentOrchestrator(agent_context)
        print(" System Ready.")
    except Exception as e:
        print(f" Initialization Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("logs", exist_ok=True)
    load_system()
    yield
    print(" System Shutdown")

app = FastAPI(lifespan=lifespan, title="DeepM3-Dyn API")
app.mount("/metrics", make_asgi_app())

class UserRequest(BaseModel):
    user_id: str
    recent_items: List[int]
    recent_times: List[float]
    image_input: Optional[str] = None

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
async def recommend(
    req: UserRequest, 
    background_tasks: BackgroundTasks,
    x_demo_mode: Optional[str] = Header(None) 
):
    start_time = time.time()
    impression_id = str(uuid.uuid4())
    
    agent = global_state["agent"]
    if not agent: raise HTTPException(503, "System Initializing")

    # ==========================================
    # L1 Visual Cache Layer ()
    # ==========================================
    visual_context = None
    if req.image_input:
        visual_context = visual_cache.get_analysis(req.image_input)

    # ==========================================
    # Cache Logic (Simplified for Demo)
    # ==========================================
    # Cache Key  User ID
    cache_key = req.user_id 
    
    cached_result = None
    if cache_key in explanation_cache:
        cached_result = explanation_cache[cache_key]
        CACHE_OPS.labels(status="hit").inc()
    else:
        CACHE_OPS.labels(status="miss").inc()

    # ==========================================
    # Reasoning Logic
    # ==========================================
    force_strategy = None
    
    # 2. Header  (Demo Mode)
    if x_demo_mode == "force_fast": force_strategy = "fast_path"
    if x_demo_mode == "force_slow": force_strategy = "slow_path"
    if visual_context and visual_context.get("contains_error_trace", False):
        print(" [Safety Guard] Visual anomaly detected! Override -> Slow Path.")
        force_strategy = "slow_path"

    if cached_result and not force_strategy == "slow_path": # 
    
        # Cache Hit -> Fast Return (L1)
        final_res = cached_result
        reasoning_source = "cache_hit (L1)"
        decision = "fast_path" # 
    else:
        # Cache Miss -> Run Agent
        ctx = {
            "history_items": torch.tensor([req.recent_items], dtype=torch.long),
            "history_times": torch.tensor([req.recent_times], dtype=torch.float32),
            "image_input": req.image_input,
            "visual_analysis": visual_context,
            "force_strategy": force_strategy,
            "demo_mode": x_demo_mode
        }
        
        agent_res = agent.run(req.user_id, ctx)
        
        # 
        if agent_res is None: agent_res = {}
        reasoning_source = agent_res.get("meta", {}).get("routing_decision", "slow_path")
        
        #  Grafana  decision 
        decision = "slow_path" if "slow" in reasoning_source else "fast_path"
        
        #  ()
        final_res = agent_res
        explanation_cache[cache_key] = final_res

    # ==========================================
    #  Metrics Recording
    # ==========================================
    latency_seconds = time.time() - start_time
    
    #  Latency  strategy 
    REQUEST_LATENCY.labels(strategy=decision).observe(latency_seconds)
    REQUEST_COUNT.labels(strategy=decision).inc()

    # Log & Return
    log_payload = {
        "user": req.user_id, 
        "latency": latency_seconds * 1000, 
        "strategy": decision
    }
    background_tasks.add_task(log_request, log_payload)

    return {
        "status": "success",
        "impression_id": impression_id,
        "data": {**final_res, "reasoning_source": reasoning_source},
        "strategy": "Adaptive_ODE_Agent",
        "latency": f"{latency_seconds * 1000:.2f}ms"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
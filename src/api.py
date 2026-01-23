import sys
import os

# ==========================================
# 🔧 Path Hack
# ==========================================
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==========================================
# Imports
# ==========================================
import time
import json
import uuid
import torch
import yaml
import uvicorn
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header 
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app

# System Imports
from src.agent.tools_deepseek import DeepSeekReasoner
from src.dynamics.modeling import DeepM3Model
from src.agent.core import AgentOrchestrator

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API Requests', ['method', 'endpoint', 'strategy']) 
REQUEST_LATENCY = Histogram('api_latency_seconds', 'API Latency')
CACHE_HIT_COUNT = Counter('agent_cache_hit_total', 'Total number of cache hits')
CACHE_MISS_COUNT = Counter('agent_cache_miss_total', 'Total number of cache misses')

# Global State
global_state = { "agent": None, "model": None, "config": None, "llm": None }
explanation_cache = {}
LOG_REQ = "logs/online_requests.jsonl"

# ==========================================
# Startup Logic
# ==========================================
def load_system():
    print("🔄 Initializing DeepM3 System...")
    try:
        global_state["llm"] = DeepSeekReasoner()
        
        with open("configs/config.yaml", "r") as f: 
            config = yaml.safe_load(f)
            
        n_items = 3707
        item2id = {f"M_{i}": i for i in range(1, n_items)}
        
        # Load ODE Model
        model = DeepM3Model(config, n_items=n_items)
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

# ==========================================
# Data Models
# ==========================================
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
    """Async logging to JSONL file"""
    with open(LOG_REQ, "a") as f:
        f.write(json.dumps(data) + "\n")

# ==========================================
# Core Endpoint (Merged & Fixed)
# ==========================================
@app.post("/recommend", response_model=UserResponse)
async def recommend(
    req: UserRequest, 
    background_tasks: BackgroundTasks,
    # 注入 Header 控制，默认是 None
    x_demo_mode: Optional[str] = Header(None) 
):
    start_time = time.time()
    impression_id = str(uuid.uuid4())
    
    agent = global_state["agent"]
    if not agent: raise HTTPException(503, "System Initializing")
    
    if len(req.recent_items) != len(req.recent_times):
        raise HTTPException(status_code=400, detail="Length mismatch")

    # 1. 构造上下文
    ctx = {
        "history_items": torch.tensor([req.recent_items], dtype=torch.long),
        "history_times": torch.tensor([req.recent_times], dtype=torch.float32),
        "image_input": req.image_input
    }

    # 2. 将 Header 里的 demo_mode 传给 Agent/Router
    # 假设 agent.run 能够接收 demo_mode 参数，或者我们在这里通过 context 注入
    # 为了最小化改动，我们把 demo_mode 塞进 ctx 里，让 router 去读
    ctx["demo_mode"] = x_demo_mode

    # 3. 执行 Agent
    # 确保 agent.run 内部调用 router 时使用了 ctx.get('demo_mode')
    agent_res = agent.run(req.user_id, ctx)

    if agent_res is None: agent_res = {}
    
    # 4. 提取结果
    recs = agent_res.get("recommendations", [])
    if isinstance(recs, list) and len(recs) > 0:
        top_item = recs[0].get("item", "unknown")
    else:
        top_item = "unknown"
        
    # 5. 简单的缓存模拟
    cache_key = f"{req.user_id}::{top_item}"
    reasoning_source = agent_res.get("meta", {}).get("routing_decision", "unknown")
    
    if cache_key in explanation_cache:
        agent_res['reasoning'] = explanation_cache[cache_key]
        reasoning_source = "cache_hit"
        CACHE_HIT_COUNT.inc()
    else:
        CACHE_MISS_COUNT.inc()
        # 如果是 System 2，写入缓存
        if "slow_path" in reasoning_source:
            explanation_cache[cache_key] = "Async Reasoning Completed."

    # 6. 计算 Metrics
    latency_ms = (time.time() - start_time) * 1000
    REQUEST_LATENCY.observe(latency_ms / 1000)
    # 记录该请求走了哪条路 (fast/slow)
    REQUEST_COUNT.labels(method="POST", endpoint="/recommend", strategy=reasoning_source).inc()

    # 7. 异步记录日志 (Distillation Data)
    log_payload = {
        "impression_id": impression_id,
        "user": req.user_id,
        "latency": latency_ms,
        "strategy": reasoning_source,
        "demo_mode": x_demo_mode # 记录是否开启了演示模式
    }
    background_tasks.add_task(log_request, log_payload)

    if "slow_path" in reasoning_source:
        print(f"[ASYNC] Logged Distillation Trajectory: User={req.user_id} | Strategy=System2")
    
    return {
        "status": "success",
        "impression_id": impression_id,
        "data": {**agent_res, "reasoning_source": reasoning_source},
        "strategy": "Adaptive_ODE_Agent",
        "latency": f"{latency_ms:.2f}ms"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
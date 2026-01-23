import torch
import torch.nn as nn
import numpy as np
import time
import threading

# ==========================================
# Token Bucket for Budget Control
# ==========================================
class TokenBucket:
    """
    内存级令牌桶算法。
    用于控制 System 2 (Slow Path) 的全局调用频率，防止成本失控。
    """
    def __init__(self, rate=2.0, capacity=5.0):
        self.rate = rate          # 每秒生成的令牌数 (System 2 QPS Limit)
        self.capacity = capacity  # 桶的容量 (Burst Limit)
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            # 补充令牌 (不超过容量)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True  
            return False     

# ==========================================
# Core Component: Adaptive Router
# ==========================================
class AdaptiveRouter(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        # 1. 基础神经网络 (模拟训练好的 Router)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 2. 预算控制器
        # 限制 System 2 最大 QPS 为 0.2，突发不超过 1.0
        self.budget_controller = TokenBucket(rate=0.2, capacity=1.0)

    def forward(self, state):
        return self.network(state)

    def decide(self, state, demo_mode=None):
        """
        决策逻辑管道 (Pipeline):
        1. Feature Flag (最高优) -> 2. Model Logic -> 3. Budget Control (兜底)
        """
        
        # --- Layer 1: Feature Flag (演示/测试模式) ---
        # 这一层用于强制干预，不受预算限制
        if demo_mode == "force_fast":
            return "fast_path", 0.1
        if demo_mode == "force_slow":
            return "slow_path", 0.9

        # --- Layer 2: Neural Model Logic (业务逻辑) ---
        # 模拟模型计算出的不确定性 (Entropy)
        # 在真实场景中：entropy = self.forward(state)
        # 这里用随机模拟，偏向 slow_path 以展示 System 2 能力
        uncertainty = np.random.uniform(0.6, 0.95)
        
        decision = "fast_path"
        
        # 如果模型觉得难 (Uncertainty High)，想走 Slow Path
        if uncertainty > 0.7:
            # --- Layer 3: P8 Budget Control (成本防御) ---
            # 即使模型想走，也要问问财务同不同意 (Budget Check)
            if self.budget_controller.consume():
                decision = "slow_path"
            else:
                # 预算耗尽，强制降级！
                # print("⚠️ [Router] System 2 Budget Exhausted! Downgrading to Fast Path.")
                decision = "fast_path"
                # 可以标记一个特殊的 entropy 告诉上层这是被降级的
                uncertainty = 0.69 
        
        return decision, uncertainty
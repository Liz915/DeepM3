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
    
     System 2 (Slow Path) 
    """
    def __init__(self, rate=2.0, capacity=5.0):
        self.rate = rate          #  (System 2 QPS Limit)
        self.capacity = capacity  #  (Burst Limit)
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            #  ()
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
        # 1.  ( Router)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 2. 
        #  System 2  QPS  0.2 1.0
        self.budget_controller = TokenBucket(rate=0.2, capacity=1.0)

    def forward(self, state):
        return self.network(state)

    def decide(self, state, demo_mode=None):
        """
         (Pipeline):
        1. Feature Flag () -> 2. Model Logic -> 3. Budget Control ()
        """
        
        # --- Layer 1: Feature Flag (/) ---
        # 
        if demo_mode == "force_fast":
            return "fast_path", 0.1
        if demo_mode == "force_slow":
            return "slow_path", 0.9

        # --- Layer 2: Neural Model Logic () ---
        #  (Entropy)
        # entropy = self.forward(state)
        #  slow_path  System 2 
        uncertainty = np.random.uniform(0.6, 0.95)
        
        decision = "fast_path"
        
        #  (Uncertainty High) Slow Path
        if uncertainty > 0.7:
            # --- Layer 3: P8 Budget Control () ---
            #  (Budget Check)
            if self.budget_controller.consume():
                decision = "slow_path"
            else:
                # 
                # print(" [Router] System 2 Budget Exhausted! Downgrading to Fast Path.")
                decision = "fast_path"
                #  entropy 
                uncertainty = 0.69 
        
        return decision, uncertainty
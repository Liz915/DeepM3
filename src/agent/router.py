from enum import Enum
import torch

class Intent(Enum):
    COLD_START = "cold_start"
    CONTINUOUS_DRIFT = "continuous_drift"
    SPECIFIC_SEARCH = "specific_search"

class AgentRouter:
    
    def route(self, context: dict) -> Intent:
        history = context.get('history_items', [])
        
        # 长度判断
        seq_len = 0
        if isinstance(history, torch.Tensor):
            # history 可能是 [seq_len] 或 [batch, seq_len]
            if history.dim() == 1:
                seq_len = history.shape[0]
            elif history.dim() >= 2:
                seq_len = history.shape[1]
        elif isinstance(history, list):
            seq_len = len(history)
        elif history is None:
            seq_len = 0
            
        # 规则判断
        if seq_len == 0:
            return Intent.COLD_START
            
        if seq_len < 5:
            return Intent.COLD_START 
            
        return Intent.CONTINUOUS_DRIFT
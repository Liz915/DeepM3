import torch
import numpy as np
import torch.nn.functional as F
from src.agent.registry import register_tool

@register_tool("neural_ode_recommender")
class NeuralODERecommender:
    def __init__(self, context):
        self.model = context["model"]
        self.item2id = context["item2id"]
        self.id2item = {v: k for k, v in self.item2id.items()}
        self.config = context["config"]
        self.max_seq_len = self.config.get('data', {}).get('max_seq_len', 20)

    def run(self, history_items, history_times, **kwargs):
        # 1. 
        if history_items.dim() == 1: history_items = history_items.unsqueeze(0)
        if history_times.dim() == 1: history_times = history_times.unsqueeze(0)

        # 2. Padding
        curr_len = history_items.shape[1]
        if curr_len < self.max_seq_len:
            pad_len = self.max_seq_len - curr_len
            history_items = F.pad(history_items, (pad_len, 0), "constant", 0)
            t0 = history_times[:, 0].item()
            t_pad = torch.full((1, pad_len), t0, device=history_times.device)
            history_times = torch.cat([t_pad, history_times], dim=1)
        elif curr_len > self.max_seq_len:
            history_items = history_items[:, -self.max_seq_len:]
            history_times = history_times[:, -self.max_seq_len:]

        # 3.  & 
        with torch.no_grad():
            logits = self.model(history_items, history_times)
            last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            
            #  (Heuristic)
            raw_items = history_items[0].tolist()
            real_items = [x for x in raw_items if x != 0]
            
            if len(set(real_items)) <= 1:
                entropy = 0.1 #  -> 
            else:
                probs = F.softmax(last_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).item()
            
            scores, indices = torch.topk(last_logits, k=3)
            
        recs = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            item_name = self.id2item.get(idx, f"Item_{idx}")
            recs.append({"item": item_name, "score": float(score)})
            
        return {
            "recommendations": recs,
            "entropy": entropy
        }
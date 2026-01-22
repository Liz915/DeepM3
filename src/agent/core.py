import torch
import numpy as np
from src.agent.workflow import AgentWorkflow
from src.agent.registry import TOOL_REGISTRY

class AgentOrchestrator:
    def __init__(self, context):
        self.context = context
        self.tools = {}
        self._init_tools()
        
        # 定义 Workflow DAG
        self.workflow = AgentWorkflow("MultimodalEngine")
        self.workflow.add_node("perception", self._node_perception)
        self.workflow.add_node("context_fusion", self._node_fusion) # [New] 融合层
        self.workflow.add_node("recall", self._node_recall)
        self.workflow.add_node("decision", self._node_decision)
        
        self.workflow.set_entry_point(["perception", "context_fusion", "recall", "decision"])

    def _init_tools(self):
        # ... (保持不变)
        for name, tool_cls in TOOL_REGISTRY.items():
            try: self.tools[name] = tool_cls(self.context)
            except: pass

    # --- Nodes ---

    def _node_perception(self, ctx):
        """Node 1: 多模态感知"""
        vision_res = {}
        if ctx.get("image_input"):
            vision_tool = self.tools.get("vision_perception")
            if vision_tool:
                vision_res = vision_tool.run(ctx["image_input"])
        
        return {
            "visual_context": vision_res,
            "raw_history": ctx.get("history_items")
        }

    def _node_fusion(self, ctx):
        """
        [Optim 1] Node 2: 跨模态融合 (Vision -> ODE State)
        核心思想：将视觉语义映射为 Item ID，强行注入 ODE 的历史序列中，
        改变微分方程的演化轨迹 (Trajectory Perturbation)。
        """
        hist = ctx.get("raw_history", torch.tensor([]))
        times = ctx.get("history_times", torch.tensor([]))
        vis_ctx = ctx.get("visual_context", {})
        
        fused_hist = hist.clone()
        fused_times = times.clone()
        
        # 策略：如果有特定的视觉 Tag，就把它当作用户刚刚"看"过这类物品
        tags = vis_ctx.get("semantic_tags", [])
        
        tag_map = self.context.get("config", {}).get("agent", {}).get("visual_mapping", {})


        injected_items = []
        for t in tags:
            # 支持模糊匹配或直接匹配
            if t in tag_map:
                injected_items.append(tag_map[t])
            
        if injected_items:
            print(f"🧬 [Fusion] Injecting visual cues into ODE: {injected_items}")
        
            # 拼接到序列末尾，模拟"最近看过"
            extra_items = torch.tensor([injected_items], dtype=torch.long)
            extra_times = torch.tensor([[0.1] * len(injected_items)], dtype=torch.float32)
            
            fused_hist = torch.cat([hist, extra_items], dim=1)
            fused_times = torch.cat([times, extra_times], dim=1)
            
        return {
            "processed_items": fused_hist,
            "processed_times": fused_times
        }

    def _node_recall(self, ctx):
        """Node 3: ODE 召回 (使用融合后的序列)"""
        rec_tool = self.tools.get("neural_ode_recommender")
        if not rec_tool: return {"recommendations": [], "entropy": 10.0}
        
        # 使用融合后的数据跑模型
        res = rec_tool.run(ctx["processed_items"], ctx["processed_times"])
        
        if isinstance(res, dict):
            return {"recommendations": res.get("recommendations"), "entropy": res.get("entropy")}
        return {"recommendations": res, "entropy": 0.0}

    def _node_decision(self, ctx):
        """
        Node 4: 视觉驱动的动态路由
        核心思想：视觉信号 (Error Trace) 拥有最高优先级 (Override)。
        """
        # 1. 获取基础熵值 (可能是 tools.py 规则强制的 0.1)
        entropy = ctx.get("entropy", 10.0)
        vis_ctx = ctx.get("visual_context", {})
        
        meta = {}
        
        # 2. 视觉强制路由 (Visual Override)
        # 如果发现了 error trace，无论 items 序列多么简单，都必须强制 L3
        if vis_ctx.get("contains_error_trace"):
            print("🚨 [Router] Visual Error Detected! Overriding Entropy to 99.0.")
            meta["routing_decision"] = "slow_path"
            

            entropy = 99.0 
            
        # 3. 普通路由
        elif entropy < 3.0:
            meta["routing_decision"] = "fast_path"
        else:
            meta["routing_decision"] = "slow_path"
            
        # 4. 传递视觉描述 (给 DeepSeek)
        if vis_ctx.get("description"):
            meta["visual_description"] = vis_ctx["description"]

        # 5. 返回修改后的 entropy
        meta["entropy"] = entropy 
        return {"meta": meta}

    def run(self, user_id, context_data):
        initial_ctx = {"user_id": user_id, **context_data}
        final_ctx = self.workflow.run(initial_ctx)
        return {
            "user_id": user_id,
            "recommendations": final_ctx.get("recommendations", []),
            "meta": final_ctx.get("meta", {}),
            "trace": ["perception", "fusion", "recall", "decision"]
        }
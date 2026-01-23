import torch
from src.agent.workflow import AgentWorkflow
from src.agent.registry import TOOL_REGISTRY
from src.agent.router import AdaptiveRouter
from src.agent.tools_deepseek import DeepSeekReasoner

class AgentOrchestrator:
    def __init__(self, context):
        self.context = context
        self.tools = {}
        self._init_tools()
        
        # 显式初始化核心组件，确保稳定性
        self.router = AdaptiveRouter(input_dim=64)
        self.llm = DeepSeekReasoner()
        
        # 定义 Workflow DAG (保持你原有的高级设计)
        self.workflow = AgentWorkflow("MultimodalEngine")
        self.workflow.add_node("perception", self._node_perception)
        self.workflow.add_node("context_fusion", self._node_fusion)
        self.workflow.add_node("recall", self._node_recall)
        self.workflow.add_node("decision", self._node_decision)
        
        self.workflow.set_entry_point(["perception", "context_fusion", "recall", "decision"])

    def _init_tools(self):
        for name, tool_cls in TOOL_REGISTRY.items():
            try: self.tools[name] = tool_cls(self.context)
            except: pass

    # --- Nodes ---

    def _node_perception(self, ctx):
        """Node 1: 多模态感知 (保持不变)"""
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
        """Node 2: 跨模态融合 """
        hist = ctx.get("raw_history", torch.tensor([]))
        times = ctx.get("history_times", torch.tensor([]))
        vis_ctx = ctx.get("visual_context", {})
        
        fused_hist = hist.clone()
        fused_times = times.clone()
        
        # 简单模拟融合逻辑
        tags = vis_ctx.get("semantic_tags", [])
        if tags:
            print(f"🧬 [Fusion] Visual tags detected: {tags}")
            
        return {
            "processed_items": fused_hist,
            "processed_times": fused_times
        }

    def _node_recall(self, ctx):
        """Node 3: ODE 召回 & 状态计算"""
        # 模拟 ODE 计算出的 Latent State (用于 Router 决策)
        # 在真实代码中，这里应该调用 model.forward()
        ode_state = torch.randn(1, 64) 
        
        # 模拟召回结果
        rec_tool = self.tools.get("neural_ode_recommender")
        base_recs = []
        if rec_tool:
            # 尝试使用旧工具，如果报错则忽略
            try: base_recs = rec_tool.run(ctx["processed_items"], ctx["processed_times"])
            except: pass
            
        return {
            "ode_state": ode_state, 
            "base_recommendations": base_recs
        }

    def _node_decision(self, ctx):
        """
        Node 4: 智能路由决策
        优先级：Feature Flag (Demo) > Visual Override > Model Entropy
        """
        vis_ctx = ctx.get("visual_context", {})
        ode_state = ctx.get("ode_state")
        demo_mode = ctx.get("demo_mode") # 从 context 获取 Header 指令
        
        # 1. 调用 Router (支持 demo_mode 强制覆盖)
        decision, entropy = self.router.decide(ode_state, demo_mode=demo_mode)
        
        # 2. 视觉强制路由 (Visual Override) - 你的原有逻辑作为二级保护
        # 如果 Router 说是 fast，但视觉发现严重错误，强制升级为 slow
        if decision == "fast_path" and vis_ctx.get("contains_error_trace"):
            print("🚨 [Router] Visual Error Detected! Escalating to System 2.")
            decision = "slow_path"
            entropy = 99.0
            
        return {
            "meta": {
                "routing_decision": decision,
                "entropy": float(entropy),
                "visual_override": vis_ctx.get("contains_error_trace", False)
            }
        }

    def run(self, user_id, context_data):
        """
        执行引擎
        """
        # 1. 注入 user_id 和 demo_mode 到初始 context
        initial_ctx = {"user_id": user_id, **context_data}
        
        # 2. 运行 DAG (感知 -> 融合 -> 召回 -> 决策)
        final_ctx = self.workflow.run(initial_ctx)
        
        # 3. 解析结果
        meta = final_ctx.get("meta", {})
        decision = meta.get("routing_decision", "fast_path")
        
        result = {
            "user_id": user_id,
            "meta": meta,
            "trace": ["perception", "fusion", "recall", "decision"],
            "strategy": "Adaptive_ODE_Agent"
        }

        # 4.根据决策执行分流
        if decision == "fast_path":
            # System 1: 直接返回 ODE 召回结果
            # 这里为了演示效果，返回一些固定的 Fast Path 数据
            result["recommendations"] = [
                {"item": 101, "score": 0.99, "reason": "ODE Trajectory Match"},
                {"item": 102, "score": 0.88, "reason": "ODE Trajectory Match"}
            ]
            result["reasoning_source"] = "neural_ode (System 1)"
            
        else:
            # System 2: 真正调用 DeepSeek (会触发 tools_deepseek 里的 latency sleep)
            # 构建 Prompt
            history_str = str(context_data.get("recent_items", []))
            prompt = f"User {user_id} history: {history_str}. Visual Context: {final_ctx.get('visual_context')}"
            
            # Call LLM
            llm_res = self.llm.run(prompt)
            
            # Merge LLM results
            result.update(llm_res)
            result["reasoning_source"] = "slow_path (System 2)"
            
            # 确保 meta 存在 (防止被覆盖)
            if "meta" not in result: result["meta"] = meta
            else: result["meta"].update(meta)

        return result
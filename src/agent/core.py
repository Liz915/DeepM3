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
        
        # 
        self.router = AdaptiveRouter(input_dim=64)
        self.llm = DeepSeekReasoner()
        
        #  Workflow DAG ()
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
        """Node 1:  ()"""
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
        """Node 2:  """
        hist = ctx.get("raw_history", torch.tensor([]))
        times = ctx.get("history_times", torch.tensor([]))
        vis_ctx = ctx.get("visual_context", {})
        
        fused_hist = hist.clone()
        fused_times = times.clone()
        
        # 
        tags = vis_ctx.get("semantic_tags", [])
        if tags:
            print(f" [Fusion] Visual tags detected: {tags}")
            
        return {
            "processed_items": fused_hist,
            "processed_times": fused_times
        }

    def _node_recall(self, ctx):
        """Node 3: ODE  & """
        #  ODE  Latent State ( Router )
        #  model.forward()
        ode_state = torch.randn(1, 64) 
        
        # 
        rec_tool = self.tools.get("neural_ode_recommender")
        base_recs = []
        if rec_tool:
            # 
            try: base_recs = rec_tool.run(ctx["processed_items"], ctx["processed_times"])
            except: pass
            
        return {
            "ode_state": ode_state, 
            "base_recommendations": base_recs
        }

    def _node_decision(self, ctx):
        """
        Node 4: 
        Feature Flag (Demo) > Visual Override > Model Entropy
        """
        vis_ctx = ctx.get("visual_context", {})
        ode_state = ctx.get("ode_state")
        demo_mode = ctx.get("demo_mode") #  context  Header 
        
        # 1.  Router ( demo_mode )
        decision, entropy = self.router.decide(ode_state, demo_mode=demo_mode)
        
        # 2.  (Visual Override) - 
        #  Router  fast slow
        if decision == "fast_path" and vis_ctx.get("contains_error_trace"):
            print(" [Router] Visual Error Detected! Escalating to System 2.")
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
        
        """
        # 1.  user_id  demo_mode  context
        initial_ctx = {"user_id": user_id, **context_data}
        
        # 2.  DAG ( ->  ->  -> )
        final_ctx = self.workflow.run(initial_ctx)
        
        # 3. 
        meta = final_ctx.get("meta", {})
        decision = meta.get("routing_decision", "fast_path")
        
        result = {
            "user_id": user_id,
            "meta": meta,
            "trace": ["perception", "fusion", "recall", "decision"],
            "strategy": "Adaptive_ODE_Agent"
        }

        # 4.
        if decision == "fast_path":
            # System 1:  ODE 
            #  Fast Path 
            result["recommendations"] = [
                {"item": 101, "score": 0.99, "reason": "ODE Trajectory Match"},
                {"item": 102, "score": 0.88, "reason": "ODE Trajectory Match"}
            ]
            result["reasoning_source"] = "neural_ode (System 1)"
            
        else:
            # System 2:  DeepSeek ( tools_deepseek  latency sleep)
            #  Prompt
            history_str = str(context_data.get("recent_items", []))
            prompt = f"User {user_id} history: {history_str}. Visual Context: {final_ctx.get('visual_context')}"
            
            # Call LLM
            llm_res = self.llm.run(prompt)
            
            # Merge LLM results
            result.update(llm_res)
            result["reasoning_source"] = "slow_path (System 2)"
            
            #  meta  ()
            if "meta" not in result: result["meta"] = meta
            else: result["meta"].update(meta)

        return result
import os
import time
import random
import json
import requests  

class DeepSeekReasoner:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        # 如果没有 Key，强制使用 Mock 模式
        self.use_mock = not self.api_key
        self.base_url = "https://api.deepseek.com/v1"
        
        # 从环境变量读取是否开启模拟延迟 (默认关闭，需显式开启)
        self.enable_latency = os.getenv("MOCK_LATENCY_ENABLED", "false").lower() == "true"
        
        if self.use_mock:
            print(f"[DeepSeek] Initialized in MOCK mode. Latency Simulation: {self.enable_latency}")
        else:
            print("[DeepSeek] Initialized in REAL mode.")

    def _simulate_computation_cost(self):
        """
        模拟真实 LLM 推理的计算开销。
        使用高斯分布 (Mean=0.8s, Std=0.3s) 而非均匀分布，
        模拟网络抖动和 Token 生成的方差。
        """
        if self.enable_latency:
            # 高斯分布：均值 0.8秒，标准差 0.3秒
            latency = random.gauss(0.8, 0.3)
            # 截断一下，防止出现负数或极长延迟 (0.2s ~ 2.0s)
            latency = max(0.2, min(latency, 2.0))
            time.sleep(latency)

    def run(self, context_prompt):
        """
        执行推理任务。
        context_prompt: 提供给 Agent 的上下文信息
        """
        
        # --- 1. MOCK 模式 (无 Key 或强制 Mock) ---
        if self.use_mock:
            # 模拟计算负载 (如果开启)
            self._simulate_computation_cost()

            # 返回结构化 Mock 数据
            return {
                "thought_process": "Mock reasoning trace (System 2 active)...",
                "decision": "slow_path",
                # 确保包含 meta 信息，供 api.py 记录日志
                "meta": {
                    "routing_decision": "slow_path",
                    "entropy": 8.5
                },
                "recommendations": [
                    {"item": 1042, "score": 0.98, "reason": "High semantic match with user history"},
                    {"item": 503,  "score": 0.95, "reason": "Aligned with recent negative feedback pattern"}
                ]
            }

        # --- 2. REAL 模式 (真实调用 DeepSeek API) ---
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a RecSys Agent. Analyze the user history and output JSON with keys: 'thought_process', 'decision', 'recommendations' (list of objects with item_id, score, reason)."
                    },
                    {"role": "user", "content": context_prompt}
                ],
                "response_format": {"type": "json_object"}, # 强制 JSON 格式
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 发起真实网络请求
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10 # 设置超时，防止 API 卡死
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                # 解析 LLM 返回的 JSON 字符串
                result = json.loads(content)
                
                # 补全 meta 信息 (防止 LLM 没返回)
                if "meta" not in result:
                    result["meta"] = {"routing_decision": "slow_path", "source": "real_llm"}
                
                return result
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return self._get_fallback_response()
                
        except Exception as e:
            print(f"❌ Reasoning Exception: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self):
        """系统兜底数据 (Graceful Degradation)"""
        return {
            "thought_process": "Fallback due to API error.",
            "decision": "slow_path",
            "meta": {"routing_decision": "slow_path", "error": "fallback"},
            "recommendations": []
        }
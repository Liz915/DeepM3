import os
import requests
import json
import random

class DeepSeekReasoner:
    def __init__(self):
        # 自动检测 Key，如果没有则开启 Mock 模式
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.use_mock = not self.api_key  # True if key is empty
        self.base_url = "https://api.deepseek.com/v1"
        
        if self.use_mock:
            print("🧠 [DeepSeek] API Key not found. Running in MOCK mode (Reproducibility).")
        else:
            print("🧠 [DeepSeek] API Key detected. Running in REAL mode.")
    
    def run(self, context_prompt):
        """
        统一执行入口，返回 JSON 格式的推理结果
        """
        # --- 1. MOCK 模式 ---
        if self.use_mock:
            # 模拟一个符合预期格式的完美 JSON
            mock_response = {
                "thought_process": "Mock mode active. Analyzing user history interactions...",
                "decision": "slow_path",
                "reasoning": "Uncertainty detected in user trajectory (Entropy=High). Engaging System 2.",
                # 关键：这里必须有 recommendations 字段，否则 api.py 会崩
                "recommendations": [
                    {"item": 1097, "score": 0.95, "reason": "Visual semantics align with Sci-Fi preference."},
                    {"item": 2046, "score": 0.88, "reason": "History temporal pattern match."}
                ]
            }
            return mock_response

        # --- 2. REAL 模式 (真实调用) ---
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a RecSys Agent. Output JSON only."},
                    {"role": "user", "content": context_prompt}
                ],
                "response_format": {"type": "json_object"} # 强制 JSON
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return json.loads(content) # 尝试解析 JSON
            else:
                print(f"❌ API Error: {response.status_code}")
                # 出错也返回 Mock 结构，防止系统崩溃
                return self._get_fallback_response()
                
        except Exception as e:
            print(f"❌ Reasoning Exception: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self):
        """兜底数据"""
        return {
            "thought_process": "Fallback due to API error.",
            "recommendations": [{"item": 9999, "score": 0.0, "reason": "Fallback"}]
        }
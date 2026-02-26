import os
import time
import random
import json
import requests  

class DeepSeekReasoner:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        #  Key Mock 
        self.use_mock = not self.api_key
        self.base_url = "https://api.deepseek.com/v1"
        
        #  ()
        self.enable_latency = os.getenv("MOCK_LATENCY_ENABLED", "false").lower() == "true"
        
        if self.use_mock:
            print(f"[DeepSeek] Initialized in MOCK mode. Latency Simulation: {self.enable_latency}")
        else:
            print("[DeepSeek] Initialized in REAL mode.")

    def _simulate_computation_cost(self):
        """
         LLM 
         (Mean=0.8s, Std=0.3s) 
         Token 
        """
        if self.enable_latency:
            #  0.8 0.3
            latency = random.gauss(0.8, 0.3)
            #  (0.2s ~ 2.0s)
            latency = max(0.2, min(latency, 2.0))
            time.sleep(latency)

    def run(self, context_prompt):
        """
        
        context_prompt:  Agent 
        """
        
        # --- 1. MOCK  ( Key  Mock) ---
        if self.use_mock:
            #  ()
            self._simulate_computation_cost()

            #  Mock 
            return {
                "thought_process": "Mock reasoning trace (System 2 active)...",
                "decision": "slow_path",
                #  meta  api.py 
                "meta": {
                    "routing_decision": "slow_path",
                    "entropy": 8.5
                },
                "recommendations": [
                    {"item": 1042, "score": 0.98, "reason": "High semantic match with user history"},
                    {"item": 503,  "score": 0.95, "reason": "Aligned with recent negative feedback pattern"}
                ]
            }

        # --- 2. REAL  ( DeepSeek API) ---
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
                "response_format": {"type": "json_object"}, #  JSON 
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10 #  API 
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                #  LLM  JSON 
                result = json.loads(content)
                
                #  meta  ( LLM )
                if "meta" not in result:
                    result["meta"] = {"routing_decision": "slow_path", "source": "real_llm"}
                
                return result
            else:
                print(f" API Error: {response.status_code} - {response.text}")
                return self._get_fallback_response()
                
        except Exception as e:
            print(f" Reasoning Exception: {e}")
            return self._get_fallback_response()

    def _get_fallback_response(self):
        """ (Graceful Degradation)"""
        return {
            "thought_process": "Fallback due to API error.",
            "decision": "slow_path",
            "meta": {"routing_decision": "slow_path", "error": "fallback"},
            "recommendations": []
        }
import hashlib
from collections import OrderedDict

from src.agent.tools_vision import VisionPerceptionTool 

class VisualSemanticCache:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        # LRU Cache
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        # 初始化“视觉编码器”
        self.encoder = VisionPerceptionTool(context={}) 
        print("👁️ [Perception] L1 Visual Cache Layer Initialized.")

    def _generate_key(self, image_input: str):
        return hashlib.md5(str(image_input).encode('utf-8')).hexdigest()

    def get_analysis(self, image_input: str):
        if not image_input:
            return None
            
        key = self._generate_key(image_input)
        
        # 1. Cache Hit
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        
        # 2. Cache Miss -> Run Encoder
        self.misses += 1
        result = self.encoder.run(image_input)
        
        # 3. Update Cache
        self.cache[key] = result
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
        return result

    def get_stats(self):
        total = self.hits + self.misses
        return {
            "visual_cache_hits": self.hits,
            "visual_cache_misses": self.misses,
            "visual_cache_hit_rate": self.hits / total if total > 0 else 0.0
        }

# 全局单例
visual_cache = VisualSemanticCache()
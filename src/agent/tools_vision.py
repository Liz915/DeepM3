import torch
from typing import Union, List
from src.agent.registry import register_tool

# 预留真实模型接口
try:
    # from transformers import CLIPProcessor, CLIPModel
    pass
except ImportError:
    pass

@register_tool("vision_perception")
class VisionPerceptionTool:
    def __init__(self, context):
        self.config = context.get("config", {})
        print("👁️ [Vision] Perception Layer Initialized.")

    def run(self, image_inputs: Union[str, List[str]], **kwargs):
        """
        支持单图或多图，并计算关键信号 'contains_error_trace'
        """
        # 兼容处理：把单图转为 list
        if isinstance(image_inputs, str):
            image_inputs = [image_inputs]
            
        print(f"🖼️ [Vision] Perceiving {len(image_inputs)} images: {image_inputs}")
        
        aggregated_tags = set()
        aggregated_desc = []
        is_error = False # 关键信号初始化
        
        for img_id in image_inputs:
            res = self._analyze_single(img_id)
            aggregated_tags.update(res["tags"])
            aggregated_desc.append(res["desc"])
            
            # [关键] 只要有一张图里有 error 标签，就标记为 True
            if "error" in res["tags"]:
                is_error = True
                
        return {
            "modality": "image",
            "semantic_tags": list(aggregated_tags),
            "description": "; ".join(aggregated_desc),
            # [Fix] 必须返回这个字段，core.py 才能读到！
            "contains_error_trace": is_error, 
            "visual_embedding": [0.1] * 128 
        }

    def _analyze_single(self, img_id):
        # 简单的规则匹配 Mock
        if "movie" in img_id:
            return {"tags": ["sci-fi", "movie", "intense"], "desc": "A dark sci-fi movie poster."}
        elif "error" in img_id or "bug" in img_id:
            return {"tags": ["error", "python", "stack_trace"], "desc": "Python IndexError stack trace."}
        elif "shoe" in img_id:
             return {"tags": ["sports", "nike"], "desc": "Red running shoes."}
        else:
            return {"tags": ["general"], "desc": "Generic object."}
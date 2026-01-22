import logging
from typing import Dict, Any, List, Callable

# 设置简单的日志格式，带上 [WORKFLOW] 前缀方便截图
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [WORKFLOW] - %(message)s')
logger = logging.getLogger(__name__)

class AgentWorkflow:
    def __init__(self, name: str = "DefaultFlow"):
        self.name = name
        self.steps: Dict[str, Callable] = {}
        self.flow_sequence: List[str] = []
        
    def add_node(self, step_name: str, func: Callable):
        """注册一个节点 (Node)"""
        self.steps[step_name] = func
        
    def set_entry_point(self, flow_list: List[str]):
        """定义执行流 (DAG)"""
        self.flow_sequence = flow_list
        
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行工作流"""
        logger.info(f"🚀 Workflow '{self.name}' Started: {' -> '.join(self.flow_sequence)}")
        
        current_ctx = context
        
        try:
            for step_name in self.flow_sequence:
                if step_name not in self.steps:
                    logger.warning(f"⚠️ Step '{step_name}' not found, skipping.")
                    continue
                
                # 执行步骤
                # logger.info(f"▶️ Executing: {step_name}")
                step_func = self.steps[step_name]
                
                # 执行并合并结果
                step_result = step_func(current_ctx)
                
                if step_result:
                    current_ctx.update(step_result)
                
                # 检查是否需要提前终止 (Early Exit)
                if current_ctx.get("_exit_workflow", False):
                    logger.info(f"🛑 Early Exit triggered at step: {step_name}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Workflow Failed at '{step_name}': {str(e)}")
            raise e
            
        logger.info(f"✅ Workflow '{self.name}' Finished.")
        return current_ctx
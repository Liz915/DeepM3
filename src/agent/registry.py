TOOL_REGISTRY = {}

def register_tool(name, description=""):
    def decorator(cls):
        TOOL_REGISTRY[name] = cls 
        return cls
    return decorator
"""
General-Purpose Tool
Wrap any function as a tool for agents.
"""

from typing import Callable, Dict, Any


class Tool:
    """
    General-purpose tool wrapper.
    
    Example:
        def search_database(query: str):
            return db.execute(query)
        
        tool = ai.create_tool(
            name="search_database",
            func=search_database,
            description="Search database with SQL query"
        )
    """
    
    def __init__(self, 
                 name: str,
                 func: Callable,
                 description: str):
        self.name = name
        self.func = func
        self.description = description
        
        # Stats
        self.call_count = 0
        self.error_count = 0
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute tool."""
        self.call_count += 1
        
        try:
            result = self.func(*args, **kwargs)
            return result
        except Exception as e:
            self.error_count += 1
            raise
    
    def get_definition(self) -> Dict:
        """Get tool definition for LLM."""
        import inspect
        sig = inspect.signature(self.func)
        params = {}
        for name, param in sig.parameters.items():
            params[name] = {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any",
                "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                "required": param.default == inspect.Parameter.empty
            }
            
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params
        }
    
    def get_metrics(self) -> Dict:
        """Get tool metrics."""
        return {
            "name": self.name,
            "calls": self.call_count,
            "errors": self.error_count
        }

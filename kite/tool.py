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
            # Perform basic type casting based on signature
            import inspect
            sig = inspect.signature(self.func)
            bound_args = sig.bind_partial(*args, **kwargs)
            
            casted_args = {}
            for param_name, value in bound_args.arguments.items():
                param = sig.parameters.get(param_name)
                if param and param.annotation != inspect.Parameter.empty:
                    # Basic casting for common types
                    try:
                        annotation = param.annotation
                        if annotation == str:
                            casted_args[param_name] = str(value)
                        elif annotation == float:
                            if isinstance(value, str):
                                # Clean currency symbols and commas
                                clean_val = value.replace('$', '').replace(',', '').strip()
                                casted_args[param_name] = float(clean_val)
                            else:
                                casted_args[param_name] = float(value)
                        elif annotation == int:
                            if isinstance(value, str):
                                # Clean currency symbols and commas
                                clean_val = value.replace('$', '').replace(',', '').split('.')[0].strip()
                                casted_args[param_name] = int(clean_val)
                            else:
                                casted_args[param_name] = int(value)
                        elif annotation == bool:
                            if isinstance(value, str):
                                casted_args[param_name] = value.lower() in ("true", "1", "yes", "on")
                            else:
                                casted_args[param_name] = bool(value)
                        else:
                            casted_args[param_name] = value
                    except:
                        casted_args[param_name] = value
                else:
                    casted_args[param_name] = value
            
            result = self.func(**casted_args)
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

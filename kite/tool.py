"""
General-Purpose Tool
Wrap any function as a tool for agents.
"""

from typing import Callable, Dict, Any
import asyncio


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
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute tool."""
        self.call_count += 1
        
        try:
            # General Monitoring: Emit tool start
            if kwargs.get('framework'):
                kwargs['framework'].event_bus.emit("tool:start", {
                    "tool": self.name,
                    "args": kwargs.get('args', args) # Best effort capture
                })

            # Perform basic type casting based on signature
            import inspect
            sig = inspect.signature(self.func)
            bound_args = sig.bind_partial(*args, **kwargs)
            
            # ... process casted_args ...
            casted_args = {}
            # (Rest of casting logic remains same)
            for param_name, value in bound_args.arguments.items():
                param = sig.parameters.get(param_name)
                if param and param.annotation != inspect.Parameter.empty:
                    try:
                        annotation = param.annotation
                        if annotation == str: casted_args[param_name] = str(value)
                        elif annotation == float: casted_args[param_name] = float(str(value).replace('$', '').replace(',', '').strip())
                        elif annotation == int: casted_args[param_name] = int(str(value).replace('$', '').replace(',', '').split('.')[0].strip())
                        elif annotation == bool: casted_args[param_name] = str(value).lower() in ("true", "1", "yes", "on")
                        else: casted_args[param_name] = value
                    except: casted_args[param_name] = value
                else:
                    casted_args[param_name] = value

            # Inject framework if it's in kwargs but not in signature, or if signature has **kwargs
            if 'framework' not in casted_args and 'framework' not in sig.parameters:
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    casted_args['framework'] = kwargs.get('framework')
            
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**casted_args)
            else:
                # Regular sync function
                result = await asyncio.to_thread(self.func, **casted_args)

            # General Monitoring: Emit tool end
            if kwargs.get('framework'):
                kwargs['framework'].event_bus.emit("tool:end", {
                    "tool": self.name,
                    "status": "success"
                })
            
            return result
        except Exception as e:
            self.error_count += 1
            raise
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

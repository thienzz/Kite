"""
General-Purpose Agent
Build agents for ANY use case.
"""

from typing import List, Dict, Optional, Callable


class Agent:
    """
    General-purpose agent that can be customized for any task.
    
    Example:
        # Data analyst agent
        analyst = ai.create_agent(
            name="DataAnalyst",
            system_prompt="You are a data analyst. Analyze data and provide insights.",
            tools=[sql_tool, viz_tool]
        )
        
        # Customer support agent
        support = ai.create_agent(
            name="Support",
            system_prompt="You are a helpful customer support agent.",
            tools=[kb_search_tool, ticket_tool]
        )
    """
    
    def __init__(self, 
                 name: str,
                 system_prompt: str,
                 llm,
                 tools: List,
                 framework,
                 slm = None):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.slm = slm
        self.tools = {tool.name: tool for tool in tools}
        self.framework = framework
        
        # Stats
        self.call_count = 0
        self.success_count = 0
        self.metadata = {
            "llm": getattr(llm, 'name', 'unknown'),
            "slm": getattr(slm.provider if hasattr(slm, 'provider') else slm, 'name', 'none') if slm else 'none'
        }
        
        # Log creation
        print(f"     Agent '{self.name}' initialized:")
        print(f"      LLM: {self.metadata['llm']}")
        if slm:
            print(f"      SLM: {self.metadata['slm']}")
    
    async def run(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Run agent on input.
        
        Args:
            user_input: User input
            context: Optional context
            
        Returns:
            Result dict
        """
        self.call_count += 1
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Add context if provided
            if context:
                messages.insert(1, {
                    "role": "system",
                    "content": f"Context: {context}"
                })
            
            # Get LLM response
            response = await self.llm.chat(messages) if hasattr(self.llm, 'chat_async') else self.llm.chat(messages)
            # Actually, most of our providers are sync for now except streaming. 
            # But making it async ready.
            
            # Check if tools needed
            tool_calls = self._extract_tool_calls(response)
            
            if tool_calls:
                # Execute tools
                tool_results = []
                for tool_name, tool_args in tool_calls:
                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(tool_args)
                        tool_results.append(result)
                
                # Get final response with tool results
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool results: {tool_results}"
                })
                response = self.llm.chat(messages)
            
            self.success_count += 1
            
            return {
                "success": True,
                "response": response,
                "agent": self.name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    def _extract_tool_calls(self, response: str) -> List:
        """Extract tool calls from response (simple parsing)."""
        # Simple heuristic: look for function call patterns
        tool_calls = []
        # TODO: More sophisticated parsing
        return tool_calls
    
    def get_metrics(self) -> Dict:
        """Get agent metrics."""
        return {
            "name": self.name,
            "calls": self.call_count,
            "success": self.success_count,
            "success_rate": (self.success_count / self.call_count * 100)
            if self.call_count > 0 else 0
        }

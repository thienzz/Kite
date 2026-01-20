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
            system_prompt = self.system_prompt
            if self.tools:
                tool_info = "\n\nCRITICAL: You MUST use the following tools to fulfill the user request. State clearly which tool you are using and with what parameters. Do NOT apologize for not having data if a tool is available.\nAvailable Tools:\n"
                for tool in self.tools.values():
                    tool_info += f"- {tool.name}: {tool.description}\n"
                system_prompt += tool_info
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Add context if provided
            if context:
                messages.insert(1, {
                    "role": "system",
                    "content": f"Context: {context}"
                })
            
            # Get LLM response (Brain)
            if hasattr(self.llm, 'chat_async'):
                response = await self.llm.chat_async(messages)
            else:
                import asyncio
                response = await asyncio.to_thread(self.llm.chat, messages)
            
            print(f"      [DEBUG] [{self.name}] LLM Brain response: {response.strip()[:100]}...")
            
            # Use SLM as "Hands" for low-level extraction (if available)
            # This is 100x cheaper than using LLM for simple parsing
            tool_calls = await self._extract_tool_calls(response)
            
            if tool_calls:
                # Execute tools
                tool_results = []
                import asyncio
                for tool_name, tool_args in tool_calls:
                    if tool_name in self.tools:
                        # Tools might be sync and contain time.sleep or heavy computation
                        # Execute in thread to avoid blocking the event loop
                        result = await asyncio.to_thread(self.tools[tool_name].execute, **tool_args if isinstance(tool_args, dict) else tool_args)
                        tool_results.append(result)
                
                # Get final response with tool results (Brain synthesizes findings)
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool results: {tool_results}"
                })
                
                if hasattr(self.llm, 'chat_async'):
                    response = await self.llm.chat_async(messages)
                else:
                    import asyncio
                    response = await asyncio.to_thread(self.llm.chat, messages)
            
            self.success_count += 1
            
            return {
                "success": True,
                "response": response,
                "agent": self.name,
                "hybrid": self.slm is not None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    async def _extract_tool_calls(self, response: str) -> List:
        """
        Extract tool calls from response using SLM as 'Hands' if available.
        Provides detailed tool definitions for better extraction.
        """
        if not self.slm or not self.tools:
            return []
            
        import json
        import asyncio
        
        # Build detailed tool definitions
        tool_defs = []
        for tool in self.tools.values():
            tool_defs.append(tool.get_definition())
            
        prompt = f"""Extract tool calls from this text.
Available tools with parameters:
{json.dumps(tool_defs, indent=2)}

Text: {response}

Output ONLY a JSON list of calls like: [{{"name": "tool_name", "args": {{"arg": "val"}}}}]]
If no tools needed, return []."""

        try:
            # Use SLM provider's complete method
            if hasattr(self.slm.provider, 'complete_async'):
                raw_extraction = await self.slm.provider.complete_async(prompt)
            else:
                raw_extraction = await asyncio.to_thread(self.slm.provider.complete, prompt)
            
            print(f"      [DEBUG] Raw SLM extraction: {raw_extraction.strip()[:100]}...")
            
            # Clean JSON
            json_text = raw_extraction.strip()
            
            # Extract JSON block if surrounded by text (common with Ollama)
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            elif "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            # Final trim
            json_text = json_text.strip()
            if not json_text.startswith('['):
                # Try simple recovery if it's just one object
                if json_text.startswith('{'):
                    json_text = f"[{json_text}]"
                else:
                    # Fuzzy recovery: look for tool names in the raw text
                    valid_calls = []
                    for name in self.tools:
                        if name in raw_extraction:
                            # If name found, try to extract something that looks like an ID/item
                            # Very simple heuristic for common ecommerce cases
                            import re
                            # Look for something in quotes or following a colon
                            match = re.search(r'["\']?(\w+-\d+|laptop|phone|tablet|stock)["\']?', raw_extraction, re.IGNORECASE)
                            if match:
                                # We found a name and a potential value, but we don't know the key.
                                # Try to match the key from tool definition
                                tool_def = self.tools[name].get_definition()
                                first_param = list(tool_def['parameters'].keys())[0] if tool_def['parameters'] else 'id'
                                valid_calls.append((name, {first_param: match.group(1)}))
                    return valid_calls

            calls = json.loads(json_text)
            valid_calls = []
            for c in calls:
                if not isinstance(c, dict): continue
                name = c.get('name')
                if name in self.tools:
                    tool = self.tools[name]
                    tool_def = tool.get_definition()
                    allowed_params = tool_def.get('parameters', {}).keys()
                    
                    # 1. Try common SLM argument keys
                    args = {}
                    for key in ['args', 'arguments', 'parameters', 'params', 'input']:
                        if key in c and isinstance(c[key], dict):
                            args = c[key]
                            break
                    
                    # 2. Fallback: If still empty, check top-level keys
                    if not args:
                        args = {k: v for k, v in c.items() if k not in ['name', 'args', 'arguments', 'parameters', 'params', 'input']}
                    
                    # 3. CRITICAL: Sanitize arguments (remove keys NOT in tool signature)
                    sanitized_args = {k: v for k, v in args.items() if k in allowed_params}
                    
                    # 4. If sanitization removed everything but prompt was valid, try positional fallback
                    if args and not sanitized_args and len(allowed_params) == 1:
                        first_param = list(allowed_params)[0]
                        
                        # Priority 1: Pick a key that contains the parameter name
                        best_val = None
                        for k, v in args.items():
                            if first_param.lower() in k.lower():
                                best_val = v
                                break
                        
                        # Priority 2: Pick the first non-empty value
                        if best_val is None:
                            best_val = next((v for v in args.values() if v), None)
                            
                        if best_val:
                            sanitized_args = {first_param: best_val}
                    
                    valid_calls.append((name, sanitized_args))
            
            if valid_calls:
                 print(f"      [DEBUG] [{self.name}] Final tool calls: {valid_calls}")
                 
            return valid_calls
        except Exception as e:
            # Fallback to simple heuristic if SLM fails or returns garbage
            print(f"      [DEBUG] [{self.name}] Tool extraction failed: {e}")
            print(f"      [DEBUG] [{self.name}] Raw response extract attempt: {raw_extraction.strip()[:100] if 'raw_extraction' in locals() else 'None'}...")
            return []
    
    def get_metrics(self) -> Dict:
        """Get agent metrics."""
        return {
            "name": self.name,
            "calls": self.call_count,
            "success": self.success_count,
            "success_rate": (self.success_count / self.call_count * 100)
            if self.call_count > 0 else 0
        }

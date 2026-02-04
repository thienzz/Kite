import json
import asyncio
import re
import time
from typing import List, Dict, Optional, Callable


class Agent:
    """
    High-reliability agent with native tool calling and direct extraction fallback.
    """
    
    def __init__(self, 
                 name: str,
                 system_prompt: str,
                 tools: List,
                 framework,
                 llm=None,
                 max_iterations: int = 10,
                 knowledge_sources: List[str] = None,
                 verbose: bool = False):
        self.name = name
        self.system_prompt = system_prompt
        # Logic: Explicit LLM > Framework LLM > Error
        self.llm = llm or getattr(framework, 'llm', None)
        if not self.llm:
            raise ValueError("Agent requires an LLM. Pass 'llm' explicitly or provide a 'framework' with an initialized LLM.")
        self.tools = {tool.name: tool for tool in tools}
        self.framework = framework
        self.max_iterations = max_iterations
        self.knowledge_sources = knowledge_sources or []
        self.verbose = verbose
        
        # Stats
        self.call_count = 0
        self.success_count = 0
        self.metadata = {
            "llm": getattr(self.llm, 'name', getattr(self.llm, 'model', 'unknown'))
        }
        
        # Log creation
        if self.verbose:
            print(f"     Agent '{self.name}' initialized (LLM: {self.metadata['llm']})")

    def record_outcome(self, outcome_type: str):
        """Record a domain-specific outcome (e.g., 'lead', 'reject')."""
        self.framework.metrics.record_outcome(self.name, outcome_type)
    
    async def run(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Run agent on input using Native Tool Calling (if supported) or ReAct fallback.
        """
        self.call_count += 1
        self.framework.event_bus.emit("agent:run:start", {"agent": self.name, "input": user_input})
        
        start_time = time.time()
        success = True
        error_type = None
        
        try:
            native_tools = [t.to_schema() for t in self.tools.values()] if self.tools else []
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_input}]
            
            # Context Injection
            if context:
                messages[0]["content"] += f"\n\nContext: {context}"
                
            # Knowledge Retrieval
            knowledge_context = ""
            if self.knowledge_sources and hasattr(self.framework, 'knowledge'):
                for source_name in self.knowledge_sources:
                    data = self.framework.knowledge.data.get(source_name)
                    if data:
                        matched = []
                        categories = []
                        for key, val in data.items():
                            categories.append(key)
                            if key.lower() in user_input.lower():
                                matched.append(f"- [{source_name}] {key}: {val}")
                        
                        if matched:
                            knowledge_context += f"\n### KNOWLEDGE CONTEXT ({source_name}):\n"
                            knowledge_context += "\n".join(matched) + "\n"
                        else:
                            knowledge_context += f"\n### AVAILABLE KNOWLEDGE CATEGORIES in {source_name}:\n"
                            knowledge_context += f"You have expert expertise in: {', '.join(categories)}.\n"

                    # Vector Memory
                    if hasattr(self.framework, 'vector_memory'):
                        try:
                            mem_results = self.framework.vector_memory.search(user_input, k=3)
                            for _, text, dist in mem_results:
                                if dist < 0.5:
                                        knowledge_context += f"- [Memory:{source_name}] {text[:200]}...\n"
                        except: pass
            
            if knowledge_context:
                messages[0]["content"] += f"\n\n{knowledge_context}"
                self.framework.event_bus.emit("knowledge:retrieved", {
                    "agent": self.name,
                    "context": knowledge_context,
                    "message": "Domain expertise injected from Knowledge Base"
                })

            # Append Tool Info for Legacy mode anyway (some models need it even with native tools, or as backup)
            if self.tools:
                tool_info = f"\n\nAVAILABLE TOOLS:\n"
                for tool in self.tools.values():
                    tool_info += f"- {tool.name}: {tool.description}\n"
                # Add ReAct instructions slightly modified to stay compatible
                tool_info += "\nNOTE: You can use tools natively if supported, OR output 'Action: [{...}]' JSON."
                messages[0]["content"] += tool_info

            all_data = {}
            last_valid_response = ""

            for i in range(self.max_iterations):
                if i > 0:
                    print(f"      [DEBUG] [{self.name}] Iteration {i+1}/{self.max_iterations}...")

                # Call LLM
                response_data = None
                try:
                    # Optimized Fallback Logic: Don't retry native tools if they already failed once
                    should_try_native = (
                        native_tools 
                        and hasattr(self.llm, 'chat_async') 
                        and not getattr(self, '_native_tools_failed', False)
                    )
                    
                    if should_try_native:
                        # Try passing tools
                        try:
                            response_data = await self.llm.chat_async(messages, tools=native_tools)
                        except Exception as e:
                            # Fallback if provider/model doesn't support tools arg (TypeError) or API rejects it (HTTP 400)
                            print(f"      [DEBUG] Native tool call failed ({e}), permanently falling back to text for this session...")
                            self._native_tools_failed = True
                            response_data = await self.llm.chat_async(messages)
                    elif hasattr(self.llm, 'chat_async'):
                        response_data = await self.llm.chat_async(messages)
                    elif hasattr(self.llm, 'complete_async'):
                        # Fallback for completion only models
                        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
                        response_data = await self.llm.complete_async(prompt)
                    else:
                        # Sync fallback
                        if hasattr(self.llm, 'chat'):
                            response_data = await asyncio.to_thread(self.llm.chat, messages)
                        else:
                            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
                            response_data = await asyncio.to_thread(self.llm.complete, prompt)
                except Exception as e:
                    print(f"      [LLM ERROR] {e}")
                    # Retry once?
                    time.sleep(1)
                    continue

                # Handle Response
                tool_calls = []
                content = ""
                
                if isinstance(response_data, dict):
                    # Native Tool Call
                    content = response_data.get("content", "")
                    tool_calls_raw = response_data.get("tool_calls", [])
                    
                    # Convert raw API tool calls to internal format
                    for tc in tool_calls_raw:
                        if isinstance(tc, dict):
                            # OpenAI/Ollama format: function: {name, arguments}
                            func = tc.get("function", {})
                            name = func.get("name")
                            args = func.get("arguments")
                            tool_id = tc.get("id")
                            if isinstance(args, str):
                                try: args = json.loads(args)
                                except: pass
                            if name:
                                tool_calls.append((name, args, tool_id))
                                
                else:
                    # Text Response (Legacy ReAct check)
                    content = str(response_data)
                    last_valid_response = content
                    tool_calls_raw = await self._extract_tool_calls(content, context)
                    # Normalize legacy extraction to include None ID
                    tool_calls = [(name, args, None) for name, args in tool_calls_raw]
                
                # Update history
                if content:
                    messages.append({"role": "assistant", "content": content})
                    last_valid_response = content
                
                if tool_calls and not content:
                     # If no content but tools, add a placeholder assistant msg (required by some APIs)
                     # For Ollama we might need to be careful.
                     # We skip adding an empty content message if we handled it via the dict above?
                     # Actually to keep the chat history valid, we should add the assistant's turn that invoked the tool.
                     messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls_raw} if isinstance(response_data, dict) else {"role": "assistant", "content": "Executing tools..."})

                # EMIT THOUGHT
                if content:
                    self.framework.event_bus.emit("agent:thought", {
                        "agent": self.name,
                        "thought": content,
                        "iteration": i + 1,
                        "task_id": context.get('task_id') if isinstance(context, dict) else None
                    })

                # Check Termination
                if not tool_calls: 
                    if "Final Answer:" in content:
                        clean_answer = content.split("Final Answer:")[-1].strip()
                        self.success_count += 1
                        return {
                            "success": True,
                            "response": clean_answer,
                            "full_log": content,
                            "agent": self.name,
                            "data": all_data,
                            "iterations": i + 1
                        }
                    
                    # In native mode, if we get text and no tools, and it's NOT a clarifying question (heuristic), we might be done.
                    # But for safety, we rely on "Final Answer" or implicit end.
                    # If we simply return text, we might be done.
                    # If we simply return text, we might be done.
                    if content and (not native_tools or not "Action:" in content):
                         # Assume done if valid content and no specific Action request
                         self.success_count += 1
                         return {
                            "success": True,
                            "response": content,
                            "full_log": content,
                            "agent": self.name,
                            "data": all_data,
                            "iterations": i + 1
                        }

                # Execute Tools
                for item in tool_calls:
                    if len(item) == 3:
                        name, args, tool_id = item
                    else:
                        name, args = item
                        tool_id = None

                    if name in self.tools:
                        if self.verbose:
                            print(f"      | [ACTION] {name}({args})")
                        try:
                            cleaned_args = {str(k): v for k, v in args.items()} if isinstance(args, dict) else {}
                            
                            self.framework.event_bus.emit("agent:tool_call", {"agent": self.name, "tool": name, "args": args})
                            result = await self.tools[name].execute(**cleaned_args, framework=self.framework)
                            self.framework.event_bus.emit("agent:tool_result", {"agent": self.name, "tool": name, "result": result, "success": True})
                            
                            all_data[name] = result
                            
                            # Append Result
                            if tool_id:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": name,
                                    "content": str(result)
                                })
                            else:
                                # Fallback for legacy extraction: Use 'user' role
                                messages.append({
                                    "role": "user",
                                    "content": f"Tool '{name}' Output: {result}"
                                })
                            
                        except Exception as e:
                            if self.verbose:
                                print(f"      [{self.name} FAILED] {name}: {e}")
                            self.framework.event_bus.emit("agent:tool_result", {"agent": self.name, "tool": name, "error": str(e), "success": False})
                            
                            if tool_id:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": name,
                                    "content": f"Error: {e}"
                                })
                            else:
                                messages.append({
                                    "role": "user",
                                    "content": f"Tool '{name}' Failed: {e}"
                                })

            # Max iterations reached
            return {
                "success": False,
                "response": "Max iterations reached without Final Answer.",
                "full_log": last_valid_response,
                "agent": self.name,
                "data": all_data,
                "iterations": self.max_iterations
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            success = False
            error_type = type(e).__name__
            return {"success": False, "error": str(e), "response": f"Error: {str(e)}", "agent": self.name}
        finally:
            duration = time.time() - start_time
            self.framework.metrics.record_request(self.name, "run", duration, success, error_type)

    def run_sync(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Synchronous wrapper for run.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.run(user_input, context))
        else:
            return asyncio.run(self.run(user_input, context))

    async def _extract_tool_calls(self, response: str, context: Optional[Dict] = None) -> List:
        """Robust tool extraction combining direct regex and LLM fallback."""
        if not self.tools: return []
        
        # 1. OPTIMIZATION: Direct Regex Extraction from response text
        # This is 100% reliable if the model follows the Action format.
        # We look for a JSON block after the word "Action:"
        action_match = re.search(r"Action:\s*(\[.*?\])", response, re.DOTALL)
        if action_match:
            try:
                raw_json = action_match.group(1).strip()
                # Pre-processing to fix common LLM JSON errors (like missing quotes on keys or trailing commas)
                raw_json = re.sub(r'\}\s*\{', '}, {', raw_json)
                calls = json.loads(raw_json)
                if isinstance(calls, list):
                    valid = self._sanitize_calls(calls)
                    if valid:
                        print(f"      [DEBUG] [{self.name}] Direct regex extraction successful: {len(valid)} calls")
                        return valid
            except Exception as e:
                print(f"      [DEBUG] [{self.name}] Direct regex parsing failed: {e}. Falling back to LLM...")

        # 2. FALLBACK: Use LLM for extraction
        tool_defs = [tool.get_definition() for tool in self.tools.values()]
        prompt = f"""Extract tool calls from this text.
Available tools: {json.dumps(tool_defs, indent=2)}
Text: {response}

### Instructions:
1. ONLY extract NEW action calls.
2. Output a valid JSON list: [{{"name": "...", "args": {{...}}}}]
3. Return [] if no new action is needed.
"""
        try:
            extractor = self.llm
            if hasattr(extractor, 'complete_async'):
                raw = await extractor.complete_async(prompt)
            else:
                raw = await asyncio.to_thread(extractor.complete, prompt)
            
            blocks = re.findall(r'(\[[\s\S]*?\]|\{[\s\S]*?\})', raw.strip())
            all_calls = []
            for block in blocks:
                try:
                    parsed = json.loads(re.sub(r'\}\s*\{', '}, {', block))
                    if isinstance(parsed, list): all_calls.extend(parsed)
                    elif isinstance(parsed, dict) and 'name' in parsed: all_calls.append(parsed)
                except: pass
            
            valid = self._sanitize_calls(all_calls)
            return valid
        except: return []

    def _sanitize_calls(self, calls: List) -> List:
        """Sanitize and validate raw tool call objects."""
        valid = []
        for c in calls:
            if not isinstance(c, dict): continue
            name = c.get('name')
            if name in self.tools:
                allowed = self.tools[name].get_definition().get('parameters', {}).keys()
                
                # Resolve args
                args = {}
                for k in ['args', 'arguments', 'parameters', 'params', 'input']:
                    if k in c and isinstance(c[k], dict): 
                        args = c[k]
                        break
                if not args: args = {k: v for k, v in c.items() if k not in ['name', 'args', 'arguments']}
                
                # Resilient mapping
                sanitized = {k: v for k, v in args.items() if k in allowed}
                if not sanitized and args and len(allowed) == 1:
                    target_key = list(allowed)[0]
                    # Map the most likely candidate from the provided dict
                    best_val = next((v for v in args.values() if isinstance(v, (str, int, float))), None)
                    if best_val:
                        sanitized = {target_key: best_val}
                
                # Placeholder filter
                if not any(isinstance(v, str) and (v.startswith('<') or '[' in v or v == 'id') for v in sanitized.values()):
                    valid.append((name, sanitized))
        return valid
    
    def get_metrics(self) -> Dict:
        return {
            "name": self.name,
            "calls": self.call_count,
            "success": self.success_count,
            "success_rate": (self.success_count / self.call_count * 100) if self.call_count > 0 else 0
        }

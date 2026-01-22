import json
import asyncio
import re
from typing import List, Dict, Optional, Callable


class Agent:
    """
    High-reliability agent with direct extraction and termination guards.
    """
    
    def __init__(self, 
                 name: str,
                 system_prompt: str,
                 llm,
                 tools: List,
                 framework,
                 max_iterations: int = 10):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.framework = framework
        self.max_iterations = max_iterations
        
        # Stats
        self.call_count = 0
        self.success_count = 0
        self.metadata = {
            "llm": getattr(llm, 'name', getattr(llm, 'model', 'unknown'))
        }
        
        # Log creation
        print(f"     Agent '{self.name}' initialized (LLM: {self.metadata['llm']})")
    
    async def run(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Run agent on input.
        """
        self.call_count += 1
        
        try:
            session_tool_results = {} # norm -> {"data": ..., "success": bool}
            successful_norms = set()
            failed_norms = {} # norm -> error_msg
            messages = []
            
            def normalize_call(name: str, args: Dict) -> str:
                if not isinstance(args, dict):
                    return f"{name}:{str(args).strip()}"
                clean = {str(k).lower(): str(v).strip() for k, v in args.items()}
                sorted_str = json.dumps(clean, sort_keys=True)
                return f"{name}:{sorted_str}"
            
            last_valid_response = ""

            for i in range(self.max_iterations):
                if i > 0:
                    print(f"      [DEBUG] [{self.name}] Iteration {i+1}/{self.max_iterations}...")

                # 1. Build Dynamic Memory
                memory_text = ""
                if session_tool_results:
                    memory_text = "\n### CURRENT MEMORY (Verified Facts):\n"
                    for norm, val in session_tool_results.items():
                        if val['success']:
                            memory_text += f"- {norm} -> {val['data']}\n"
                
                # 2. Build Prompt
                full_system_prompt = self.system_prompt
                if context:
                    full_system_prompt += f"\n\nContext: {context}"
                
                if self.tools:
                    tool_info = f"\n\nAVAILABLE TOOLS:\n"
                    for tool in self.tools.values():
                        tool_info += f"- {tool.name}: {tool.description}\n"
                    
                    tool_info += f"""
{memory_text}

### CRITICAL OPERATIONAL RULES:
1. **Mental Lock**: If a fact is in CURRENT MEMORY, you MUST NOT ask for it again. PROPOSING A REDUNDANT TOOL IS A FAILURE.
2. **True Termination**: You MUST provide a "Final Answer: [result]" and STOP ONLY when the user's request is 100% fulfilled.
3. **Evidence-Based Checklist**: When marking a task as [Done], you MUST include the SPECIFIC data retrieved. 
   - Good: [Done] Found order ORD-001: Status is Shipped, delivery tomorrow.
4. **Action Format**: Action: [{{"name": "...", "args": {{...}}}}] (Only if NEW data is needed)
5. **No Placeholders**: Never use "Final Answer: None" or "Final Answer: Not yet". If you aren't done, ONLY provide an Action or Thought.

### Reasoning Format:
Thought: 
  Goal: [objective]
  Checklist:
  - [status] task 1 (include specific data if Done)
  Reasoning: [Analysis of Memory vs Goal]
Action: [{{"name": "...", "args": {{...}}}}] (OMIT IF GOAL IS REACHED)
Final Answer: [The final response to the user]
"""
                    full_system_prompt += tool_info

                system_msg = {"role": "system", "content": full_system_prompt}
                if i == 0:
                    messages = [system_msg, {"role": "user", "content": user_input}]
                else:
                    messages[0] = system_msg 
                
                # Get response
                if hasattr(self.llm, 'chat_async'):
                    response = await self.llm.chat_async(messages)
                elif hasattr(self.llm, 'complete_async'):
                    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
                    response = await self.llm.complete_async(prompt)
                else:
                    if hasattr(self.llm, 'chat'):
                        response = await asyncio.to_thread(self.llm.chat, messages)
                    else:
                        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"
                        response = await asyncio.to_thread(self.llm.complete, prompt)
                
                # IMMEDIATE State Commitment
                last_valid_response = response.strip()
                
                # High-Visibility Reasoning Box
                border = "=" * 60
                print(f"\n      {border}")
                print(f"      [ {self.name} REASONING ]")
                print(f"      {border}")
                for line in last_valid_response.split('\n'):
                    print(f"      | {line.strip()}")
                print(f"      {border}\n")
                
                # 1. Extract tool calls FIRST to prevent premature termination
                tool_calls = await self._extract_tool_calls(last_valid_response, context=context)
                
                # 2. Robust Termination Check
                # Only break if "Final Answer:" exists AND hasn't been blocked by the presence of actions
                final_answer_match = re.search(r"Final Answer:\s*([\s\S]+)", last_valid_response, re.IGNORECASE)
                has_valid_answer = False
                if final_answer_match:
                    answer_text = final_answer_match.group(1).strip().lower()
                    placeholders = ["none", "not yet", "waiting", "i'll wait", "not yet retrieved", "waiting for response"]
                    if answer_text and not any(p in answer_text for p in placeholders):
                         has_valid_answer = True
                
                # We ONLY break if we have a valid answer AND no pending tools
                if has_valid_answer and not tool_calls:
                    break
                
                if not tool_calls:
                    print(f"      [{self.name}] No Action and no valid Final Answer. Nudging...")
                    messages.append({"role": "assistant", "content": last_valid_response})
                    messages.append({
                        "role": "user", 
                        "content": "ERROR: You provided neither an 'Action:' nor a 'Final Answer:'. If you are done, provide your final response after 'Final Answer:'. If not, provide a NEW 'Action:'."
                    })
                    continue

                # Filter and Executerd
                filtered_tool_calls = []
                skipped_observations = []
                for name, args in tool_calls:
                    norm = normalize_call(name, args)
                    if norm in successful_norms:
                        print(f"      [{self.name}] Skipping redundant success: {name}({args})")
                        skipped_observations.append(f"- Tool '{name}' previously returned: {session_tool_results[norm]['data']}")
                    elif norm in failed_norms:
                        print(f"      [{self.name}] Blocking known failure: {name}({args})")
                        skipped_observations.append(f"- Tool '{name}' previously FAILED: {failed_norms[norm]}. Fix parameters.")
                    else:
                        filtered_tool_calls.append((name, args))
                
                if not filtered_tool_calls:
                    print(f"      [{self.name}] All proposed actions are redundant. Forcing Final Answer.")
                    messages.append({"role": "assistant", "content": last_valid_response})
                    messages.append({
                        "role": "user",
                        "content": "CRITICAL: You are repeating tool calls already in your MEMORY. PROVIDE YOUR 'Final Answer:' NOW."
                    })
                    continue

                # Execute
                tool_results = []
                for tool_name, tool_args in filtered_tool_calls:
                    if tool_name in self.tools:
                        norm = normalize_call(tool_name, tool_args)
                        try:
                            cleaned_args = {str(k): v for k, v in tool_args.items()} if isinstance(tool_args, dict) else {}
                            print(f"      | [ACTION] {tool_name}({tool_args})")
                            result = await asyncio.to_thread(self.tools[tool_name].execute, **cleaned_args)
                            
                            tool_results.append({"tool": tool_name, "result": result, "success": True})
                            session_tool_results[norm] = {"data": result, "success": True}
                            successful_norms.add(norm)
                            if norm in failed_norms: del failed_norms[norm]
                        except Exception as tool_err:
                            err_msg = str(tool_err)
                            print(f"      [{self.name} FAILED] {tool_name}: {err_msg}")
                            tool_results.append({"tool": tool_name, "error": err_msg, "success": False})
                            session_tool_results[norm] = {"data": err_msg, "success": False}
                            failed_norms[norm] = err_msg
                
                messages.append({"role": "assistant", "content": last_valid_response})
                observation_text = "Observations:\n"
                for obs in skipped_observations:
                    observation_text += obs + "\n"
                for res in tool_results:
                    if res.get('success'):
                        observation_text += f"- Tool '{res['tool']}' returned: {res['result']}\n"
                    else:
                        observation_text += f"- Tool '{res['tool']}' FAILED: {res.get('error')}\n"
                messages.append({"role": "user", "content": observation_text})
            
            # CLEAN FINAL ANSWER EXTRACTION
            clean_answer = last_valid_response
            final_answer_match = re.search(r"Final Answer:\s*([\s\S]+)", last_valid_response, re.IGNORECASE)
            if final_answer_match:
                clean_answer = final_answer_match.group(1).strip()
            
            self.success_count += 1
            return {
                "success": True,
                "response": clean_answer, # Return ONLY the clean answer to the caller
                "full_log": last_valid_response, # Keep logs for internal audit
                "agent": self.name,
                "iterations": i + 1
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "agent": self.name}
    
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

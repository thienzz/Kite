"""
ReActAgent - Autonomous agent using the Think-Act-Observe pattern.
"""

import time
import json
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from ..agent import Agent
from ..safety.kill_switch import KillSwitch


class ReActAgent(Agent):
    """
    Autonomous agent that implements the ReAct (Reason + Act) loop.
    It thinks, acts using tools, and observes results until a goal is achieved
    or a safety limit is triggered.
    """
    
    def __init__(self, 
                 name: str,
                 system_prompt: str,
                 llm,
                 tools: List,
                 framework,
                 max_iterations: int = 15,
                 kill_switch: Optional[KillSwitch] = None,
                 knowledge_sources: List[str] = None):
        super().__init__(name, system_prompt, llm, tools, framework, max_iterations=max_iterations, knowledge_sources=knowledge_sources)
        self.kill_switch = kill_switch or KillSwitch(max_time=600, max_iterations=15)
        
    async def run(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """Override base run to use autonomous logic."""
        return await self.run_autonomous(user_input, context)

    async def run_autonomous(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the agent autonomously to achieve a goal.
        """
        state = {
            'goal': goal,
            'steps': 0,
            'history': [], # List of {thought, action, observation}
            'confirmed_facts': [],
            'missing_info': [goal],
            'total_cost': 0.0,
            'start_time': time.time(),
            'completed': False,
            'final_answer': None,
            'context': context or {},
            'data': {} # Map of tool_name -> result
        }
        
        print(f"\n[{self.name}] Starting advanced autonomous loop for: {goal}")
        
        while True:
            # 1. Check Kill Switch
            should_stop, reason = self.kill_switch.check(state)
            if should_stop:
                print(f"[{self.name}] Loop terminated: {reason}")
                break
                
            state['steps'] += 1
            if state['steps'] > self.max_iterations:
                self.logger.warning(f"[{self.name}] Max iterations reached.")
                self.framework.event_bus.emit("agent:error", {"agent": self.name, "error": "Max iterations reached"})
                break

            # 1b. Check for repetitive failures
            if len(state['history']) >= 3:
                last_three = state['history'][-3:]
                actions = [h['action'].get('tool') for h in last_three]
                status = [h['action'].get('status') for h in last_three]
                if len(set(actions)) == 1 and all(s != 'success' for s in status):
                    print(f"   [{self.name}] Detected loop on failed action '{actions[0]}'. Terminating.")
                    state['completed'] = False
                    state['final_answer'] = f"Loop detected on failed action: {actions[0]}"
                    break

            self.framework.event_bus.emit("agent:step", {
                "agent": self.name, 
                "step": state['steps'],
                "goal": state['goal'],
                "summary": state['history'][-1]['reasoning'] if state['history'] else "Starting task..."
            })
            
            # 2. THINK & PLAN (Structured Extraction)
            print(f"[{self.name}] Reflecting and deciding next action...")
            prompt = self._build_structured_prompt(state)
            
            try:
                response = await self._get_llm_response(prompt)
                structured_output = self._parse_structured_output(response)
                
                # Support common aliases for keys
                reasoning = structured_output.get('reasoning') or structured_output.get('thought') or structured_output.get('reflection', 'No reasoning.')
                action_name = structured_output.get('tool') or structured_output.get('action') or structured_output.get('name')
                action_args = structured_output.get('arguments') or structured_output.get('args') or structured_output.get('parameters', {})
                is_final = structured_output.get('is_final') or structured_output.get('done', False)
                confidence = structured_output.get('confidence', 1.0)
                
                self.framework.event_bus.emit("agent:thought", {
                    "agent": self.name, 
                    "reasoning": reasoning,
                    "confidence": confidence
                })
                print(f"   [{self.name}] Thinking: {reasoning}")
                
                if is_final:
                    state['completed'] = True
                    state['final_answer'] = structured_output.get('answer', 'Goal achieved.')
                    self.framework.event_bus.emit("agent:complete", {
                        "agent": self.name, 
                        "answer": state['final_answer']
                    })
                    break
                
                # 3. ACT
                if not action_name or action_name not in self.tools:
                    if is_final:
                        observation = "Goal marked as complete."
                        action_record = {'tool': 'None', 'args': {}, 'status': 'success'}
                    else:
                        observation = f"Error: Tool '{action_name}' not found. Please choose from: {list(self.tools.keys())} or signal is_final: true."
                        action_record = {'tool': action_name, 'args': action_args, 'status': 'failed'}
                else:
                    self.framework.event_bus.emit("agent:action", {
                        "agent": self.name,
                        "tool": action_name,
                        "args": action_args
                    })
                    print(f"   [{self.name}] Action: {action_name}({action_args})")
                    try:
                        # Correct async-safe execution
                        result = await self.tools[action_name].execute(**action_args, framework=self.framework)
                        
                        observation = self._format_observation(result)
                        action_record = {'tool': action_name, 'args': action_args, 'status': 'success'}
                    except Exception as e:
                        observation = f"Error executing {action_name}: {str(e)}"
                        action_record = {'tool': action_name, 'args': action_args, 'status': 'error'}
                
                if not observation or observation == "[]":
                    observation = "[] (WARNING: No results found for this query. Do NOT imagine data. Try a broader search or a different tool.)"
                
                # Handling for Drifting / Missing Tools
                if not action_name and not is_final:
                    available_tools = list(self.tools.keys())
                    observation = f"Error: You provided reasoning but NO tool names. Choose from: {available_tools}, or signal is_final: true."
                    action_record = {'tool': 'None', 'args': {}, 'status': 'failed'}
                
                # Console Visibility for Observations
                obs_preview = observation[:200].replace('\n', ' ') + "..." if len(observation) > 200 else observation
                print(f"   [{self.name}] Observation: {obs_preview}")
                
                # 4. RECORD
                state['history'].append({
                    'step': state['steps'],
                    'reasoning': reasoning,
                    'action': action_record,
                    'observation': observation,
                    'confidence': confidence
                })
                # EMIT FULL OBSERVATION FOR DASHBOARD
                self.framework.event_bus.emit("agent:observation", {
                    "agent": self.name,
                    "observation": observation,
                    "full_data": result if action_record['status'] == 'success' else None
                })
                
                # Dynamic Fact Extraction
                if action_record['status'] == 'success':
                    state['confirmed_facts'].append(f"Result of {action_name}: {observation}")
                    state['data'][action_name] = result
                
            except Exception as e:
                import traceback
                print(f"[{self.name}] Error in loop: {str(e)}")
                # traceback.print_exc() # Keep it quiet but logged for now
                observation = f"Internal Error: {str(e)}"
                state['history'].append({
                    'step': state['steps'], 
                    'reasoning': "Self-correction: An internal error occurred.",
                    'action': {'tool': 'recovery', 'args': {}, 'status': 'error'},
                    'observation': observation
                })
                if state['steps'] >= self.max_iterations: break

        return {
            "success": state['completed'],
            "response": state.get('final_answer') or "No final answer reached. This usually happens if tool results were empty or the mission was impossible.",
            "goal": goal,
            "steps": state['steps'],
            "history": state['history'],
            "agent": self.name,
            "data": state['data']
        }
    
    def _build_structured_prompt(self, state: Dict) -> str:
        tool_desc = ""
        for n, t in self.tools.items():
            tool_desc += f"- {n}: {t.description}\n"
        
        # Build advanced reflection context (more concise)
        history_text = ""
        if state['history']:
            history_text = "\n### Execution History (Steps Taken)\n"
            for h in state['history']:
                # Truncate observation for prompt efficiency
                obs_summary = h['observation'][:500] + "..." if len(h['observation']) > 500 else h['observation']
                history_text += f"\n[Step {h['step']}] Action: {h['action'].get('tool')} -> Output Summary: {obs_summary}\n"

        return f"""
# PERSONA & CONTEXT
{self.system_prompt}

## 🎯 THE MISSION (UNALTERABLE)
Goal: {state['goal']}

## 🧠 CURRENT KNOWLEDGE BASE
{history_text}

### EXPERT SEARCH TEMPLATES (Native Knowledge)
{self._get_native_knowledge(state['goal'])}

## 🚀 OPERATIONAL GUIDELINES:
1. **Target Human Beings**: Your goal is to find NAMES of founders, CTOs, and specific COMPANY NEEDS. 
2. **Beware of Ads**: Do NOT adopt marketing copy as your goal.
3. **Efficiency**: Stop once you have 2 High-Quality leads.
4. **ANTI-HALLUCINATION GUARD**: 
   - NEVER invent names like 'John Doe' or 'Jane Smith' if no leads are found.
   - NEVER invent profile or post links. If not found, report 'N/A'.
   - NEVER use names from the Observation (like people's names or 'Unknown') as tool names.
   - If a tool returns `[]`, YOU MUST acknowledge 'No leads found' instead of inventing them.
   - If results are missing, suggest a better search query instead of finishing with fake data.

Available Tools:
{tool_desc}

### Strict JSON Output
{{
  "reasoning": "Specifically analyze what was found and what is missing.",
  "tool": "tool_name",
  "arguments": {{"param_name": "value"}},
  "is_final": false,
  "answer": "Final report string (only if is_final is true)"
}}
"""

    async def _get_llm_response(self, prompt: str) -> str:
        stop_tokens = ["\nObservation:", "\nThought:", "\nAction:", "\nStep"]
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt, temperature=0.1, stop=stop_tokens)
        import asyncio
        return await asyncio.to_thread(self.llm.complete, prompt, temperature=0.1, stop=stop_tokens)

    def _clean_json(self, s: str) -> str:
        """Handle common LLM JSON formatting errors."""
        import re
        # Remove markdown code blocks
        s = re.sub(r'```json\s*(.*?)\s*```', r'\1', s, flags=re.DOTALL)
        s = re.sub(r'```\s*(.*?)\s*```', r'\1', s, flags=re.DOTALL)
        
        # Balance braces if truncated
        open_braces = s.count('{')
        close_braces = s.count('}')
        if open_braces > close_braces:
            s += '}' * (open_braces - close_braces)
            
        # Fix single quotes to double quotes for keys/values
        # This is risky but helpful for some models
        # s = re.sub(r"'(.*?)'", r'"\1"', s) 
        
        # Remove trailing commas before closing braces/brackets
        s = re.sub(r',\s*\}', '}', s)
        s = re.sub(r',\s*\]', ']', s)
        
        return s.strip()

    def _parse_structured_output(self, response: str) -> Dict:
        try:
            clean_res = self._clean_json(response)
            # Find the FIRST '{' and the LAST '}' to extract potential JSON block
            start_idx = clean_res.find('{')
            if start_idx == -1:
                return {"reasoning": f"No JSON found. Raw: {response[:50]}", "is_final": False}
            
            # Find the matching closing brace for the FIRST object to handle "Extra data"
            depth = 0
            end_idx = -1
            for i, char in enumerate(clean_res[start_idx:], start=start_idx):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            
            if end_idx == -1:
                 # Fallback to rfind if depth counting fails
                 end_idx = clean_res.rfind('}')
            
            json_str = clean_res[start_idx:end_idx+1]
            data = json.loads(json_str)
            if 'reasoning' not in data and 'thought' in data:
                data['reasoning'] = data['thought']
            return data
            
        except Exception as e:
            # Fallback A: Try regex finding the first {...} block
            try:
                import re
                # Pre-clean the response to handle unescaped newlines within JSON strings
                # This is a common issue with smaller LLMs
                fixed_response = re.sub(r'(?<=: ")(.*?)(?=",)', lambda m: m.group(1).replace('\n', ' '), response, flags=re.DOTALL)
                
                match = re.search(r'\{.*\}', fixed_response, re.DOTALL)
                if match:
                    json_str = match.group()
                    # Fix truncated JSON if possible
                    if json_str.count('{') > json_str.count('}'):
                        json_str += '}' * (json_str.count('{') - json_str.count('}'))
                    return json.loads(json_str)
            except: pass
            
            print(f"[{self.name}] JSON Parse Error: {e}")
            return {"reasoning": f"Parse Error: {str(e)}. Try to strictly follow the JSON output format.", "is_final": False}

    def _get_native_knowledge(self, goal: str) -> str:
        """Fetch relevant expert templates from framework knowledge."""
        if not hasattr(self.framework, 'knowledge'):
            return "No knowledge store available."
            
        expert_queries = self.framework.knowledge.get("linkedin_queries")
        if not expert_queries:
            return "No expert templates found."
            
        relevant = []
        for key, val in expert_queries.items():
            # Match goal or name
            if key.lower() in goal.lower() or key.lower() in self.name.lower():
                relevant.append(f"- Expert '{key}' Query: {val}")
        
        return "\n".join(relevant) if relevant else "No specific expert templates matched this mission."

    def _format_observation(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2)
            except:
                return str(result)
        return str(result)

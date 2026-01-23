"""
ReActAgent - Autonomous agent using the Think-Act-Observe pattern.
"""

import time
import json
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
                 max_iterations: int = 10,
                 kill_switch: Optional[KillSwitch] = None):
        super().__init__(name, system_prompt, llm, tools, framework, max_iterations=max_iterations, agent_type="react")
        self.kill_switch = kill_switch or KillSwitch()
        
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
            'context': context or {}
        }
        
        print(f"\n[{self.name}] Starting advanced autonomous loop for: {goal}")
        
        while True:
            # 1. Check Kill Switch
            should_stop, reason = self.kill_switch.check(state)
            if should_stop:
                print(f"[{self.name}] Loop terminated: {reason}")
                break
                
            state['steps'] += 1
            print(f"\n--- Step {state['steps']} ---")
            
            # 2. THINK & PLAN (Structured Extraction)
            print(f"[{self.name}] Reflecting and deciding next action...")
            prompt = self._build_structured_prompt(state)
            
            try:
                response = await self._get_llm_response(prompt)
                structured_output = self._parse_structured_output(response)
                
                # Support both 'reasoning' and 'thought' for resilience
                reasoning = structured_output.get('reasoning') or structured_output.get('thought', 'No reasoning provided.')
                action_name = structured_output.get('tool')
                action_args = structured_output.get('arguments') or structured_output.get('args', {})
                is_final = structured_output.get('is_final', False)
                confidence = structured_output.get('confidence', 1.0)
                
                print(f"Reasoning: {reasoning}")
                if confidence < 0.6:
                    print(f"Warning: Low confidence ({confidence:.2f})")
                
                if is_final:
                    state['completed'] = True
                    state['final_answer'] = structured_output.get('answer', 'Goal achieved.')
                    print(f"[{self.name}] Final Answer: {state['final_answer']}")
                    break
                
                # 3. ACT
                if not action_name or action_name not in self.tools:
                    observation = f"Error: Tool '{action_name}' not found. Please choose from: {list(self.tools.keys())} or signal is_final: true."
                    action_record = {'tool': action_name, 'args': action_args, 'status': 'failed'}
                else:
                    print(f"Action: {action_name}({action_args})")
                    try:
                        result = self.tools[action_name].execute(**action_args)
                        observation = self._format_observation(result)
                        action_record = {'tool': action_name, 'args': action_args, 'status': 'success'}
                    except Exception as e:
                        observation = f"Error executing {action_name}: {str(e)}"
                        action_record = {'tool': action_name, 'args': action_args, 'status': 'error'}
                
                # 4. RECORD
                state['history'].append({
                    'step': state['steps'],
                    'reasoning': reasoning,
                    'action': action_record,
                    'observation': observation,
                    'confidence': confidence
                })
                print(f"Observation: {observation}")
                
                # Dynamic Fact Extraction (Simplified)
                if action_record['status'] == 'success':
                    state['confirmed_facts'].append(f"Result of {action_name}: {observation}")
                    # Remove the successful action's intent from missing_info if relevant
                    # For now, we trust the LLM to see the Facts and re-reflect.
                
            except Exception as e:
                print(f"[{self.name}] Error in loop: {e}")
                observation = f"Internal Error: {str(e)}"
                state['history'].append({
                    'step': state['steps'], 
                    'reasoning': "Self-correction: An internal error occurred.",
                    'action': {'tool': 'recovery', 'args': {}, 'status': 'error'},
                    'observation': observation
                })
                if state['steps'] >= 10: break

        return {
            "success": state['completed'],
            "goal": goal,
            "answer": state.get('final_answer'),
            "steps": state['steps'],
            "history": state['history'],
            "reason": reason if not state['completed'] else "Completed"
        }
    
    def _build_structured_prompt(self, state: Dict) -> str:
        tool_desc = ""
        for n, t in self.tools.items():
            tool_desc += f"- {n}: {t.description}\n"
        
        # Build advanced reflection context
        history_text = ""
        if state['history']:
            history_text = "\n### Execution History\n"
            for h in state['history']:
                history_text += f"\n- Step {h['step']}: Action `{h['action'].get('tool')}` -> Result: {h['observation']}\n"

        facts_text = "\n".join([f"- {f}" for f in state['confirmed_facts']]) if state['confirmed_facts'] else "No facts confirmed yet."

        return f"""Goal: {state['goal']}

### Knowledge & Reflection
- What I know so far:
{facts_text}

{history_text}

### Instructions
You are an autonomous agent using the ReAct (Reasoning + Acting) pattern. 
Your goal is complex. You MUST confirm all parts of it using tools.

1. REVIEW: Look at "What I know so far". Does it fully resolve the Goal?
2. ANTI-LAZINESS: Do NOT signal is_final: true if you still have "No facts confirmed" or if parts of the goal are missing. 
3. DECIDE: If you lack information, you MUST use a tool.

Available Tools:
{tool_desc}

### Strict JSON Response Format
Respond with valid JSON ONLY.
{{
  "reasoning": "Specifically list: 1. Goal part A status. 2. Goal part B status. 3. Target tool.",
  "tool": "tool_name",
  "arguments": {{"arg_name": "value"}},
  "confidence": 0.9,
  "is_final": false,
  "answer": "Complete final answer explaining all facts found."
}}
"""

    async def _get_llm_response(self, prompt: str) -> str:
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt, temperature=0.1)
        import asyncio
        return await asyncio.to_thread(self.llm.complete, prompt, temperature=0.1)

    def _parse_structured_output(self, response: str) -> Dict:
        try:
            clean_res = response.strip()
            start_idx = clean_res.find('{')
            if start_idx == -1:
                return {"reasoning": f"No JSON found. Raw: {response[:50]}", "is_final": False}
            
            end_idx = clean_res.rfind('}')
            if end_idx == -1:
                # Attempt to close truncated JSON
                json_str = clean_res[start_idx:]
                if json_str.count('"') % 2 != 0: json_str += '"'
                json_str += '}'
                try: return json.loads(json_str)
                except: return {"reasoning": "Truncated JSON", "is_final": False}
                
            json_str = clean_res[start_idx:end_idx+1]
            data = json.loads(json_str)
            if 'reasoning' not in data and 'thought' in data:
                data['reasoning'] = data['thought']
            return data
            
        except Exception as e:
            print(f"[{self.name}] JSON Parse Warning: {e}")
            return {"reasoning": f"Parse Error: {str(e)}", "is_final": False}

    def _format_observation(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)

    def _format_observation(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)

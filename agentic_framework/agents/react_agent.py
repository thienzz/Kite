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
                 slm = None,
                 kill_switch: Optional[KillSwitch] = None):
        super().__init__(name, system_prompt, llm, tools, framework, slm)
        self.kill_switch = kill_switch or KillSwitch()
        
    async def run_autonomous(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the agent autonomously to achieve a goal.
        
        Args:
            goal: The research/action goal.
            context: Optional initial context.
            
        Returns:
            Dictionary containing the research process and findings.
        """
        state = {
            'goal': goal,
            'steps': 0,
            'thoughts': [],
            'actions': [],
            'observations': [],
            'total_cost': 0.0,
            'start_time': time.time(),
            'completed': False,
            'context': context or {}
        }
        
        print(f"\n[ReActAgent] Starting autonomous loop for: {goal}")
        
        while True:
            # 1. Check Kill Switch
            should_stop, reason = self.kill_switch.check(state)
            if should_stop:
                print(f"[ReActAgent] Loop terminated: {reason}")
                break
                
            state['steps'] += 1
            print(f"\n--- Step {state['steps']} ---")
            
            # 2. THINK
            thought_context = self._build_thought_context(state)
            if hasattr(self.llm, 'complete_async'):
                thought = await self.llm.complete_async(thought_context)
            else:
                import asyncio
                thought = await asyncio.to_thread(self.llm.complete, thought_context)
            
            state['thoughts'].append(thought)
            print(f"Thought: {thought[:150]}...")
            
            # Check for completion
            if "GOAL_ACHIEVED" in thought.upper() or "SUFFICIENT" in thought.upper():
                state['completed'] = True
                print("[ReActAgent] Agent signaled goal achieved.")
                continue
                
            # 3. ACT
            action = await self._decide_and_act(thought, state)
            state['actions'].append(action)
            state['total_cost'] += action.get('cost', 0.0)
            
            # 4. OBSERVE
            observation = self._generate_observation(action)
            state['observations'].append(observation)
            print(f"Observation: {observation}")
            
        # 5. SYNTHESIZE
        final_summary = await self._synthesize_findings(state)
        
        return {
            "success": state['completed'],
            "goal": goal,
            "summary": final_summary,
            "steps": state['steps'],
            "total_cost": state['total_cost'],
            "reason": reason,
            "process": state
        }
    
    def _build_thought_context(self, state: Dict) -> str:
        recent_obs = state['observations'][-3:] if state['observations'] else ["None yet"]
        obs_text = "\n".join([f"- {o}" for o in recent_obs])
        
        available_tools = "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])
        
        return f"""Goal: {state['goal']}

Steps taken: {state['steps']}

Recent Observations:
{obs_text}

Available Actions (Tools):
{available_tools}
- GOAL_ACHIEVED: Signal if you have sufficient information

Instructions:
Think step-by-step about what to do next. If you have enough info, type GOAL_ACHIEVED.
Otherwise, choose one tool to call.
"""

    async def _decide_and_act(self, thought: str, state: Dict) -> Dict:
        """Decide which tool to call based on thought."""
        thought_lower = thought.lower()
        
        # Simple heuristic matching (improvement: LLM can output structured JSON)
        # For now, we'll keep the logic from case4 but more robust
        
        best_tool = None
        for name in self.tools:
            if name.lower() in thought_lower:
                best_tool = self.tools[name]
                break
        
        if not best_tool and self.tools:
            # Fallback to first tool or a default search if it exists
            best_tool = list(self.tools.values())[0]

        if best_tool:
            # In a real implementation, we'd extract arguments from thought
            # For this refactor, we'll use state['goal'] as a default arg if applicable
            # or try to extract from thought.
            
            # Mocking argument extraction
            args = {"query": state['goal']} 
            
            try:
                result = best_tool.execute(**args)
                return {
                    'type': best_tool.name,
                    'result': result,
                    'cost': 0.01 # Mock cost
                }
            except Exception as e:
                return {
                    'type': 'error',
                    'result': str(e),
                    'cost': 0.0
                }
        
        return {
            'type': 'none',
            'result': 'No suitable tool found',
            'cost': 0.0
        }

    def _generate_observation(self, action: Dict) -> str:
        action_type = action.get('type')
        result = action.get('result', {})
        
        if action_type == 'error':
            return f"Error occurred: {result}"
        if action_type == 'none':
            return "No action taken."
            
        # Basic formatting of results
        if isinstance(result, dict):
            if 'results' in result:
                count = len(result['results'])
                return f"Found {count} results via {action_type}."
            if 'title' in result:
                return f"Retrieved info about: {result['title']}"
        
        return f"Action {action_type} completed."

    async def _synthesize_findings(self, state: Dict) -> str:
        prompt = f"""Synthesize the findings for the goal: {state['goal']}
        
Observations:
{chr(10).join(state['observations'])}

Provide a concise summary of the key findings."""
        
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt)
        else:
            import asyncio
            return await asyncio.to_thread(self.llm.complete, prompt)

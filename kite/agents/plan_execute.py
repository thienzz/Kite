"""
Plan-and-Execute Agent
Decomposes complex goals into steps before execution.
"""

import json
from typing import List, Dict, Optional, Any
from ..agent import Agent

from dataclasses import dataclass, field

@dataclass
class Plan:
    """A plan with steps."""
    goal: str
    steps: List[str]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)

class PlanExecuteAgent(Agent):
    """
    Agent that implements the Plan-and-Execute pattern.
    1. Plan: Decompose goal into steps upfront.
    2. Execute: Run each step sequentially.
    3. Re-plan: Adjust remaining steps if needed.
    """
    
    def __init__(self, name, system_prompt, llm, tools, framework, max_iterations=10):
        super().__init__(name, system_prompt, llm, tools, framework)
        self.max_iterations = max_iterations

    async def run(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Override base run to use plan-and-execute logic for this agent."""
        return await self.run_plan(goal, context)

    async def run_plan(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the planning and execution loop.
        """
        print(f"\n[PlanAndExecute] Goal: {goal}")
        
        # Step 1: PLAN
        print("  [Step 1] Creating initial plan...")
        plan_obj = await self._create_plan_obj(goal, context)
        
        # Limit plan to max_iterations
        if len(plan_obj.steps) > self.max_iterations:
            print(f"  [WARNING] Plan has {len(plan_obj.steps)} steps, truncating to {self.max_iterations}")
            plan_obj.steps = plan_obj.steps[:self.max_iterations]
            
        print(f"  [OK] Generated {len(plan_obj.steps)} steps")
        
        results = []
        # Step 2: EXECUTE
        for i, step in enumerate(plan_obj.steps, 1):
            print(f"\n--- Step {i}/{len(plan_obj.steps)}: {step} ---")
            
            # Execute step
            result = await self._execute_step(step, results, context)
            results.append(result)
            
            # Check if we need to replan
            if result.get("needs_replan"):
                print("  [PlanAndExecute] Replanning required...")
                new_steps = await self._replan(goal, results, plan_obj.steps[i:])
                plan_obj.steps = plan_obj.steps[:i] + new_steps
                print(f"  [OK] Updated plan with {len(new_steps)} remaining steps")

        # Step 3: SYNTHESIZE
        print("\n  [Step 3] Synthesizing final answer...")
        final_answer = await self._synthesize_results(goal, results)
        
        return {
            "success": all(r.get('success', False) for r in results),
            "goal": goal,
            "plan": plan_obj.steps,
            "results": results,
            "answer": final_answer
        }

    async def _create_plan_obj(self, goal: str, context: Optional[Dict]) -> Plan:
        """Create a plan using LLM."""
        tools_desc = "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])

        prompt = f"""You are a strategic planner. Create a step-by-step plan to achieve this goal.

Goal: {goal}

Available tools:
{tools_desc}

{f"Context: {json.dumps(context)}" if context else "None"}

Create a detailed plan with actionable steps.
Respond with JSON:
{{
  "reasoning": "Why this plan will work",
  "steps": [
    "Step 1: ...",
    "Step 2: ..."
  ]
}}"""

        response = await self._llm_complete(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                steps = data.get("steps", [])
                
                # Robust step parsing: handle if LLM returns dicts instead of strings
                processed_steps = []
                for s in steps:
                    if isinstance(s, dict):
                        # Use description or some field if available, else stringify
                        processed_steps.append(s.get("description", s.get("step", str(s))))
                    else:
                        processed_steps.append(str(s))
                
                print(f"  [Plan Reasoning] {data.get('reasoning', 'No reasoning provided')}")
                return Plan(goal=goal, steps=processed_steps)
            return Plan(goal=goal, steps=[response.strip()])
        except Exception as e:
            print(f"  [Error] Failed to parse plan JSON: {e}")
            return Plan(goal=goal, steps=[goal])

    async def _execute_step(self, step: str, previous_results: List, context: Optional[Dict]) -> Dict:
        """Execute a single step."""
        # Use history from previous results for context
        history = "\n".join([
            f"Step: {r.get('step')}\nResult: {r.get('result')}"
            for r in previous_results[-3:]
        ])
        
        # We call the super().run to avoid infinite recursion while still using the agent's core engine
        step_result = await super().run(step, context=context)
        
        return {
            "step": step,
            "result": step_result.get('response', 'Error'),
            "success": step_result.get('success', False),
            "needs_replan": "ERROR" in str(step_result.get('response', '')).upper() or not step_result.get('success')
        }

    async def _replan(self, goal: str, completed_steps: List, remaining_steps: List[str]) -> List[str]:
        """Replan if something went wrong."""
        prompt = f"""Something went wrong or the environment changed. Create a new plan for the remaining work.

Original goal: {goal}

Completed steps:
{json.dumps([{str(r['step']): r['result']} for r in completed_steps], indent=2)}

Remaining steps originally planned:
{json.dumps(remaining_steps, indent=2)}

Create a new plan to complete the goal.
Respond with JSON array of new steps:
["New Step 1", "New Step 2"]"""

        response = await self._llm_complete(prompt)

        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return remaining_steps
        except:
            return remaining_steps

    async def _synthesize_results(self, goal: str, results: List[Dict]) -> str:
        """Synthesize all results into final answer."""
        prompt = f"""Synthesize the results to answer the goal.

Goal: {goal}

Execution Results:
{json.dumps(results, indent=2)}

Provide a clear, comprehensive answer to the original goal."""

        return await self._llm_complete(prompt)

    async def _llm_complete(self, prompt: str) -> str:
        """Helper for LLM completion."""
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt)
        else:
            import asyncio
            return await asyncio.to_thread(self.llm.complete, prompt)

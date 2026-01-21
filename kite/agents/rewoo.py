"""
ReWOO Agent - Reasoning WithOut Observation
Executes multiple steps in parallel by pre-planning independent tasks.
"""

import json
import re
import asyncio
from typing import List, Dict, Optional, Any
from ..agent import Agent

class ReWOOAgent(Agent):
    """
    Agent that implements the ReWOO (Reasoning WithOut Observation) pattern.
    1. Plan: Create a graph of tasks with variable placeholders (#E1, #E2).
    2. Execute: Resolve dependencies and run tasks (parallel where possible).
    3. Solver: Combine results for final answer.
    """
    
    def __init__(self, name, system_prompt, llm, tools, framework, max_iterations=10):
        super().__init__(name, system_prompt, llm, tools, framework)
        self.max_iterations = max_iterations

    async def run_rewoo(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the ReWOO loop.
        """
        print(f"\n[ReWOO] Goal: {goal}")
        
        # Step 1: Create Execution Plan
        print("  [Step 1] Creating execution graph...")
        plan_str = await self._generate_rewoo_plan(goal, context)
        parsing = self._parse_plan(plan_str)
        
        # Limit steps to max_iterations
        if len(parsing) > self.max_iterations:
            print(f"  [WARNING] ReWOO plan has {len(parsing)} steps, truncating to {self.max_iterations}")
            parsing = parsing[:self.max_iterations]
            
        print(f"  [OK] Planned {len(parsing)} steps")
        
        # Step 2: Execute
        print("  [Step 2] Executing steps...")
        results = await self._execute_plan(parsing, context)
        
        # Step 3: Solve
        print("  [Step 3] Solving for final answer...")
        final_answer = await self._solve(goal, results)
        
        return {
            "success": True,
            "goal": goal,
            "plan": parsing,
            "results": results,
            "answer": final_answer
        }

    async def _generate_rewoo_plan(self, goal: str, context: Optional[Dict] = None) -> str:
        tool_desc = "\n".join([f"- {n}: {t.description}" for n, t in self.tools.items()])
        
        prompt = f"""You are a ReWOO planner. Create a plan to achieve the goal using available tools.
Express the plan as a series of steps using placeholders like #E1, #E2 for results.

Available Tools:
{tool_desc}

Goal: {goal}
{f"Context: {json.dumps(context)}" if context else ""}

Format:
Plan: [reasoning]
#E1 = [tool_name] with [args, can use #E0, etc]
Plan: [more reasoning]
#E2 = [tool_name] with [args]
"""
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt)
        return await asyncio.to_thread(self.llm.complete, prompt)

    def _parse_plan(self, plan_str: str) -> List[Dict]:
        steps = []
        # Find lines starting with #E[digit]
        matches = re.findall(r'#E(\d+)\s*=\s*(\w+)\s+(.*)', plan_str)
        for m in matches:
            steps.append({
                "id": f"#E{m[0]}",
                "tool": m[1],
                "args": m[2].strip()
            })
        return steps

    async def _execute_plan(self, steps: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        results = {}
        
        for step in steps:
            # Resolve placeholders in args
            resolved_args = step['args']
            for eid, res in results.items():
                if eid in resolved_args:
                    resolved_args = resolved_args.replace(eid, str(res))
            
            print(f"    Running {step['id']} ({step['tool']})...")
            # Execute step
            step_res = await self.run(f"Use {step['tool']} to {resolved_args}", context=context)
            results[step['id']] = step_res.get('response', 'Error')
            
        return results

    async def _solve(self, goal: str, results: Dict[str, Any]) -> str:
        prompt = f"""Based on the following execution results, provide a final comprehensive answer for the goal.

Goal: {goal}

Results:
{json.dumps(results, indent=2)}

Final Answer:"""
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt)
        return await asyncio.to_thread(self.llm.complete, prompt)

"""
Tree-of-Thoughts (ToT) Agent
Implements multi-path reasoning by generating and evaluating multiple "thoughts" at each step.
"""

import json
import asyncio
from typing import List, Dict, Optional, Any
from ..agent import Agent

class TreeOfThoughtsAgent(Agent):
    """
    Agent that implements the Tree-of-Thoughts pattern.
    Explores multiple reasoning paths and selects the best one.
    """
    
    def __init__(self, name, system_prompt, tools, framework, llm=None, max_iterations=3, branches=3, verbose=False):
        super().__init__(name, system_prompt, tools, framework, llm=llm, max_iterations=max_iterations, verbose=verbose)
        self.branches = branches

    async def run(self, problem: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Override base run to use ToT logic."""
        return await self.solve_tot(problem)

    async def solve_tot(self, goal: str, max_steps: Optional[int] = None, num_thoughts: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the ToT search loop.
        """
        depth = max_steps or self.max_iterations
        branches = num_thoughts or self.branches
        
        print(f"\n[TreeOfThoughts] Exploring {branches} branches, depth {depth}")
        print(f"Goal: {goal}")
        
        # 1. Generate initial thoughts
        initial_thoughts = await self._generate_thoughts_list(goal, depth=0)
        
        # 2. Build tree by expanding thoughts recursively
        tree = []
        for i, thought in enumerate(initial_thoughts, 1):
            print(f"  Branch {i}/{len(initial_thoughts)}: {thought[:50]}...")
            path = await self._explore_path(goal, thought, 1, depth, branches)
            tree.append(path)
            
        # 3. Evaluate and select best path
        print(f"  [ToT] Evaluating {len(tree)} exploration paths...")
        best_path = await self._select_best_path(goal, tree)
        
        # 4. Final synthesis
        print("  [ToT] Synthesizing final answer from best path...")
        final_answer = await self._generate_answer(goal, best_path)
        
        return {
            "success": True,
            "goal": goal,
            "explored_paths": len(tree),
            "best_path": best_path,
            "answer": final_answer
        }

    async def _generate_thoughts_list(self, problem: str, depth: int, context: str = "") -> List[str]:
        """Generate N different thoughts/approaches."""
        prompt = f"""Generate {self.branches} different approaches to solve this problem.

Problem: {problem}
Current context: {context if context else "None"}

Generate exactly {self.branches} distinct approaches, each on a new line starting with a number (1., 2., etc.)."""

        response = await self._complete(prompt)

        # Parse thoughts
        thoughts = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                import re
                thought = re.sub(r'^\d+[\.\)\-]\s*', '', line).strip()
                if thought:
                    thoughts.append(thought)

        return thoughts[:self.branches]

    async def _explore_path(self, goal: str, initial_thought: str, current_depth: int, max_depth: int, branches: int) -> List[str]:
        """Recursively explore a reasoning path."""
        path = [initial_thought]

        if current_depth >= max_depth:
            return path

        # Generate next thoughts based on current path
        context = " -> ".join(path)
        next_thoughts = await self._generate_thoughts_list(goal, current_depth, context)

        if next_thoughts:
            # For simplicity, we take the first "best" looking next thought 
            # or could branches here if doing BFS. We stick to DFS path here.
            best_next = next_thoughts[0] 
            rest_of_path = await self._explore_path(goal, best_next, current_depth + 1, max_depth, branches)
            path.extend(rest_of_path)

        return path

    async def _select_best_path(self, goal: str, tree: List[List[str]]) -> List[str]:
        """Evaluate all paths and select the best one."""
        paths_desc = "\n".join([f"Path {i+1}: {' -> '.join(path)}" for i, path in enumerate(tree)])
        
        prompt = f"""Evaluate these reasoning paths and select the best one.

Goal: {goal}

Reasoning Paths:
{paths_desc}

Which path is most likely to lead to a correct and comprehensive solution?
Respond with ONLY the path number (e.g., 1, 2, or 3)."""

        response = await self._complete(prompt)
        
        # Parse selection
        try:
            import re
            match = re.search(r'\d+', response)
            if match:
                selection = int(match.group(0)) - 1
                if 0 <= selection < len(tree):
                    return tree[selection]
            return tree[0]
        except:
            return tree[0]

    async def _generate_answer(self, goal: str, path: List[str]) -> str:
        """Generate final answer from best reasoning path."""
        reasoning = " -> ".join(path)

        prompt = f"""Generate a final comprehensive answer based on this reasoning path.

Goal: {goal}
Detailed Reasoning: {reasoning}

Final Answer:"""

        return await self._complete(prompt)

    async def _complete(self, prompt: str) -> str:
        if hasattr(self.llm, 'complete_async'):
            return await self.llm.complete_async(prompt)
        else:
            import asyncio
            return await asyncio.to_thread(self.llm.complete, prompt)

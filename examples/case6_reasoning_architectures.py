"""
CASE 6: AGENT REASONING ARCHITECTURES
=====================================
A comparative showcase of different agent cognitive architectures.
Problem: "Design a scalable Real-Time Analytics System for 10M concurrent users."

Architectures:
1. ReAct (Reasoning + Acting): Standard loop.
2. ReWOO (Reasoning without Observation): Plans everything upfront, then fills variables.
3. Tree of Thoughts (ToT): Explores multiple solution paths.
"""

import os
import sys
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite
from kite.agents.react_agent import ReActAgent
from kite.agents.rewoo import ReWOOAgent
from kite.agents.tot import TreeOfThoughtsAgent
from kite.llm_providers import LLMFactory

# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("\n" + "=" * 80)
    print("CASE 6: AGENT ARCHITECTURES SHOWCASE")
    print("=" * 80)
    print("Problem: Design a Scalable Real-time Analytics System (10M users)")
    
    ai = Kite()

    # helper to get model
    def get_model(name):
        provider, model = name.split("/", 1)
        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        return LLMFactory.create(provider, model, api_key=api_key)
    


    # We use a mock search tool for consistency
    search_tool = ai.tools.get("search_google") or (lambda x: f"Found findings for {x}")
    # Fix: Wrap lambda as proper tool if needed, but for now we assume framework has search
    # Or strict definition:
    from kite.tool import Tool
    class MockSearch(Tool):
        def __init__(self): super().__init__("search", self.execute, "Search technical docs")
        async def execute(self, query: str, **kwargs): return f"[Mock Search Result for '{query}': Kafka, Flink, Druid are good options.]"
    
    tools = [MockSearch()]
    
    # ------------------------------------------------------------------------
    # 1. ReAct Agent
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("1. ReAct Agent (Standard)")
    print("-" * 40)
    react = ReActAgent(
        name="Engineer_ReAct",
        system_prompt="You are a System Architect. Think step-by-step.",
        tools=tools,
        framework=ai,
        verbose=True
    )
    start = time.time()
    res_react = await react.run("Design high-level architecture for real-time analytics system (10M daily users). Output key components.")
    print(f" Time: {time.time()-start:.1f}s")
    
    # ------------------------------------------------------------------------
    # 2. ReWOO Agent
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("2. ReWOO (Reasoning Without Observation)")
    print("-" * 40)
    # ReWOO is great for planning efficient tool use
    rewoo = ReWOOAgent(
        name="Engineer_ReWOO",
        system_prompt="You are a System Architect. Plan the design steps first.",
        tools=tools,
        framework=ai,
        verbose=True
    )
    start = time.time()
    res_rewoo = await rewoo.run("Design high-level architecture for real-time analytics system (10M daily users). Output key components.")
    print(f"Time: {time.time()-start:.1f}s")

    # ------------------------------------------------------------------------
    # 3. Tree of Thoughts (ToT)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("3. Tree of Thoughts (ToT)")
    print("-" * 40)
    # ToT explores multiple branches
    tot = TreeOfThoughtsAgent(
        name="Engineer_ToT",
        system_prompt="You are a System Architect. Explore 3 different architectural patterns (Lambda vs Kappa vs Microservices).",
        tools=tools,
        framework=ai,
        verbose=True
    )
    start = time.time()
    # ToT usually takes a simpler input as it generates thoughts itself
    res_tot = await tot.run("Design scalable analytics architecture.")
    print(f" Time: {time.time()-start:.1f}s")
    
    print("\n" + "=" * 80)
    print("REASONING COMPARISON COMPLETE")
    print("Check logs to see difference in thought process:\n- ReAct: Linear\n- ReWOO: Planned Batching\n- ToT: Branching")

if __name__ == "__main__":
    asyncio.run(main())

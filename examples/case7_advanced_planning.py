"""
Case Study 7: Advanced Planning Patterns
Demonstrates a realistic Market Research scenario using Plan-and-Execute.
"""

import asyncio
import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite.core import Kite

async def main():
    print("="*80)
    print("CASE STUDY 7: AI STARTUP MARKET RESEARCH")
    print("="*80)

    # Initialize Kite (using local Ollama for reliability in this demo)
    ai = Kite()
    
    # Define Tools
    def market_search(query: str):
        """Simulated search for market data."""
        data = {
            "ai agents market 2024": "Market size estimated at $5B, growing 30% YoY.",
            "top ai agent startups": "Notable players: LangChain, CrewAI, AutoGPT.",
            "competitor pricing": "Typical SaaS model: $20/user/month for mid-tier."
        }
        for k, v in data.items():
            if k in query.lower():
                return v
        return f"General research findings for {query}: High demand for autonomous workflows."

    def financial_analyst(data_point: str):
        """Analyze financial data points."""
        if "5B" in data_point:
            return "Analysis: High potential for disruption. TAM is significant."
        return "Analysis: Stable growth observed in this segment."

    ai.create_tool("search", market_search, "Search for market data")
    ai.create_tool("analyze", financial_analyst, "Analyze financial data")

    # ========================================================================
    # STEP 1: Plan-and-Execute (Market Entry Strategy)
    # ========================================================================
    print("\n" + "-"*40)
    print("1. Strategy: PLAN-AND-EXECUTE (Market Entry)")
    print("-"*40)
    
    # New unified API
    planner = ai.create_planning_agent(
        name="StrategySpecialist",
        system_prompt="You are a senior market analyst. Create a detailed entry strategy.",
        strategy="plan-and-execute",
        tools=[ai.tools.get("search"), ai.tools.get("analyze")]
    )
    
    goal = "Research AI Agent market size 2024, identify top startups, and suggest a pricing model."
    res1 = await planner.run_plan(goal)
    
    print(f"\n[FINAL STRATEGY REPORT]\n{res1['answer']}")

    # ========================================================================
    # STEP 2: ReWOO (Quick Competitor Check)
    # ========================================================================
    print("\n" + "-"*40)
    print("2. Strategy: ReWOO (Parallel Search)")
    print("-"*40)
    
    rewoo = ai.create_planning_agent(
        name="FastSearcher",
        system_prompt="Execute multiple search tasks in parallel.",
        strategy="rewoo",
        tools=[ai.tools.get("search")]
    )
    
    res2 = await rewoo.run_rewoo("Get news for LangChain, CrewAI, and AutoGPT simultaneously.")
    print(f"\n[PARALLEL RESULTS]\n{res2['answer']}")

    # ========================================================================
    # STEP 3: Tree-of-Thoughts (Complex Reasoning)
    # ========================================================================
    print("\n" + "-"*40)
    print("3. Strategy: TREE-OF-THOUGHTS (Multi-path Reasoning)")
    print("-"*40)
    
    tot = ai.create_planning_agent(
        name="DeepThinker",
        strategy="tot",
        max_iterations=2  # Depth of reasoning
    )
    
    complex_problem = "Evaluate the impact of high-interest rates on AI startup funding and suggest 3 mitigation strategies."
    res3 = await tot.run(complex_problem)
    
    print(f"\n[THOUGHT TREE ANALYSIS]\n{res3['answer']}")

    print("\n" + "="*80)
    print("[OK] CASE STUDY 7 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())

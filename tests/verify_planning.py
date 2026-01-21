import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

async def test_all():
    ai = Kite()
    
    # 1. Test Plan-and-Execute
    print("\n--- Testing Plan-and-Execute ---")
    planner = ai.create_planning_agent(strategy="plan-and-execute", max_iterations=2)
    assert planner.name == "PlanAndExecuteAgent"
    assert planner.max_iterations == 2
    
    def dummy_tool(arg: str): return f"Processed {arg}"
    ai.create_tool("dummy", dummy_tool, "A dummy tool")
    
    res1 = await planner.run_plan("Test goal")
    print(f"Result: {res1['success']}")
    assert "answer" in res1
    
    # 2. Test ReWOO
    print("\n--- Testing ReWOO ---")
    rewoo = ai.create_planning_agent(strategy="rewoo", max_iterations=5)
    assert rewoo.name == "RewooAgent"
    assert rewoo.max_iterations == 5
    
    def search(query: str): return f"Info about {query}"
    ai.create_tool("search", search, "Search tool")
    
    res2 = await rewoo.run_rewoo("Research X and Y")
    print(f"Result: {res2['success']}")
    assert "answer" in res2
    
    # 3. Test Tree-of-Thoughts
    print("\n--- Testing Tree-of-Thoughts ---")
    tot = ai.create_planning_agent(strategy="tot", max_iterations=2)
    assert tot.name == "TotAgent"
    assert tot.max_iterations == 2
    
    res3 = await tot.solve_tot("Solve a riddle", num_thoughts=2)
    print(f"Result: {res3['success']}")
    assert "answer" in res3
    assert len(res3["best_path"]) <= 2

    print("\n[OK] ALL PLANNING AGENTS VERIFIED")

if __name__ == "__main__":
    asyncio.run(test_all())

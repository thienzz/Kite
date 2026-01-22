import asyncio
import pytest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

@pytest.mark.async_timeout(30)
class TestPlanningAgents:
    @pytest.fixture
    def ai(self):
        return Kite()

    @pytest.mark.asyncio
    async def test_plan_execute_api(self, ai):
        # Test unified API defaults
        planner = ai.create_planning_agent(strategy="plan-and-execute", max_iterations=2)
        assert planner.name == "PlanAndExecuteAgent"
        assert planner.max_iterations == 2
        
        # Test execution
        def dummy_tool(arg: str):
            return f"Processed {arg}"
        ai.create_tool("dummy", dummy_tool, "A dummy tool")
        
        # We need to mock or ensure the LLM returns a plan with 2 steps to test truncation
        # But for basic verification, let's just run it
        res = await planner.run_plan("Test goal")
        assert "answer" in res
        assert "plan" in res

    @pytest.mark.asyncio
    async def test_rewoo_api(self, ai):
        rewoo = ai.create_planning_agent(strategy="rewoo", max_iterations=5)
        assert rewoo.name == "RewooAgent"
        assert rewoo.max_iterations == 5
        
        def search(query: str):
            return f"Info about {query}"
        ai.create_tool("search", search, "Search tool")
        
        res = await rewoo.run_rewoo("Research X and Y")
        assert "answer" in res
        assert "results" in res

    @pytest.mark.asyncio
    async def test_tot_api(self, ai):
        tot = ai.create_planning_agent(strategy="tot", max_iterations=2)
        assert tot.name == "TotAgent"
        assert tot.max_iterations == 2
        
        res = await tot.solve_tot("Solve a riddle", num_thoughts=2)
        assert "answer" in res
        assert len(res["path"]) <= 2

if __name__ == "__main__":
    import sys
    pytest.main([__file__])

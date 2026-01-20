import pytest
import asyncio
import time
from kite import Kite

def test_process_parallel():
    async def run_test():
        ai = Kite()
        
        # Create a mock tool
        async def delayed_tool(input_val):
            await asyncio.sleep(0.1) # Much shorter wait
            return {"result": f"processed {input_val}"}
            
        tool = ai.create_tool("delayed_tool", delayed_tool, "A tool with delay")
        
        # Create agents with mock LLM
        agent1 = ai.create_agent("Agent1", "Helpful", tools=[tool], llm_provider="mock")
        agent2 = ai.create_agent("Agent2", "Helpful", tools=[tool], llm_provider="mock")
        
        tasks = [
            {"input": "Task 1", "agent": agent1},
            {"input": "Task 2", "agent": agent2}
        ]
        
        start_time = time.time()
        results = await ai.process_parallel(tasks)
        end_time = time.time()
        
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is True
        
        duration = end_time - start_time
        print(f"Parallel duration: {duration:.2f}s")
        assert duration < 1.8 

    asyncio.run(run_test())

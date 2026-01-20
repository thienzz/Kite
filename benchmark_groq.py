import asyncio
import os
import time
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_framework import AgenticAI

async def benchmark_groq():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY in .env to run this benchmark.")
        return

    print("="*80)
    print("GROQ BENCHMARK: ULTRALIGHT SPEED (TARGET 1-2s)")
    print("="*80)

    # Force Groq provider
    os.environ["llm_provider"] = "groq"
    os.environ["slm_provider"] = "groq"
    
    ai = AgenticAI()
    
    # Simple tool
    def check_stock(item: str):
        return f"Stock for {item}: 100"
    
    tool = ai.create_tool("check_stock", check_stock, "Check stock")
    agent = ai.create_agent("Speedy", "Help users fast.", tools=[tool])
    
    test_queries = [
        {"input": "Laptop stock?", "agent": agent},
        {"input": "Phone stock?", "agent": agent},
        {"input": "Tablet stock?", "agent": agent}
    ]
    
    print(f"\n[EXEC] Processing {len(test_queries)} queries in PARALLEL via Groq...")
    start_time = time.time()
    results = await ai.process_parallel(test_queries)
    end_time = time.time()
    
    print(f"\n[OK] Parallel processing completed in {end_time - start_time:.2f}s")
    
    for i, r in enumerate(results):
        print(f"   Q{i+1}: {test_queries[i]['input']} -> {r.get('response')[:50]}...")

if __name__ == "__main__":
    asyncio.run(benchmark_groq())

import asyncio
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kite.llm_providers import LLMFactory

async def test_concurrency():
    print("Testing Concurrency with MockLLMProvider...")
    # Using MockProvider as it's safe and fast for testing the logic
    llm = LLMFactory.create("mock")
    
    start_time = time.time()
    
    # Run 5 calls concurrently
    tasks = [llm.complete_async(f"Task {i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Completed {len(results)} tasks in {elapsed:.4f} seconds")
    for r in results:
        print(f"  - {r}")

    # Test with Ollama if configured (just to be sure)
    try:
        ollama = LLMFactory.create("ollama")
        print("\nTesting Concurrency with OllamaProvider (if running)...")
        start_time = time.time()
        tasks = [ollama.complete_async("Say 'Hi'") for _ in range(2)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        print(f"Completed Ollama tasks in {elapsed:.4f} seconds")
    except Exception as e:
        print(f"\nOllama test skipped or failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_concurrency())

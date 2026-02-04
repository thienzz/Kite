import asyncio
from kite.llm_providers import OllamaProvider
from kite.monitoring import MetricsCollector

async def test_token_tracking():
    print("Testing Token Usage Tracking...")
    
    # 1. Initialize Components
    metrics = MetricsCollector(enable_prometheus=False)
    llm = OllamaProvider(model="deepseek-r1:14b")
    
    # 2. Make Request with metrics
    print("   Sending request...")
    try:
        response = await llm.chat_async(
            messages=[{"role": "user", "content": "Hello! Say hi."}],
            metrics=metrics
        )
        print(f"   Response: {response}")
        
        # 3. Verify Metrics
        report = metrics.get_detailed_report()
        print("\n   [Report Preview]")
        print(report)
        
        data = metrics.get_metrics()
        usage_key = f"llm_usage.{llm.model}"
        
        if usage_key in data:
            usage = data[usage_key]
            print(f"\n   Usage Data: {usage}")
            assert usage['count'] > 0
            assert usage['tokens_in'] > 0
            assert usage['tokens_out'] > 0
            # Cost is 0 for local models, but key should exist
            assert 'cost' in usage
            print("   [PASS] Metrics recorded successfully.")
        else:
            print(f"   [FAIL] Usage key {usage_key} not found in metrics.")
            
    except Exception as e:
        print(f"   [FAIL] Error: {e}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_token_tracking())

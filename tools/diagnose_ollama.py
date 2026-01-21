import time
import json
import httpx
import asyncio

async def diagnose_ollama(base_url="http://localhost:11434", model="llama3.1:8b"):
    print("="*80)
    print(f"OLLAMA DIAGNOSTIC TOOL")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Check Connection
        try:
            start = time.time()
            res = await client.get(f"{base_url}/api/tags")
            latency = time.time() - start
            if res.status_code == 200:
                models = [m['name'] for m in res.json().get('models', [])]
                print(f"[OK] Connected to Ollama (Ping: {latency*1000:.2f}ms)")
                print(f"Available Models: {', '.join(models)}")
                if model not in models:
                    print(f"[WARN] Target model '{model}' not found in local tags!")
            else:
                print(f"[ERROR] Ollama returned {res.status_code}")
                return
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama at {base_url}: {e}")
            return

    # 2. Test Non-Streaming Latency
    print(f"\n[TEST] Non-streaming generation (Model: {model})...")
    payload = {
        "model": model,
        "prompt": "Write a one-sentence greeting.",
        "stream": False,
        "options": {"num_predict": 20}
    }
    
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{base_url}/api/generate", json=payload)
        latency = time.time() - start
        
        if res.status_code == 200:
            data = res.json()
            response_text = data.get('response', '').strip()
            total_duration = data.get('total_duration', 0) / 1e9 # ns to s
            eval_count = data.get('eval_count', 1)
            tps = eval_count / total_duration if total_duration > 0 else 0
            
            print(f"   Response: \"{response_text}\"")
            print(f"   Wall Clock: {latency:.2f}s")
            print(f"   Ollama Reported Duration: {total_duration:.2f}s")
            print(f"   Inference Speed: {tps:.2f} tokens/s")
        else:
            print(f"   [ERROR] Generation failed: {res.text}")
    except Exception as e:
        print(f"   [ERROR] Request failed: {e}")

    # 3. Test Streaming Latency (First Token Time)
    print(f"\n[TEST] Streaming generation (First Token Latency)...")
    payload["stream"] = True
    
    try:
        start = time.time()
        first_token_time = None
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{base_url}/api/generate", json=payload) as response:
                async for line in response.aiter_lines():
                    if not first_token_time:
                        first_token_time = time.time() - start
                    if line:
                        chunk = json.loads(line)
                        if chunk.get('done'):
                            break
        
        print(f"   Time to First Token (TTFT): {first_token_time:.2f}s")
        print(f"   Total Streaming Time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"   [ERROR] Streaming failed: {e}")

if __name__ == "__main__":
    import os
    target_model = os.getenv("LLM_MODEL", "llama3.1:8b")
    asyncio.run(diagnose_ollama(model=target_model))

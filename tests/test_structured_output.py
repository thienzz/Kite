import asyncio
import json
from kite.llm_providers import OllamaProvider

async def test_structured_output():
    print("Testing Structured Output (JSON Mode)...")
    
    # 1. Initialize Provider
    llm = OllamaProvider(model="deepseek-r1:14b", timeout=60.0)
    
    # 2. Define Schema
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "topics": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["sentiment", "score", "topics"]
    }
    
    # 3. Prompt
    prompt = "Analyze this review: 'The product works great but shipping was slow.' Return results in JSON."
    
    # 4. Call LLM with schema
    print(f"   Prompt: {prompt}")
    print("   Requesting JSON schema...")
    
    try:
        response = await llm.chat_async(
            messages=[{"role": "user", "content": prompt}],
            response_schema=schema
        )
        
        print(f"   Raw Response: {response}")
        
        # 5. Validate
        data = json.loads(response)
        assert "sentiment" in data
        assert "score" in data
        assert isinstance(data["topics"], list)
        
        print("   [PASS] Output is valid JSON and matches schema keys.")
        print(f"   Parsed Data: {data}")
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        # If model is offline, we skip or mock. 
        # But here we assume local ollama is up as per user context.

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_structured_output())

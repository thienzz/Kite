import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

async def test_llm_routing():
    print("Testing Semantic Router with LLM strategy...")
    
    # Initialize framework with LLM strategy
    ai = Kite(config={
        "router_strategy": "llm",
        "llm_provider": "groq",
        "llm_model": "llama-3.3-70b-versatile"
    })
    
    # Add some routes
    ai.semantic_router.add_route(
        name="weather",
        description="Questions about weather and forecast",
        examples=["How is the weather today?", "Will it rain tomorrow?"],
        handler=lambda q: f"Weather response for: {q}"
    )
    
    ai.semantic_router.add_route(
        name="math",
        description="Mathematical calculations and expressions",
        examples=["What is 2+2?", "Solve for x: 2x = 10"],
        handler=lambda q: f"Math response for: {q}"
    )
    
    # Test queries
    queries = [
        "What's the forecast for Hanoi?",
        "Calculate the square root of 144",
        "Is it sunny in Saigon?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = await ai.semantic_router.route(query)
        print(f"Result: {result['route']} (confidence: {result['confidence']:.2f})")
        print(f"Response: {result['response']}")

async def test_hybrid_routing():
    print("\n" + "="*50)
    print("Testing Semantic Router with Hybrid strategy...")
    
    ai = Kite(config={
        "router_strategy": "hybrid",
        "router_confidence_threshold": 0.8
    })
    
    # Add routes
    ai.semantic_router.add_route(
        name="order",
        description="Order tracking and management",
        examples=["Track my order", "Where is my package?"],
        handler=lambda q: f"Order specialist result for {q}"
    )
    
    # Query that might have low confidence in embedding but clear to LLM
    query = "Where are my goods?" 
    
    print(f"\nQuery: {query}")
    result = await ai.semantic_router.route(query)
    print(f"Result: {result['route']} (confidence: {result['confidence']:.2f})")

if __name__ == "__main__":
    asyncio.run(test_llm_routing())
    asyncio.run(test_hybrid_routing())

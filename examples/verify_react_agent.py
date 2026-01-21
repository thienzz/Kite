"""
Verification Script for Modernized ReActAgent
"""

import os
import sys
import asyncio

# Add framework to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

async def main():
    print("="*80)
    print("VERIFYING ENHANCED ReAct AGENT")
    print("="*80)
    
    ai = Kite()
    
    # 1. Setup Mock Tools
    def web_search(query: str):
        """Search the web for general facts, descriptions, and colors. Use this for the 'color' part of the goal."""
        print(f"     [Action] Searching for: {query}")
        if "apple" in query.lower():
            return "FACT: Apples are primarily RED, GREEN, or YELLOW in color."
        return "No specific information found."
        
    def get_price(item: str):
        """Get the CURRENT MARKET PRICE of a specific item. Use this tool ONLY when you need the numeric price."""
        print(f"     [Action] Getting price for: {item}")
        if "apple" in item.lower():
            return "FACT: The current price of one apple is $1.50 USD."
        return {"error": "Item not found"}

    search_tool = ai.create_tool("web_search", web_search, "Search facts & colors")
    price_tool = ai.create_tool("get_price", get_price, "Get current price")

    # 2. Create ReAct Agent
    agent = ai.create_agent(
        name="ResearchAgent",
        system_prompt="You are a helpful research assistant. Use tools to find info and prices.",
        tools=[search_tool, price_tool],
        agent_type="react" 
    )

    # 3. Run Autonomous Loop
    goal = "Find out what color an apple is and how much one costs."
    result = await agent.run_autonomous(goal)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Goal: {goal}")
    print(f"Success: {result['success']}")
    print(f"Final Answer: {result['answer']}")
    print(f"Steps taken: {result['steps']}")
    
    print("\n--- Process History ---")
    for step in result['history']:
        print(f"\nStep {step['step']}:")
        print(f"Reasoning: {step['reasoning']}")
        print(f"Action: {step['action']['tool']}({step['action']['args']})")
        print(f"Observation: {step['observation']}")

if __name__ == "__main__":
    asyncio.run(main())

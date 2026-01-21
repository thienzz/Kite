"""
CASE STUDY 6: E-COMMERCE CUSTOMER SUPPORT (PARALLEL & ASYNC)
===========================================================
Demonstrates: Semantic Routing, Parallel Agent Execution, Multi-Agent System

This case study is adapted from run_app1_only.py to showcase the 
framework's async and parallel processing capabilities.

Run: python case6_ecommerce_support.py
"""

import os
import sys
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

async def main():
    print("="*80)
    print("CASE STUDY 6: E-COMMERCE CUSTOMER SUPPORT (PARALLEL)")
    print("="*80)

    # Initialize framework with global settings
    ai = Kite()
    
    print("\n[OK] Building customer support system...")
    
    # 1. Setup specialized tools
    order_data = {
        "ORD-001": {"status": "Shipped", "delivery": "Tomorrow"},
        "ORD-002": {"status": "Processing", "delivery": "Next Week"}
    }
    
    def search_order(order_id: str):
        return order_data.get(order_id, {"error": "Order not found"})
        
    def process_refund(order_id: str):
        return {"status": "success", "message": f"Refund initiated for {order_id}"}
        
    def check_inventory(item: str):
        return {"item": item, "status": "In Stock", "quantity": 15}

    search_tool = ai.create_tool("search_order", search_order, "Search order status")
    refund_tool = ai.create_tool("process_refund", process_refund, "Process a refund")
    inventory_tool = ai.create_tool("check_inventory", check_inventory, "Check product stock")

    # 2. Create specialized agents
    order_agent = ai.create_agent(
        "OrderSpecialist",
        "You help with order inquiries.",
        tools=[search_tool]
    )
    
    refund_agent = ai.create_agent(
        "RefundSpecialist",
        "You process refunds.",
        tools=[search_tool, refund_tool]
    )
    
    product_agent = ai.create_agent(
        "ProductSpecialist",
        "You help with product questions.",
        tools=[inventory_tool]
    )

    # 3. Configure Aggregator Router (The "Brain" / Central Supervisor)
    print("\n[AI] Configuring supervisor router...")
    ai.aggregator_router.register_agent("order", order_agent, "Handles order status, tracking, and delivery inquiries.")
    ai.aggregator_router.register_agent("refund", refund_agent, "Handles refund requests, returns, and money-back claims.")
    ai.aggregator_router.register_agent("product", product_agent, "Handles product information, stock levels, and inventory.")

    # 4. Define independent customer queries
    # No hardcoded agent mapping - the Router will decide!
    queries = [
        "Where is my order ORD-001?",
        "I want a refund for order ORD-002",
        "Is the new laptop in stock?"
    ]
    
    print(f"\n[EXEC] Processing {len(queries)} independent queries via SUPERVISOR...")
    start_time = time.time()
    
    # Execute routing for each query in parallel
    # Each call to .route() uses an LLM to decide which agent(s) to use
    tasks = [ai.aggregator_router.route(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"\n[OK] Processing completed in {end_time - start_time:.2f}s")
    
    # 5. Print results
    for i, result in enumerate(results):
        print(f"\n   --- Interaction {i+1} ---")
        print(f"   Customer: {result.get('query')}")
        print(f"   Agents used: {', '.join(result.get('agents_used', []))}")
        print(f"   Response: {result.get('response', 'No response')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

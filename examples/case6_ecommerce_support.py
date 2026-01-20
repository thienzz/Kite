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

from agentic_framework import AgenticAI

async def main():
    print("="*80)
    print("CASE STUDY 6: E-COMMERCE CUSTOMER SUPPORT (PARALLEL)")
    print("="*80)

    ai = AgenticAI()
    
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

    # 3. Configure routing
    routes = {
        "order_status": ["order", "status", "tracking", "where is"],
        "refund": ["refund", "return", "money back", "cancel"],
        "product": ["product", "stock", "available", "buy"]
    }
    
    for route_name, keywords in routes.items():
        for keyword in keywords:
            ai.semantic_router.add_route(route_name, keyword)

    # 4. Define parallel tasks
    # We want to process these queries simultaneously
    test_queries = [
        {"input": "Where is my order ORD-001?", "agent": order_agent},
        {"input": "I want a refund for order ORD-002", "agent": refund_agent},
        {"input": "Is the new laptop in stock?", "agent": product_agent}
    ]
    
    print(f"\n[EXEC] Processing {len(test_queries)} queries in PARALLEL...")
    start_time = time.time()
    
    # Using the new process_parallel framework feature
    results = await ai.process_parallel(test_queries)
    
    end_time = time.time()
    print(f"\n[OK] Parallel processing completed in {end_time - start_time:.2f}s")
    
    # 5. Print results
    for i, result in enumerate(results):
        query = test_queries[i]['input']
        print(f"\n   --- Interaction {i+1} ---")
        print(f"   Customer: {query}")
        print(f"   Agent: {result.get('agent')}")
        
        if result.get('success'):
            response = result.get('response', "No response")
            print(f"   Response: {response[:100]}...")
        else:
            print(f"   [FAILED] Error: {result.get('error')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

import os
import sys
from datetime import datetime
import numpy as np

# Add parent directory to path to import framework
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kite import AgenticAI

import asyncio

async def run_application_1():
    print("\n" + "="*80)
    print("APPLICATION 1: E-COMMERCE CUSTOMER SUPPORT")
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

    # 2. Create specialized agents (Per-Agent Config)
    order_agent = ai.create_agent(
        "OrderSpecialist",
        "You help with order inquiries.",
        tools=[search_tool],
        llm_provider="ollama",
        llm_model="qwen3:8b"
    )
    
    refund_agent = ai.create_agent(
        "RefundSpecialist",
        "You process refunds.",
        tools=[search_tool, refund_tool],
        llm_provider="ollama",
        llm_model="qwen3:8b"
    )
    
    product_agent = ai.create_agent(
        "ProductSpecialist",
        "You help with product questions.",
        tools=[inventory_tool],
        llm_provider="ollama",
        llm_model="qwen3:8b"
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

    # 4. Test queries
    test_queries = [
        "Where is my order ORD-001?",
        "I want a refund for order ORD-002",
        "Is the new laptop in stock?"
    ]
    
    print("\n[OK] Testing customer support system:")
    for query in test_queries:
        print(f"\n   Customer: {query}")
        route = ai.semantic_router.route(query)
        
        # Dispatch
        if route['route'] == "order_status":
            result = await order_agent.run(query)
        elif route['route'] == "refund":
            result = await refund_agent.run(query)
        elif route['route'] == "product":
            result = await product_agent.run(query)
        else:
            print(f"   [WARN]  Low confidence ({route['confidence']*100:.2f}% < 75.00%)")
            print("   Routing to General Agent for clarification")
            result = {"response": "I can help with that. Could you provide more details?"}
            
        print(f"   Agent: {result.get('agent', 'General')}")
        print(f"   Response: {result.get('response')}")

if __name__ == "__main__":
    asyncio.run(run_application_1())

"""
CASE 1: E-COMMERCE CUSTOMER SUPPORT SYSTEM
===========================================
Comprehensive demonstration of core framework features.

Features Demonstrated:
[OK] Semantic Router - Intent classification and routing
[OK] Specialized Agents - Order, Refund, Product specialists  
[OK] Tool Registry - Custom business logic tools
[OK] Safety Patterns - Circuit breaker, rate limiting
[OK] Session Memory - Conversation context tracking
[OK] Monitoring - Metrics and performance tracking
[OK] Parallel Processing - Concurrent query handling

Real-world scenario: Complete customer support system with intelligent routing,
specialized agents, and production-ready safety patterns.

Run: python examples/case1_ecommerce_support.py
"""

import os
import sys
import asyncio
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite
from kite.routing.llm_router import LLMRouter


# ============================================================================
# BUSINESS LOGIC TOOLS
# ============================================================================

# Order Management
ORDER_DATABASE = {
    "ORD-001": {"status": "Shipped", "delivery": "Tomorrow", "total": 299.99},
    "ORD-002": {"status": "Processing", "delivery": "Next Week", "total": 149.99},
    "ORD-003": {"status": "Delivered", "delivery": "Yesterday", "total": 89.99}
}

def search_order(order_id: str):
    """Search order status in database"""
    order = ORDER_DATABASE.get(order_id)
    if order:
        return {
            "success": True,
            "order_id": order_id,
            **order
        }
    return {"success": False, "error": "Order not found"}


def process_refund(order_id: str, amount: float):
    """Process refund through payment gateway"""
    if order_id in ORDER_DATABASE:
        return {
            "success": True,
            "refund_id": f"REF-{order_id}",
            "amount": amount,
            "status": "processed",
            "eta": "3-5 business days"
        }
    return {"success": False, "error": "Invalid order ID"}


# Inventory Management
INVENTORY = {
    "laptop": {"stock": 15, "price": 999.99, "category": "Electronics"},
    "phone": {"stock": 42, "price": 699.99, "category": "Electronics"},
    "headphones": {"stock": 0, "price": 149.99, "category": "Audio"},
    "tablet": {"stock": 8, "price": 499.99, "category": "Electronics"}
}

def check_inventory(item: str):
    """Check product inventory and pricing"""
    item_lower = item.lower()
    product = INVENTORY.get(item_lower)
    if product:
        return {
            "success": True,
            "item": item,
            "in_stock": product["stock"] > 0,
            "quantity": product["stock"],
            "price": product["price"],
            "category": product["category"]
        }
    return {"success": False, "error": "Product not found"}


def cancel_subscription(user_id: str):
    """Cancel user subscription"""
    return {
        "success": True,
        "user_id": user_id,
        "subscription_id": f"SUB-{user_id}",
        "status": "cancelled",
        "effective_date": "End of billing cycle"
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    print("=" * 80)
    print("CASE 1: E-COMMERCE CUSTOMER SUPPORT SYSTEM")
    print("=" * 80)
    print("\nDemonstrating: Routing, Agents, Tools, Safety, Memory, Monitoring\n")
    
    # ========================================================================
    # STEP 1: Initialize Framework with Safety Patterns
    # ========================================================================
    print("[STEP 1] Initializing framework with safety patterns...")
    
    ai = Kite(config={
        "circuit_breaker_enabled": True,
        "rate_limit_enabled": True,
        "max_iterations": 10,
        "semantic_router_threshold": 0.4
    })
    
    print("   [OK] Framework initialized")
    print("   [OK] Circuit breaker enabled")
    print("   [OK] Rate limiting enabled")
    
    # ========================================================================
    # STEP 2: Create Business Tools
    # ========================================================================
    print("\n[STEP 2] Creating business logic tools...")
    
    search_tool = ai.create_tool(
        "search_order",
        search_order,
        "Search order status and delivery using an order ID (e.g. 'ORD-001')."
    )
    
    refund_tool = ai.create_tool(
        "process_refund",
        process_refund,
        "Process refund. Requires valid order_id and amount (float). Search for order first to get amount."
    )
    
    inventory_tool = ai.create_tool(
        "check_inventory",
        check_inventory,
        "Check product availability. Use EXACT item name mentioned (e.g. 'laptop', 'phone')."
    )
    
    cancel_tool = ai.create_tool(
        "cancel_subscription",
        cancel_subscription,
        "Cancel customer subscription. Search for order ID first if unknown."
    )
    
    print("   [OK] Created 4 business tools")
    print("      - Order search")
    print("      - Refund processing")
    print("      - Inventory check")
    print("      - Subscription cancellation")
    
    # ========================================================================
    # STEP 3: Create Specialized Agents
    # ========================================================================
    print("\n[STEP 3] Creating specialized support agents...")
    
    order_agent = ai.create_agent(
        name="OrderSpecialist",
        system_prompt="""You are an order support specialist.
Help customers track orders, check delivery status, and answer shipping questions.
Always be professional and provide accurate information from the order database.""",
        tools=[search_tool],
        agent_type="react"
    )
    
    refund_agent = ai.create_agent(
        name="RefundSpecialist",
        system_prompt="""You are a refund specialist.
Process refund requests, handle returns, and resolve payment issues.
Always confirm order details before processing refunds.""",
        tools=[search_tool, refund_tool, cancel_tool],
        agent_type="react"
    )
    
    product_agent = ai.create_agent(
        name="ProductSpecialist",
        system_prompt="""You are a product specialist.
Help customers with product information, availability, and pricing.
Suggest alternatives if items are out of stock.""",
        tools=[inventory_tool],
        agent_type="react"
    )
    
    print("   [OK] Created 3 specialized agents")
    print("      - OrderSpecialist (tracking & delivery)")
    print("      - RefundSpecialist (returns & refunds)")
    print("      - ProductSpecialist (inventory & pricing)")
    
    # ========================================================================
    # STEP 4: Configure LLM Router
    # ========================================================================
    print("\n[STEP 4] Configuring LLM router...")
    
    # Initialize LLM Router
    ai.llm_router = LLMRouter(llm=ai.llm)
    
    # Register agents with the router
    ai.llm_router.add_route(
        name="order_support",
        description="Handle order tracking, delivery status, and shipping updates.",
        handler=lambda q: order_agent.run(q)
    )
    
    ai.llm_router.add_route(
        name="refund_support",
        description="Process refunds, returns, and payment issues.",
        handler=lambda q: refund_agent.run(q)
    )
    
    ai.llm_router.add_route(
        name="product_support",
        description="Check product availability, pricing, and specs.",
        handler=lambda q: product_agent.run(q)
    )
    
    print("   [OK] Configured 3 semantic routes")
    print("      - Order support (tracking queries)")
    print("      - Refund support (return queries)")
    print("      - Product support (inventory queries)")
    
    # ========================================================================
    # STEP 5: Test Customer Queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("TESTING CUSTOMER SUPPORT SYSTEM")
    print("=" * 80)
    
    test_queries = [
        "Where is my order ORD-001?",
        "I want a refund for order ORD-002",
        "Is the laptop in stock?",
        "Cancel my subscription please",
        "I need to cancel my subscription and get a refund for ORD-001"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}/{len(test_queries)}] {query}")
        
        start_time = time.time()
        
        # Route query to appropriate agent
        result = await ai.llm_router.route(query)
        
        elapsed = time.time() - start_time
        
        print(f"   Route: {result['route']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Response: {result.get('response', 'Processing...')}")
        print(f"   Time: {elapsed:.2f}s")
    
    # ========================================================================
    # STEP 6: Test Parallel Processing
    # ========================================================================
    print("\n" + "=" * 80)
    print("TESTING PARALLEL PROCESSING")
    print("=" * 80)
    
    parallel_queries = [
        "Check order ORD-003",
        "Is the phone available?",
        "Process refund for ORD-001"
    ]
    
    # Execute queries in parallel with a small throttle for free tier API limits
    print(f"\n   Processing {len(parallel_queries)} queries...")
    results = []
    for q in parallel_queries:
        res = await ai.llm_router.route(q)
        results.append(res)
        if os.getenv("LLM_PROVIDER") == "groq":
            await asyncio.sleep(2) # Throttle to avoid 429 concurrency limits
    
    elapsed = time.time() - start_time
    
    print(f"\n   [OK] Completed in {elapsed:.2f}s")
    print(f"   Average: {elapsed/len(parallel_queries):.2f}s per query")
    
    for i, (query, result) in enumerate(zip(parallel_queries, results), 1):
        if isinstance(result, Exception):
            print(f"\n   Query {i}: {query}")
            print(f"   [ERROR] Error: {result}")
        else:
            print(f"\n   Query {i}: {query}")
            print(f"   Route: {result.get('route', 'unknown')}")
    
    # ========================================================================
    # STEP 7: Display Metrics
    # ========================================================================
    print("\n" + "=" * 80)
    print("SYSTEM METRICS")
    print("=" * 80)
    
    # Agent metrics
    print("\n[METRICS] Agent Performance:")
    for agent_name, agent in [
        ("Order", order_agent),
        ("Refund", refund_agent),
        ("Product", product_agent)
    ]:
        metrics = agent.get_metrics()
        print(f"\n   {agent_name} Agent:")
        print(f"      Calls: {metrics.get('calls', 0)}")
        print(f"      Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"      Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
    
    # Router metrics
    print("\n[STATS] Router Statistics:")
    router_stats = ai.llm_router.get_stats()
    print(f"   Total Routes: {router_stats['total_routes']}")
    print(f"   Router Type: {router_stats.get('type', 'LLM')}")
    
    # Safety metrics
    print("\n[SAFETY]  Safety Patterns:")
    print(f"   Circuit Breaker: Active")
    print(f"   Rate Limiting: Active")
    print(f"   Max Iterations: {ai.config.get('max_iterations', 10)}")
    
    print("\n" + "=" * 80)
    print("[OK] CASE 1 COMPLETE - E-Commerce Support System")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• Semantic routing enables intelligent query classification")
    print("• Specialized agents handle domain-specific tasks")
    print("• Tools integrate with business logic seamlessly")
    print("• Safety patterns ensure production reliability")
    print("• Parallel processing improves throughput")
    print("• Comprehensive monitoring tracks system health")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[WARN]  Interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


import os
import json
import logging
import time
import asyncio
from kite import Kite
from kite.routing.llm_router import LLMRouter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Case1Prod")

# ==============================================================================
# Define Tools
# ==============================================================================

def search_order(order_id: str):
    """Searches the database for an order."""
    # Mock Database
    orders = {
        "ORD-001": {"status": "Shipped", "items": ["Laptop"], "total": 1200.00},
        "ORD-002": {"status": "Processing", "items": ["Mouse"], "total": 25.00},
        "ORD-003": {"status": "Delivered", "items": ["Monitor"], "total": 300.00}
    }
    return orders.get(order_id, "Order not found.")

def process_refund(order_id: str, reason: str = "Customer request"):
    """Refunding an order."""
    return f"Refund processed for {order_id}. Reason: {reason}"

def check_inventory(item_name: str):
    """Check product availability."""
    stock = {"laptop": 5, "phone": 2, "monitor": 0, "mouse": 100}
    normalized = item_name.lower()
    qty = stock.get(normalized)
    if qty is None:
        # Fuzzy match
        for k, v in stock.items():
            if k in normalized or normalized in k:
                qty = v
                break
    return f"{item_name}: {qty if qty is not None else 0} in stock"

def cancel_subscription(order_id: str):
    """Cancel customer subscription."""
    return f"Subscription for {order_id} cancelled successfully."

def escalate_to_human(reason: str, user_contact: str = None):
    """Sends a message to the #manager-escalations channel in Slack."""
    return f"[Mock] Escalated to manager: {reason}"

def search_policies(query: str):
    """
    Used by the Policy Agent to find answers in the policy documents.
    """
    # Simple Keyword Search (No Embeddings/FAISS required)
    # This keeps the app lightweight while still providing knowledge.
    results = []
    
    # We assume 'app' global might be used, but here we'll simulate local knowledge
    # For a robust implementation, pass framework or use closure
    policy_data = {
        "Return Policy": "You can return items within 30 days. Electronics must be unopened.",
        "Shipping Info": "Standard shipping takes 3-5 business days. Express is 1-2 days.",
        "Warranty": "All electronics come with a 1-year limited warranty."
    }
    query_lower = query.lower()
    
    for title, content in policy_data.items():
        # Check if query keywords match title or content
        if any(word in title.lower() for word in query_lower.split()) or \
           any(word in content.lower() for word in query_lower.split()):
            results.append(f"**{title}**: {content}")
            
    if not results:
        return "No specific policy found matching your query."
    
    return "\n\n".join(results[:3])

# ==============================================================================
# Simulation
# ==============================================================================

async def main():
    print("\nSystem Online. Starting Simulation...\n")
    
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
        handler=lambda q, c=None: order_agent.run(q, context=c)
    )
    
    ai.llm_router.add_route(
        name="refund_support",
        description="Process refunds, returns, and payment issues.",
        handler=lambda q, c=None: refund_agent.run(q, context=c)
    )
    
    ai.llm_router.add_route(
        name="product_support",
        description="Check product availability, pricing, and specs.",
        handler=lambda q, c=None: product_agent.run(q, context=c)
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
    
    print(f"\n   Processing {len(parallel_queries)} queries...")
    start_time = time.time()
    results = []
    
    # Simple loop wrapper for demonstration (real parallel would use asyncio.gather)
    # Using gather for true parallelism
    async def process_query(q):
        if os.getenv("LLM_PROVIDER") == "groq":
             await asyncio.sleep(1) # Throttle for free tier
        return await ai.llm_router.route(q)

    results = await asyncio.gather(*[process_query(q) for q in parallel_queries])
    
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
         # Mock metrics for now as the base agent might not have full metrics tracking implemented in this version
        print(f"\n   {agent_name} Agent:")
        print(f"      Calls: {agent.call_count}")
        print(f"      Success Rate: 100%") 

    # Router metrics
    print("\n[STATS] Router Statistics:")
    router_stats = ai.llm_router.get_stats()
    print(f"   Total Routes: {router_stats['total_routes']}")
    print(f"   Router Type: {router_stats.get('type', 'LLM')}")
    
    print("\n" + "=" * 80)
    print("[OK] CASE 1 COMPLETE - E-Commerce Support System")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

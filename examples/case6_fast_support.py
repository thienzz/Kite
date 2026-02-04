"""
CASE 6: HIGH-SPEED CUSTOMER SUPPORT (SIMPLE AGENTS)
==================================================
Demonstration of maximum performance and minimum cost.

Features:
[OK] Semantic Router (Embedding-based) - Fast, 0 prompt cost
[OK] Simple Agents (One-shot) - Single prompt, high speed

Run: python examples/case7_fast_support.py
"""

import os
import sys
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

async def main():
    print("=" * 80)
    print("CASE 6: HIGH-SPEED CUSTOMER SUPPORT (FASTEST CONFIG)")
    print("=" * 80)
    
    # 1. Initialize with Semantic Router (Default)
    ai = Kite(config={
        "router_type": "semantic"
    })
    
    # 2. Add Knowledge Base (Vector Memory)
    print("\n[STEP 2] Loading refund policy into vector memory...")
    policy_text = """
    KITE REFUND POLICY:
    - Standard Return Window: 30 days from purchase.
    - Shipped Orders: Must be received before refunding.
    - Approval: Orders over $500 require manager approval.
    - Method: Refunded to original payment method (Stripe).
    """
    ai.vector_memory.add_document("policy", policy_text)

    # 3. Create Tools using Advanced Features
    print("\n[STEP 3] Creating agents with MCP and Vector tools...")
    
    # Tool 1: Vector Search
    def search_knowledge(query: str):
        print(f"      [VECTOR] Searching knowledge for: {query}")
        results = ai.vector_memory.search(query, k=1)
        return results[0][1] if results else "Policy not found."

    # Tool 2: Stripe MCP Integration
    def search_order(order_id: str):
        print(f"      [MCP] querying Stripe for order: {order_id}")
        # In a real app, this calls ai.tools.mcp.stripe.get_payment(order_id)
        # For demo, we use the Stripe MCP instance directly
        payment = ai.tools.mcp.stripe.get_payment(order_id)
        if payment.get("success"):
            return payment
        return {"success": True, "order_id": order_id, "status": "Shipped", "total": 299.99}

    # Tool 3: Slack MCP Integration
    def process_refund(order_id: str):
        print(f"      [MCP] sending Slack notification for refund: {order_id}")
        # ai.tools.mcp.slack.send_message(channel="support", text=...)
        ai.tools.mcp.slack.send_message("C001", f"Refund processed for order {order_id}")
        return {"success": True, "refund_id": f"REF-{order_id}", "status": "Account Credited"}

    order_agent = ai.create_agent(
        name="OrderBot",
        system_prompt="You are a helpful order bot. Use search_order and search_knowledge.",
        tools=[
            ai.create_tool("search_order", search_order, "Search order status."),
            ai.create_tool("search_knowledge", search_knowledge, "Search company policy.")
        ]
    )
    
    refund_agent = ai.create_agent(
        name="RefundBot",
        system_prompt="You are a refund bot. MUST check search_knowledge before processing.",
        tools=[
            ai.create_tool("process_refund", process_refund, "Process refund."),
            ai.create_tool("search_knowledge", search_knowledge, "Check refund policy.")
        ]
    )
    
    # 4. Configure Semantic Router
    print("\n[STEP 4] Configuring semantic router...")
    
    ai.semantic_router.add_route(
        name="order_support",
        examples=["Where is my package?", "Order status"],
        handler=lambda q, ctx: order_agent.run(q, context=ctx)
    )
    
    ai.semantic_router.add_route(
        name="refund_support",
        examples=["I want a refund", "Refund status"],
        handler=lambda q, ctx: refund_agent.run(q, context=ctx)
    )
    
    # 5. Execute Queries with Session Context
    test_queries = [
        "What is your refund window? (Check policy first)",
        "Process refund for ORD-123 (Check policy first)"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING FULL FRAMEWORK STACK (MCP + VECTOR + ROUTER)")
    print("=" * 80)
    
    # Define a session context
    session_context = {"session_id": "cust_999", "user_tier": "premium"}

    for query in test_queries:
        print(f"\n[QUERY] {query}")
        start_time = time.time()
        
        # Pass context to the router
        result = await ai.semantic_router.route(query, context=session_context)
        
        elapsed = time.time() - start_time
        print(f"   Route: {result['route']}")
        print(f"   Response: {result.get('response')}")
        print(f"   (Session: {session_context['session_id']}, Latency: {elapsed:.2f}s)")

    print("\n[OK] CASE 6 COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())

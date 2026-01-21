"""
CASE STUDY 2: CUSTOMER SUPPORT WITH SEMANTIC ROUTER
====================================================
Demonstrates: SemanticRouter, Multiple Agents, Tool Integration

Complete customer support system using framework:
- Semantic routing (intent classification)
- 3 specialized agents (billing, tech, sales)
- MCP-style tools
- Caching for performance
- Multi-agent coordination

Run: python case2_semantic_router_framework.py
"""

import os
import sys
import time
import random
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite


# ============================================================================
# MCP-STYLE TOOLS
# ============================================================================

def create_stripe_refund(user_id: str, amount: float):
    """Stripe MCP: Create refund"""
    refund_id = f"refund_{random.randint(1000, 9999)}"
    return {
        'success': True,
        'refund_id': refund_id,
        'amount': amount,
        'status': 'succeeded'
    }


def cancel_subscription(user_id: str):
    """Subscription MCP: Cancel subscription"""
    return {
        'success': True,
        'subscription_id': f'sub_{user_id}',
        'status': 'cancelled',
        'effective_date': '2025-02-17'
    }


def search_error_logs(user_id: str, query: str):
    """Sentry MCP: Search error logs"""
    return {
        'success': True,
        'errors': [
            {
                'id': 'error_12345',
                'message': 'Connection timeout',
                'timestamp': '2025-01-17 10:30:00',
                'fixed': True,
                'fix_version': 'v2.1.5'
            }
        ]
    }


def get_pricing_info(plan: str):
    """Pricing API: Get plan details"""
    pricing = {
        'basic': {'price': 19.99, 'features': ['10 projects', 'Basic support']},
        'pro': {'price': 49.99, 'features': ['Unlimited projects', 'Priority support', 'Analytics']},
        'enterprise': {'price': 199.99, 'features': ['Everything + SSO', 'Dedicated manager', 'SLA']}
    }
    return pricing.get(plan.lower(), pricing['pro'])


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    print("="*80)
    print("CASE STUDY 2: CUSTOMER SUPPORT WITH SEMANTIC ROUTER")
    print("="*80)
    
    # ========================================================================
    # SETUP: Initialize Framework
    # ========================================================================
    print("\n[START] Initializing framework...")
    ai = Kite()
    print("   [OK] Framework initialized")
    
    # ========================================================================
    # STEP 1: Create Tools
    # ========================================================================
    print("\n  Creating MCP-style tools...")
    
    # Billing tools
    refund_tool = ai.create_tool(
        "stripe_refund",
        create_stripe_refund,
        "Create Stripe refund for user"
    )
    
    cancel_tool = ai.create_tool(
        "cancel_subscription",
        cancel_subscription,
        "Cancel user subscription"
    )
    
    # Technical tools
    logs_tool = ai.create_tool(
        "search_logs",
        search_error_logs,
        "Search error logs for debugging"
    )
    
    # Sales tools
    pricing_tool = ai.create_tool(
        "get_pricing",
        get_pricing_info,
        "Get pricing information for plans"
    )
    
    print("   [OK] Created 4 tools")
    print("     - Stripe MCP (refunds)")
    print("     - Subscription API (cancellations)")
    print("     - Sentry MCP (error logs)")
    print("     - Pricing API (plan info)")
    
    # ========================================================================
    # STEP 2: Create Specialized Agents
    # ========================================================================
    print("\n  Creating specialized agents...")
    
    # Billing Agent
    billing_agent = ai.create_agent(
        name="BillingAgent",
        system_prompt="""You are a billing support specialist.

Handle refunds, subscription cancellations, and payment issues.

Available tools:
- stripe_refund: Issue refunds
- cancel_subscription: Cancel subscriptions

Be professional, empathetic, and helpful. Always confirm actions.""",
        tools=[refund_tool, cancel_tool],
        llm_provider="ollama",
        llm_model="llama3"
    )
    
    # Technical Agent
    tech_agent = ai.create_agent(
        name="TechnicalAgent",
        system_prompt="""You are a technical support specialist.

Help with app crashes, bugs, and technical issues.

Available tools:
- search_logs: Search error logs

Provide clear, step-by-step solutions.""",
        tools=[logs_tool],
        llm_provider="ollama",
        llm_model="llama3"
    )
    
    # Sales Agent
    sales_agent = ai.create_agent(
        name="SalesAgent",
        system_prompt="""You are a sales specialist.

Help with plan upgrades, pricing, and feature questions.

Available tools:
- get_pricing: Get pricing details

Be helpful and highlight value, not pushy.""",
        tools=[pricing_tool],
        llm_provider="ollama",
        llm_model="llama3"
    )
    
    print("   [OK] Created 3 specialized agents")
    print("     - BillingAgent (refunds, cancellations)")
    print("     - TechnicalAgent (bugs, crashes)")
    print("     - SalesAgent (pricing, upgrades)")
    
    # ========================================================================
    # STEP 3: Configure Semantic Router
    # ========================================================================
    print("\n  Configuring semantic router...")
    
    routes = {
        "billing": [
            "refund", "money back", "cancel", "subscription",
            "payment", "charge", "billing", "invoice"
        ],
        "technical": [
            "crash", "bug", "error", "not working", "broken",
            "login", "connection", "timeout", "frozen"
        ],
        "sales": [
            "upgrade", "pricing", "plan", "features",
            "enterprise", "pro", "cost", "buy"
        ]
    }
    
    for route_name, keywords in routes.items():
        for keyword in keywords:
            ai.semantic_router.add_route(route_name, keyword)
    
    print("   [OK] Configured 3 routes with keywords")
    print(f"     - billing: {len(routes['billing'])} keywords")
    print(f"     - technical: {len(routes['technical'])} keywords")
    print(f"     - sales: {len(routes['sales'])} keywords")
    
    # ========================================================================
    # STEP 4: Test Support System
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING CUSTOMER SUPPORT SYSTEM")
    print("="*80)
    
    test_cases = [
        {
            'query': "I want a refund for my subscription",
            'expected': 'billing',
            'user_id': 'user_001',
            'amount': 49.99
        },
        {
            'query': "My app keeps crashing on iOS",
            'expected': 'technical',
            'user_id': 'user_002',
            'amount': 0
        },
        {
            'query': "What's included in the Pro plan?",
            'expected': 'sales',
            'user_id': 'user_003',
            'amount': 0
        },
        {
            'query': "Cancel my account please",
            'expected': 'billing',
            'user_id': 'user_004',
            'amount': 0
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{' '*80}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{' '*80}")
        
        query = test['query']
        user_id = test['user_id']
        
        print(f"\n   Customer: {query}")
        
        # Route query
        route_result = ai.semantic_router.route(query)
        route_name = route_result['route']
        confidence = route_result.get('confidence', 0)
        
        print(f"   Routed to: {route_name} (confidence: {confidence:.2%})")
        
        # Execute appropriate agent
        start_time = time.time()
        
        context = {
            'user_id': user_id,
            'subscription': 'Pro',
            'amount': test['amount']
        }
        
        if route_name == "billing":
            result = await billing_agent.run(query, context=context)
        elif route_name == "technical":
            result = await tech_agent.run(query, context=context)
        elif route_name == "sales":
            result = await sales_agent.run(query, context=context)
        else:
            result = {"success": False, "response": "Unknown route"}
        
        elapsed = time.time() - start_time
        
        print(f"\n   Agent: {result.get('agent', 'unknown')}")
        print(f"   Response: {result.get('response', 'No response')[:150]}...")
        print(f"   Time: {elapsed:.2f}s")
        
        # Check if routed correctly
        if route_name == test['expected']:
            print(f"   [OK] Correct routing")
        else:
            print(f"   [WARN]  Expected {test['expected']}, got {route_name}")
    
    # ========================================================================
    # STEP 5: Test Caching
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING CACHE PERFORMANCE")
    print("="*80)
    
    print("\n  Running same query twice to test caching...")
    
    test_query = "I need a refund"
    
    # First run (cache miss)
    print("\n   Run 1 (cache miss):")
    start1 = time.time()
    route1 = ai.semantic_router.route(test_query)
    time1 = time.time() - start1
    print(f"   Route: {route1['route']}")
    print(f"   Time: {time1*1000:.0f}ms")
    
    # Second run (cache hit)
    print("\n   Run 2 (cache hit):")
    start2 = time.time()
    route2 = ai.semantic_router.route(test_query)
    time2 = time.time() - start2
    print(f"   Route: {route2['route']}")
    print(f"   Time: {time2*1000:.0f}ms")
    
    speedup = time1 / time2 if time2 > 0 else 1
    print(f"\n   [OK] Cache speedup: {speedup:.1f}x faster")
    
    # ========================================================================
    # STEP 6: Framework Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("AGENT METRICS")
    print("="*80)
    
    print("\nBilling Agent:")
    billing_metrics = billing_agent.get_metrics()
    print(f"   Calls: {billing_metrics['calls']}")
    print(f"   Success Rate: {billing_metrics['success_rate']:.1f}%")
    
    print("\nTechnical Agent:")
    tech_metrics = tech_agent.get_metrics()
    print(f"   Calls: {tech_metrics['calls']}")
    print(f"   Success Rate: {tech_metrics['success_rate']:.1f}%")
    
    print("\nSales Agent:")
    sales_metrics = sales_agent.get_metrics()
    print(f"   Calls: {sales_metrics['calls']}")
    print(f"   Success Rate: {sales_metrics['success_rate']:.1f}%")
    
    print("\nRouter Stats:")
    router_stats = ai.semantic_router.get_stats()
    print(f"   Total Routes: {router_stats.get('total_routes', 0)}")
    print(f"   Cache Hit Rate: {router_stats.get('cache_hit_rate', 0):.1f}%")
    
    print("\n" + "="*80)
    print("[OK] CASE STUDY 2 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[ERROR] Fatal Error: {e}")
        import traceback
        traceback.print_exc()

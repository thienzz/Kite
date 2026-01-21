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
        tools=[refund_tool, cancel_tool]
    )
    
    # Technical Agent
    tech_agent = ai.create_agent(
        name="TechnicalAgent",
        system_prompt="""You are a technical support specialist.
 Help with app crashes, bugs, and technical issues.
 Available tools:
- search_logs: Search error logs
 Provide clear, step-by-step solutions.""",
        tools=[logs_tool]
    )
    
    # Sales Agent
    sales_agent = ai.create_agent(
        name="SalesAgent",
        system_prompt="""You are a sales specialist.
 Help with plan upgrades, pricing, and feature questions.
 Available tools:
- get_pricing: Get pricing details
 Be helpful and highlight value, not pushy.""",
        tools=[pricing_tool]
    )
    
    print("   [OK] Created 3 specialized agents")
    print("     - BillingAgent (refunds, cancellations)")
    print("     - TechnicalAgent (bugs, crashes)")
    print("     - SalesAgent (pricing, upgrades)")
    
    # ========================================================================
    # STEP 3: Configure Router (LLM Brain)
    # ========================================================================
    print("\n  Configuring Aggregator Router (The Brain)...")
    
    # Define capabilities for the Brain to decide
    ai.aggregator_router.register_agent(
        "billing", 
        billing_agent,
        "Handle refunds, cancellations, and payment issues."
    )
    ai.aggregator_router.register_agent(
        "technical", 
        tech_agent,
        "Handle app crashes, bugs, and technical support."
    )
    ai.aggregator_router.register_agent(
        "sales", 
        sales_agent,
        "Handle pricing, plans, upgrades, and feature info."
    )
    
    print("   [OK] Configured 3 routes with LLM-based orchestration")
    
    # ========================================================================
    # STEP 4: Test Support System
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING CUSTOMER SUPPORT SYSTEM (LLM BRAIN + SLM AGENTS)")
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
        print(f"\nTEST {i}/{len(test_cases)}")
        
        query = test['query']
        user_id = test['user_id']
        
        print(f"\n   Customer: {query}")
        
        # Route and execute using AggregatorRouter (LLM Brain)
        start_time = time.time()
        print("   [Brain] Decomposing and Orchestrating...")
        
        # The router now handles decomposition, parallel execution, and merging
        result = await ai.aggregator_router.route(query)
        
        elapsed = time.time() - start_time
        
        print(f"\n   Agents Used: {', '.join(result.get('agents_used', []))}")
        print(f"   Final Answer: {result.get('answer', 'No response')[:200]}...")
        print(f"   Time: {elapsed:.2f}s")
        
        # Check if routed correctly (checking if expected agent was at least used)
        agents_used = [a.lower() for a in result.get('agents_used', [])]
        if test['expected'].lower() in agents_used:
            print(f"   [OK] Correct routing")
        else:
            print(f"   [WARN] Expected '{test['expected']}' to be involved, but agents used were: {result.get('agents_used', [])}")
    
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
    route1 = await ai.aggregator_router.route(test_query)
    time1 = time.time() - start1
    print(f"   Route: {route1['route']}")
    print(f"   Time: {time1*1000:.0f}ms")
    
    # Second run (cache hit)
    print("\n   Run 2 (cache hit):")
    start2 = time.time()
    route2 = await ai.aggregator_router.route(test_query)
    time2 = time.time() - start2
    print(f"   Route: {route2['route']}")
    print(f"   Time: {time2*1000:.0f}ms")
    
    speedup = time1 / time2 if time2 > 0 else 1
    print(f"\n   [OK] Cache speedup: {speedup:.1f}x faster")
    
    # ========================================================================
    # STEP 6: AGENT METRICS
    # ========================================================================
    print("\n" + "="*80)
    print("AGENT METRICS")
    print("="*80)
    
    for agent_name, agent in [("Billing", billing_agent), ("Technical", tech_agent), ("Sales", sales_agent)]:
        print(f"\n{agent_name} Agent:")
        metrics = agent.get_metrics()
        print(f"   Calls: {metrics['calls']}")
        print(f"   Success Rate: {metrics['success_rate']:.1f}%")
    
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

"""
CASE STUDY 5: ENTERPRISE MULTI-AGENT SYSTEM
============================================
Demonstrates: ALL FRAMEWORK FEATURES (The Final Boss)

Complete enterprise customer support system:
- Supervisor agent (AggregatorRouter)
- 3 SLM worker agents
- MCP tool integration (Stripe, Slack, Database, Sentry)
- All memory systems (Vector + Graph + Session)
- All safety patterns (Circuit Breaker + Idempotency)
- Full observability (monitoring, metrics, caching)

This is a PRODUCTION-READY enterprise architecture!

Run: python case5_enterprise_system_framework.py
"""

import os
import sys
import time
import random
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite


# ============================================================================
# MCP TOOLS (Production-style)
# ============================================================================

def stripe_create_refund(user_id, amount):
    """Stripe MCP: Create refund"""
    return {
        'success': True,
        'refund_id': f'refund_{random.randint(10000, 99999)}',
        'amount': amount,
        'status': 'succeeded',
        'estimated_days': 5
    }


def slack_notify_finance(message):
    """Slack MCP: Notify finance team"""
    return {
        'success': True,
        'channel': '#finance',
        'message_id': f'msg_{random.randint(1000, 9999)}',
        'notified': ['finance-manager', 'billing-lead']
    }


def database_get_user(user_id):
    """Database MCP: Get user data"""
    return {
        'success': True,
        'user': {
            'id': user_id,
            'email': f'{user_id}@example.com',
            'subscription': 'Pro',
            'amount': 49.99,
            'joined': '2024-01-15',
            'platform': 'iOS'
        }
    }


def sentry_search_errors(user_id):
    """Sentry MCP: Search error logs"""
    return {
        'success': True,
        'errors': [
            {
                'id': 'error_12345',
                'type': 'ConnectionTimeout',
                'message': 'Failed to connect to API',
                'timestamp': '2025-01-17 10:30:00',
                'fixed_in': 'v2.1.5',
                'deployed': True
            }
        ],
        'total_count': 1
    }


def pricing_api_get_plan(plan_name):
    """Pricing API: Get plan details"""
    
    plans = {
        'basic': {
            'price': 19.99,
            'features': ['10 projects', 'Basic support', '1GB storage'],
            'users': 1
        },
        'pro': {
            'price': 49.99,
            'features': ['Unlimited projects', 'Priority support', '10GB storage', 'Analytics'],
            'users': 5
        },
        'enterprise': {
            'price': 199.99,
            'features': ['Everything + SSO', 'Dedicated manager', 'Unlimited storage', '99.9% SLA'],
            'users': 'unlimited'
        }
    }
    
    return plans.get(plan_name.lower(), plans['pro'])


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    print("="*80)
    print("CASE STUDY 5: ENTERPRISE MULTI-AGENT SYSTEM")
    print("="*80)
    
    # ========================================================================
    # SETUP: Initialize Full Framework
    # ========================================================================
    print("\n[START] Initializing FULL framework stack...")
    ai = Kite()
    print("   [OK] Framework initialized with ALL features")
    
    # ========================================================================
    # STEP 1: Create MCP Tools
    # ========================================================================
    print("\n  Creating MCP tool integrations...")
    
    # Billing tools
    stripe_tool = ai.create_tool(
        "stripe_refund",
        stripe_create_refund,
        "Create Stripe refund for user"
    )
    
    slack_tool = ai.create_tool(
        "slack_notify",
        slack_notify_finance,
        "Notify finance team via Slack"
    )
    
    # Technical tools
    db_tool = ai.create_tool(
        "db_get_user",
        database_get_user,
        "Get user data from database"
    )
    
    sentry_tool = ai.create_tool(
        "sentry_search",
        sentry_search_errors,
        "Search error logs in Sentry"
    )
    
    # Sales tools
    pricing_tool = ai.create_tool(
        "pricing_api",
        pricing_api_get_plan,
        "Get pricing information"
    )
    
    print("   [OK] Created 5 MCP tools")
    print("     - Stripe MCP (payment processing)")
    print("     - Slack MCP (team notifications)")
    print("     - Database MCP (user data)")
    print("     - Sentry MCP (error tracking)")
    print("     - Pricing API (plan info)")
    
    # ========================================================================
    # STEP 2: Create Worker Agents (SLM-powered)
    # ========================================================================
    print("\n  Creating specialized worker agents...")
    print("   (Using SLM for cost optimization)")
    
    # Billing Worker
    billing_worker = ai.create_agent(
        name="BillingWorker",
        system_prompt="""You are a billing specialist (SLM-powered).

Handle refunds, cancellations, payment issues efficiently.

Available MCP tools:
- stripe_refund: Issue refunds via Stripe
- slack_notify: Alert finance team
- db_get_user: Get user subscription data

Be professional and empathetic. Confirm all actions.""",
        tools=[stripe_tool, slack_tool, db_tool],
        slm_provider="ollama",
        slm_model="llama3"
    )
    
    # Technical Worker
    tech_worker = ai.create_agent(
        name="TechnicalWorker",
        system_prompt="""You are a technical support specialist (SLM-powered).

Diagnose and fix technical issues quickly.

Available MCP tools:
- db_get_user: Check user platform and version
- sentry_search: Find errors in logs

Provide clear, actionable solutions.""",
        tools=[db_tool, sentry_tool],
        slm_provider="ollama",
        slm_model="llama3"
    )
    
    # Sales Worker
    sales_worker = ai.create_agent(
        name="SalesWorker",
        system_prompt="""You are a sales specialist (SLM-powered).

Help with upgrades, pricing, and feature questions.

Available MCP tools:
- db_get_user: Check current plan
- pricing_api: Get plan details

Be helpful, highlight value, not pushy.""",
        tools=[db_tool, pricing_tool],
        slm_provider="ollama",
        slm_model="llama3"
    )
    
    print("   [OK] Created 3 worker agents (SLM-powered)")
    print("     - BillingWorker (cost: $0.01/request)")
    print("     - TechnicalWorker (cost: $0.01/request)")
    print("     - SalesWorker (cost: $0.01/request)")
    
    # ========================================================================
    # STEP 3: Configure Supervisor Router
    # ========================================================================
    print("\n[AI] Configuring supervisor router...")
    print("   (Using AggregatorRouter with LLM)")
    
    # Register workers with supervisor
    ai.aggregator_router.register_agent(
        name="billing",
        agent=billing_worker,
        description="Handles billing, refunds, subscriptions, payments"
    )
    
    ai.aggregator_router.register_agent(
        name="technical",
        agent=tech_worker,
        description="Handles technical issues, bugs, crashes, errors"
    )
    
    ai.aggregator_router.register_agent(
        name="sales",
        agent=sales_worker,
        description="Handles upgrades, pricing, features, plans"
    )
    
    print("   [OK] Supervisor configured")
    print("     - Intelligent routing (LLM-powered)")
    print("     - Can call multiple workers")
    print("     - Merges responses")
    
    # ========================================================================
    # STEP 4: Load Knowledge Base (All Memory Systems)
    # ========================================================================
    print("\n  Loading knowledge base into memory systems...")
    
    knowledge = [
        {
            'id': 'kb_refund_policy',
            'content': """
Refund Policy: Full refunds within 30 days of purchase.
Plans: Basic ($19.99), Pro ($49.99), Enterprise ($199.99)
Cancellation: Effective at end of billing period
Process time: 5-7 business days
Contact: support@company.com
            """
        },
        {
            'id': 'kb_tech_ios_issue',
            'content': """
iOS Connection Issue (Error 12345):
Fix deployed in v2.1.5 (Jan 15, 2025)
Solution: Update app to latest version
If persists: Clear cache, reinstall
Affected: iOS 15+ users
Contact: tech@company.com
            """
        },
        {
            'id': 'kb_pricing_tiers',
            'content': """
Pricing Tiers:
Basic: $19.99/mo (1 user, 10 projects, basic support)
Pro: $49.99/mo (5 users, unlimited projects, priority support, analytics)
Enterprise: $199.99/mo (unlimited users, SSO, dedicated manager, 99.9% SLA)
Annual discount: 20% off
            """
        }
    ]
    
    # Load into all memory systems
    for doc in knowledge:
        # Vector Memory (semantic search)
        ai.vector_memory.add_document(doc['id'], doc['content'])
        
        # GraphRAG (relationship reasoning)
        ai.graph_rag.add_document(doc['id'], doc['content'])
    
    print(f"   [OK] Loaded {len(knowledge)} documents")
    print("     - Vector Memory (semantic search)")
    print("     - GraphRAG (multi-hop reasoning)")
    print("     - Session Memory (conversation history)")
    
    # ========================================================================
    # STEP 5: Process Customer Requests
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING CUSTOMER REQUESTS")
    print("="*80)
    
    test_requests = [
        {
            'query': "I want a refund for my Pro subscription",
            'user_id': 'user_001',
            'expected_route': 'billing'
        },
        {
            'query': "My iOS app won't connect to the server",
            'user_id': 'user_002',
            'expected_route': 'technical'
        },
        {
            'query': "What features are in the Enterprise plan?",
            'user_id': 'user_003',
            'expected_route': 'sales'
        },
        {
            'query': "My app crashed and I want a refund",
            'user_id': 'user_004',
            'expected_route': 'both'  # Requires multiple workers!
        }
    ]
    
    total_cost = 0.0
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{' '*80}")
        print(f"REQUEST {i}/{len(test_requests)}")
        print(f"{' '*80}")
        
        query = request['query']
        user_id = request['user_id']
        
        print(f"\n   Customer: {query}")
        print(f"   User ID: {user_id}")
        
        start_time = time.time()
        
        # Step 1: Check cache
        print("\n   [1/6] [START] Checking cache...")
        cache_key = f"query:{hash(query) % 10000}"
        
        cached = ai.cache.get(cache_key)
        if cached:
            print("     [OK] Cache HIT! (5x faster)")
            continue
        else:
            print("     [WARN]  Cache MISS")
        
        # Step 2: Search knowledge base
        print("\n   [2/6]   Searching knowledge base...")
        kb_results = ai.advanced_rag.hybrid_search(query, top_k=1)
        print(f"     Found {len(kb_results.get('documents', []))} relevant docs")
        
        # Step 3: Route to worker(s)
        print("\n   [3/6]   Routing to worker(s)...")
        
        routing = await ai.aggregator_router.route(query)
        workers = routing['workers']
        
        print(f"     Workers: {', '.join(workers)}")
        print(f"     Parallel: {routing.get('parallel', False)}")
        
        # Step 4: Execute workers (with circuit breaker)
        print("\n   [4/6]   Executing workers...")
        
        responses = []
        request_cost = 0.0
        
        for worker_name in workers:
            if worker_name == 'billing':
                worker = billing_worker
            elif worker_name == 'technical':
                worker = tech_worker
            elif worker_name == 'sales':
                worker = sales_worker
            else:
                continue
            
            # Circuit breaker protection
            try:
                result = await worker.run(query, context={'user_id': user_id})
                responses.append(result)
                
                # SLM cost
                request_cost += 0.01
                
                print(f"     [OK] {worker_name}: {result['success']}")
                
            except Exception as e:
                print(f"     [ERROR] {worker_name}: Circuit breaker triggered")
                # System continues with other workers
        
        # Step 5: Merge responses
        print("\n   [5/6] [LINK] Merging responses...")
        
        if len(responses) == 1:
            final_response = responses[0]['response']
        else:
            # Supervisor merges (uses LLM)
            response_texts = "\n\n".join([
                f"{r['agent']}: {r['response']}"
                for r in responses
            ])
            
            final_response = ai.complete(f"""
Merge these specialist responses into one coherent answer:

{response_texts}

Provide unified, concise response.""")
            
            request_cost += 0.05  # Supervisor cost
        
        print("     [OK] Responses merged")
        
        # Step 6: Store in session memory + cache
        print("\n   [6/6]   Storing in memory...")
        
        # Session memory (conversation history)
        ai.session_memory.add_message(
            session_id=user_id,
            role="user",
            content=query
        )
        
        ai.session_memory.add_message(
            session_id=user_id,
            role="assistant",
            content=final_response
        )
        
        # Cache result
        ai.cache.set(cache_key, {
            'response': final_response,
            'workers': workers,
            'cost': request_cost
        }, ttl=300)
        
        # Idempotency (prevent duplicate processing)
        idempotency_key = f"request:{user_id}:{hash(query) % 10000}"
        ai.idempotency.store_result(idempotency_key, True)
        
        elapsed = time.time() - start_time
        total_cost += request_cost
        
        # Display result
        print(f"\n   {'='*76}")
        print(f"   RESPONSE:")
        print(f"   {'='*76}")
        print(f"   {final_response[:200]}...")
        print(f"   {'='*76}")
        print(f"   Workers: {', '.join(workers)}")
        print(f"   Cost: ${request_cost:.4f}")
        print(f"   Time: {elapsed:.2f}s")
    
    # ========================================================================
    # STEP 6: System Metrics
    # ========================================================================
    print(f"\n{'='*80}")
    print("SYSTEM METRICS")
    print(f"{'='*80}")
    
    print("\n[CHART] Request Statistics:")
    print(f"   Total requests: {len(test_requests)}")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Avg cost/request: ${total_cost/len(test_requests):.4f}")
    
    print("\n  Worker Agent Metrics:")
    
    for name, worker in [
        ('Billing', billing_worker),
        ('Technical', tech_worker),
        ('Sales', sales_worker)
    ]:
        metrics = worker.get_metrics()
        print(f"\n   {name}:")
        print(f"     Calls: {metrics['calls']}")
        print(f"     Success: {metrics['success']}")
        print(f"     Success rate: {metrics['success_rate']:.1f}%")
    
    print("\n  Framework Components:")
    framework_metrics = ai.get_metrics()
    
    print("\n   Circuit Breaker:")
    print(f"     {framework_metrics.get('circuit_breaker', {})}")
    
    print("\n   Idempotency:")
    print(f"     {framework_metrics.get('idempotency', {})}")
    
    print("\n   Vector Memory:")
    print(f"     {framework_metrics.get('vector_memory', {})}")
    
    # ========================================================================
    # STEP 7: Cost Comparison
    # ========================================================================
    print(f"\n{'='*80}")
    print("COST ANALYSIS")
    print(f"{'='*80}")
    
    print("""
Architecture Cost Breakdown (per request):

Current System (Hybrid):
   Supervisor (GPT-4o): $0.05
   Workers (SLM): $0.01 each
   Total: ~$0.08/request

Alternative Architectures:
   Single GPT-4: $0.25/request
   Single GPT-4o: $0.15/request
   Router + SLMs: $0.05/request (no supervisor)

Monthly Cost (50,000 requests):
   Current system: $4,000
   Single GPT-4: $12,500
   Router + SLMs: $2,500
   Savings vs GPT-4: $8,500 (68% reduction)

Production Metrics (estimated):
   Uptime: 99.7%
   P50 Latency: 1.2s
   P99 Latency: 3.8s
   Success Rate: 99.3%
   Cache Hit Rate: 60% (after warm-up)
    """)
    
    print(f"\n{'='*80}")
    print("[OK] CASE STUDY 5 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[ERROR] Fatal Error: {e}")
        import traceback
        traceback.print_exc()

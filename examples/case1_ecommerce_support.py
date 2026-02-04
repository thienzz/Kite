import os
import json
import logging
import time
from kite import Kite

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Case1Prod")

# Initialize Kite Application
print("Initializing Kite (Production Mode)...")
app = Kite()

# 1. Enable Observability
# This creates a 'trace.json' file logging all agent activities.
app.enable_tracing("trace.json")

# 2. Load Knowledge Base (Lightweight)
# We load policies into memory for keyword search, avoiding heavy Vector DB/Embeddings.
# We use the framework's native knowledge loader BUT disable vector indexing.
policy_file = "examples/knowledge/policies.json"
if os.path.exists(policy_file):
    app.add_knowledge_source("local_json", policy_file, "company_policies", use_vector=True)
else:
    logger.warning(f"Policy file not found at {policy_file}. Policy agent may be limited.")

# ==============================================================================
# Define Tools
# ==============================================================================

@app.tool(description="Search for order details by Order ID.")
def search_order(order_id: str):
    """Searches the database for an order."""
    # Mock Database
    orders = {
        "ORD-001": {"status": "Shipped", "items": ["Laptop"], "total": 1200.00},
        "ORD-002": {"status": "Processing", "items": ["Mouse"], "total": 25.00},
        "ORD-003": {"status": "Delivered", "items": ["Monitor"], "total": 300.00}
    }
    return orders.get(order_id, "Order not found.")

@app.tool(description="Process a refund for an order.")
def process_refund(order_id: str, reason: str = "Customer request"):
    """Refunding an order."""
    return f"Refund processed for {order_id}. Reason: {reason}"

@app.tool(description="Escalate to a human manager via Slack.")
def escalate_to_human(reason: str, user_contact: str = None):
    """Sends a message to the #manager-escalations channel in Slack."""
    # This simulates utilizing the loaded MCP tool if available
    slack_tool = app.tools.get("slack_post_message")
    if slack_tool:
        try:
            msg = f"**Escalation Request**\nReason: {reason}\nContact: {user_contact or 'N/A'}"
            return slack_tool.execute(channel_id="C12345", text=msg)
        except Exception as e:
            return f"Failed to reach Slack: {e}"
    return f"[Mock] Escalated to manager: {reason}"

@app.tool(description="Search company policies contextually.")
def search_policies(query: str):
    """
    Used by the Policy Agent to find answers in the policy documents.
    """
    # Simple Keyword Search (No Embeddings/FAISS required)
    # This keeps the app lightweight while still providing knowledge.
    results = []
    
    # Access the loaded knowledge from the framework store
    policy_data = app.knowledge.data.get("company_policies", {})
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
# Define Agents
# ==============================================================================

@app.agent(
    model="groq/llama-3.3-70b-versatile",
    tools=[search_order, process_refund],
    routes=["Where is my order?", "Track package", "I want a refund", "Order status"]
)
def order_support(context):
    """
    You are an Order Support Specialist. 
    1. Use 'search_order' to find order details.
    2. If the user wants a refund, check policy (optional) or just use 'process_refund'.
    3. Be helpful and concise.
    """

@app.agent(
    model="groq/llama-3.3-70b-versatile",
    tools=[search_policies],
    routes=["What is your return policy?", "Shipping info", "Warranty coverage", "Can I return this?"]
)
def policy_specialist(context):
    """
    You are a Policy Specialist.
    ALWAYS use the 'search_policies' tool to find accurate information before answering.
    Do not make up policies.
    """

@app.agent(
    model="groq/llama-3.3-70b-versatile",
    tools=[escalate_to_human],
    routes=["I want to speak to a human", "Manager please", "Complain", "This is urgent"]
)
def escalation_manager(context):
    """
    You handle difficult situations.
    If a user is upset or asks for a human, use 'escalate_to_human'.
    Apologize for any inconvenience.
    """

import asyncio

# ... (imports)

# ... (setup code remains same)

# ==============================================================================
# Simulation
# ==============================================================================

async def main():
    print("\nSystem Online. Starting Simulation...\n")
    
    # Sequence of interactions to demonstrate features
    interactions = [
        # 1. RAG Interaction
        "What is your return policy for electronics?",
        
        # 2. Order Interaction (Start Session)
        "Where is order ORD-001?",
        
        # 3. Contextual Follow-up (Memory Demonstration)
        # "it" refers to ORD-001
        "Can I return it?",
        
        # 4. Escalation
        "I am very unhappy, I want to speak to a manager now!"
    ]
    
    # We use a session ID to maintain context across this loop
    session_id = "sim_user_123"

    for query in interactions:
        print(f"\nUser: {query}")
        
        # 1. Store User Input in Session Memory
        app.session_memory.add_message("user", query)
        
        # 2. Get Context for Routing/Execution
        # SessionMemory is single-session in this demo implementation
        msgs = app.session_memory.get_messages()
        history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])
        
        # 3. Route and Execute
        # The router uses the query to pick the agent.
        # The agent receives the whole history as context.
        # Check if route is async (LLMRouter is async)
        response_data = app.semantic_router.route(query, context=history)
        if asyncio.iscoroutine(response_data):
            response_data = await response_data
            
        # 4. Store Agent Response
        # The router returns a dict or object. LLMRouter returns dict.
        # SemanticRouter returns RouteResult object.
        # we need to handle both if we want to be generic, but here we know it's LLMRouter (default)
        
        if isinstance(response_data, dict):
            # LLMRouter returns dict
            resp_text = response_data.get('response')
        else:
            # SemanticRouter returns object
            resp_text = response_data.response

        print(f"Agent: {resp_text}")
        app.session_memory.add_message("assistant", str(resp_text))
        
        time.sleep(1)

    print("\n[Trace] Activities logged to trace.json")
    print("[Done] Simulation Complete.")

if __name__ == "__main__":
    asyncio.run(main())

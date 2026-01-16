from agentic_framework.agent import Agent
from agentic_framework.llm_providers import LLMFactory
import os
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json

load_dotenv()


# ============================================================================
# SPECIALIST AGENTS
# ============================================================================

@dataclass
class AgentResponse:
    """Response from a specialist agent."""
    agent_name: str
    subtask: str
    response: str
    success: bool
    metadata: Dict = field(default_factory=dict)


class SpecialistAgent(Agent): # Inherit from Agent
    """Base class for specialist agents using real LLM logic."""
    
    def __init__(self, name: str, specialty: str, instructions: str, llm = None):
        super().__init__(
            name=name, 
            system_prompt=f"Specialty: {specialty}\nInstructions: {instructions}", 
            llm=llm or LLMFactory.auto_detect(),
            tools=[],
            framework=None
        )
        self.specialty = specialty
        self.instructions = instructions
    
    async def handle(self, task: str) -> AgentResponse:
        """Handle a specific task using LLM."""
        print(f"    [{self.name}] Processing: {task}")
        
        prompt = f"""You are the {self.name} specialist. 
Your specialty: {self.specialty}
Instructions: {self.instructions}

Task to perform: {task}

Provide a detailed and helpful response."""
        
        try:
            # Use the Agent's run method, which internally uses self.llm.complete
            response = await self.run(prompt) 
            return AgentResponse(
                agent_name=self.name,
                subtask=task,
                response=response,
                success=True
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                subtask=task,
                response=f"Error processing task: {str(e)}",
                success=False
            )


class TechnicalSupportAgent(SpecialistAgent):
    """Handles technical issues."""
    
    def __init__(self, llm=None):
        super().__init__(
            name="TechnicalSupport",
            specialty="Technical issues, connectivity, errors, system problems",
            instructions="Analyze the technical issue and provide troubleshooting steps or a resolution status.",
            llm=llm
        )
    
    async def handle(self, task: str) -> AgentResponse:
        print(f"    [{self.name}] Processing: {task}")
        
        # Simulate processing
        await asyncio.sleep(0.5)
        
        # In real implementation, this would:
        # 1. Check system logs
        # 2. Run diagnostics
        # 3. Query knowledge base
        # 4. Generate solution
        
        response = f"""I've checked the system logs and diagnostics for your connectivity issue.

Issue detected: Network routing configuration error
Solution: I've reset your connection settings and cleared the DNS cache.
Expected resolution: Within 5-10 minutes

If the issue persists after 10 minutes, please let me know and I'll escalate to our engineering team."""
        
        return AgentResponse(
            agent_name=self.name,
            subtask=task,
            response=response,
            success=True,
            metadata={"issue_type": "connectivity", "severity": "medium"}
        )


class BillingAgent(SpecialistAgent):
    """Handles billing and refunds."""
    
    def __init__(self, llm=None):
        super().__init__(
            name="Billing",
            specialty="Billing, refunds, charges, payments, invoices",
            instructions="Review billing history and process refunds or explain charges.",
            llm=llm
        )
    
    async def handle(self, task: str) -> AgentResponse:
        print(f"    [{self.name}] Processing: {task}")
        
        await asyncio.sleep(0.3)
        
        response = f"""I've reviewed your account and billing history.

Account Status: Active
Current billing cycle: Jan 1-31, 2026
Amount due: $49.99

Due to the connectivity issues you've experienced, I've approved a prorated refund of $8.33 (representing 5 days of service interruption).

The refund will be processed within 5-7 business days to your original payment method.
Reference number: REF-2026-001234"""
        
        return AgentResponse(
            agent_name=self.name,
            subtask=task,
            response=response,
            success=True,
            metadata={"refund_amount": 8.33, "refund_id": "REF-2026-001234"}
        )


class ProductInfoAgent(SpecialistAgent):
    """Handles product information queries."""
    
    def __init__(self, llm=None):
        super().__init__(
            name="ProductInfo",
            specialty="Product features, specifications, pricing, availability",
            instructions="Provide detailed product specs, pricing, and availability info.",
            llm=llm
        )
    
    async def handle(self, task: str) -> AgentResponse:
        print(f"    [{self.name}] Processing: {task}")
        
        await asyncio.sleep(0.4)
        
        response = f"""Here's the information about our products:

Product Catalog:
- Internet Basic: 50 Mbps - $29.99/month
- Internet Pro: 200 Mbps - $49.99/month   (Your current plan)
- Internet Ultra: 1 Gbps - $79.99/month

All plans include:
[OK] No data caps
[OK] Free modem rental
[OK] 24/7 technical support
[OK] 30-day money-back guarantee"""
        
        return AgentResponse(
            agent_name=self.name,
            subtask=task,
            response=response,
            success=True,
            metadata={"product_count": 3}
        )


# ============================================================================
# TASK DECOMPOSITION
# ============================================================================

@dataclass
class Subtask:
    """A decomposed subtask."""
    description: str
    assigned_agent: str
    priority: int = 1  # 1 = highest


class TaskDecomposer:
    """
    Decomposes complex queries into subtasks.
    
    Uses LLM to analyze intent and split into actionable subtasks.
    """
    
    def __init__(self, llm = None):
        self.llm = llm
        self.agents_info = {
            "TechnicalSupport": "Handles technical issues, connectivity problems, errors",
            "Billing": "Handles billing, refunds, charges, payments",
            "ProductInfo": "Handles product information, pricing, features"
        }
    
    def decompose(self, query: str) -> List[Subtask]:
        """
        Decompose query into subtasks.
        
        Args:
            query: User's complex query
            
        Returns:
            List of subtasks with assigned agents
        """
        print(f"\n  Decomposing query: {query}")
        
        # Create prompt for LLM
        agents_desc = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.agents_info.items()
        ])
        
        prompt = f"""You are a task decomposition expert. Analyze this user query and break it into subtasks.

Available agents:
{agents_desc}

User query: "{query}"

Decompose this into 1-3 specific subtasks. For each subtask:
1. Write a clear, actionable description
2. Assign to the most appropriate agent
3. Set priority (1=highest, 3=lowest)

Respond ONLY with valid JSON array:
[
  {{"description": "...", "agent": "TechnicalSupport", "priority": 1}},
  {{"description": "...", "agent": "Billing", "priority": 2}}
]"""
        
        if self.llm:
            response = self.llm.complete(prompt, temperature=0.3)
            content = response.strip()
        else:
            return [Subtask(description=query, assigned_agent="TechnicalSupport", priority=1)]
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        try:
            tasks_data = json.loads(content)
            subtasks = [
                Subtask(
                    description=task["description"],
                    assigned_agent=task["agent"],
                    priority=task.get("priority", 1)
                )
                for task in tasks_data
            ]
            
            print(f"  [OK] Decomposed into {len(subtasks)} subtasks:")
            for i, task in enumerate(subtasks, 1):
                print(f"    {i}. [{task.assigned_agent}] {task.description}")
            
            return subtasks
            
        except json.JSONDecodeError as e:
            print(f"    Failed to parse LLM response: {e}")
            print(f"  Raw response: {content}")
            
            # Fallback: treat as single task
            return [Subtask(
                description=query,
                assigned_agent="TechnicalSupport",
                priority=1
            )]


# ============================================================================
# AGGREGATOR ROUTER
# ============================================================================

class AggregatorRouter:
    """
    The Router (Aggregator Agent) from Chapter 1.3.
    
    Responsibilities:
    1. Analyze user intent
    2. Decompose into subtasks
    3. Route to specialist agents
    4. Execute in parallel
    5. Merge results
    6. Present unified response
    
    Example:
        router = AggregatorRouter()
        result = await router.route("My internet is down, refund my bill")
    """
    
    def __init__(self, llm = None):
        self.llm = llm or LLMFactory.auto_detect()
        
        # Initialize specialist agents
        self.agents: Dict[str, SpecialistAgent] = {
            "TechnicalSupport": TechnicalSupportAgent(llm=self.llm),
            "Billing": BillingAgent(llm=self.llm),
            "ProductInfo": ProductInfoAgent(llm=self.llm)
        }
        
        # Task decomposer
        self.decomposer = TaskDecomposer(llm=llm)
        
        # Conversation state
        self.conversation_history: List[Dict] = []
        
        print("[OK] Aggregator Router initialized")
        print(f"  Registered agents: {', '.join(self.agents.keys())}")
    
    def register_agent(self, name: str, agent: Any, description: Optional[str] = None):
        """Register a new specialist agent."""
        self.agents[name] = agent
        # Also update decomposer's info
        if description:
            self.decomposer.agents_info[name] = description
        elif hasattr(agent, 'specialty'):
            self.decomposer.agents_info[name] = agent.specialty
        elif hasattr(agent, 'metadata'):
            self.decomposer.agents_info[name] = agent.metadata.get('specialty', 'Specialist agent')
        else:
            self.decomposer.agents_info[name] = "Specialist agent"
        
        print(f"  [OK] Registered agent: {name}")

    async def _execute_subtask(self, subtask: Subtask) -> AgentResponse:
        """Execute a single subtask with assigned agent."""
        agent = self.agents.get(subtask.assigned_agent)
        
        if not agent:
            print(f"    Agent not found: {subtask.assigned_agent}")
            return AgentResponse(
                agent_name="Router",
                subtask=subtask.description,
                response=f"Error: Agent {subtask.assigned_agent} not available",
                success=False
            )
        
        return await agent.handle(subtask.description)
    
    async def _execute_parallel(self, subtasks: List[Subtask]) -> List[AgentResponse]:
        """
        Execute multiple subtasks in parallel.
        
        This is a key feature: we don't wait for one agent to finish
        before starting the next. All agents work simultaneously.
        """
        print(f"\n  Executing {len(subtasks)} subtasks in parallel...")
        
        # Sort by priority (lower number = higher priority)
        subtasks.sort(key=lambda x: x.priority)
        
        # Execute all subtasks concurrently
        tasks = [self._execute_subtask(task) for task in subtasks]
        responses = await asyncio.gather(*tasks)
        
        print(f"  [OK] All {len(responses)} agents completed")
        
        return responses
    
    def _merge_responses(self, responses: List[AgentResponse], query: str) -> str:
        """
        Merge multiple agent responses into unified answer using LLM.
        """
        print(f"\n  Merging {len(responses)} responses using LLM...")
        
        successful = [r for r in responses if r.success]
        if not successful:
            return "I apologize, but I encountered errors processing your request."
        
        context = "\n\n".join([f"Agent: {r.agent_name}\nResponse: {r.response}" for r in successful])
        
        prompt = f"""You are the Multi-Agent Aggregator. Your goal is to combine specialist responses into a single, cohesive answer for the user.

User original query: "{query}"

Specialist Responses:
{context}

Respond as a single helpful assistant. Maintain the specific details provided by each specialist."""

        return self.llm.complete(prompt)
    
    async def route(self, query: str) -> Dict[str, Any]:
        """
        Main routing method.
        
        This is the complete flow from Chapter 1.3:
        1. Decompose query
        2. Execute in parallel
        3. Merge results
        4. Return unified response
        
        Args:
            query: User's input
            
        Returns:
            Dictionary with response and metadata
        """
        print(f"\n{'='*70}")
        print(f"ROUTING REQUEST")
        print('='*70)
        
        # Step 1: Decompose
        subtasks = self.decomposer.decompose(query)
        
        # Step 2: Execute in parallel
        responses = await self._execute_parallel(subtasks)
        
        # Step 3: Merge
        final_response = self._merge_responses(responses, query)
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "subtasks": [{"desc": t.description, "agent": t.assigned_agent} for t in subtasks],
            "responses": [{"agent": r.agent_name, "success": r.success} for r in responses]
        })
        
        return {
            "query": query,
            "subtasks_count": len(subtasks),
            "agents_used": list(set(r.agent_name for r in responses)),
            "workers": list(set(r.agent_name for r in responses)), # Case 5 compatibility
            "parallel": True, # AggregatorRouter is parallel by default
            "response": final_response,
            "answer": final_response, # Case 5 compatibility
            "metadata": {
                "successful_tasks": sum(1 for r in responses if r.success),
                "failed_tasks": sum(1 for r in responses if not r.success),
                "total_tasks": len(responses)
            }
        }
    
    def get_stats(self) -> Dict:
        """Get router statistics."""
        return {
            "total_requests": len(self.conversation_history),
            "registered_agents": len(self.agents),
            "average_subtasks": (
                sum(len(h["subtasks"]) for h in self.conversation_history) / len(self.conversation_history)
                if self.conversation_history else 0
            )
        }


# ============================================================================
# DEMO
# ============================================================================

async def demo():
    print("=" * 70)
    print("MULTI-AGENT ROUTER (AGGREGATOR) DEMO")
    print("=" * 70)
    print("\nBased on Chapter 1.3: Production Architecture for 2026")
    print("\nThe Router analyzes intent, routes to specialists,")
    print("executes in parallel, and merges results.\n")
    print("=" * 70)
    
    # Initialize router
    router = AggregatorRouter()
    
    # Test queries (from simple to complex)
    test_queries = [
        # Simple (single agent)
        "My internet connection keeps dropping",
        
        # Medium (two agents)
        "I was charged but my service isn't working",
        
        # Complex (multiple agents from book example)
        "My internet is down and I want a refund on my bill",
        
        # Complex with product info
        "My connection is slow, I want to upgrade, and need a refund for downtime"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print('='*70)
        
        result = await router.route(query)
        
        print(f"\n[CHART] RESULT:")
        print(f"  Subtasks: {result['subtasks_count']}")
        print(f"  Agents used: {', '.join(result['agents_used'])}")
        print(f"  Success rate: {result['metadata']['successful_tasks']}/{result['metadata']['total_tasks']}")
        
        print(f"\n[CHAT] UNIFIED RESPONSE:")
        print(" " * 70)
        print(result['response'])
        print(" " * 70)
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Show stats
    print(f"\n{'='*70}")
    print("ROUTER STATISTICS")
    print('='*70)
    stats = router.get_stats()
    print(f"Total requests processed: {stats['total_requests']}")
    print(f"Registered agents: {stats['registered_agents']}")
    print(f"Average subtasks per request: {stats['average_subtasks']:.1f}")
    
    print("\n" + "="*70)
    print("ARCHITECTURE BENEFITS")
    print("="*70)
    print("""
1. SEPARATION OF CONCERNS
   - Router: Planning & orchestration
   - Agents: Specialized execution
   
2. PARALLEL EXECUTION
   - Multiple agents work simultaneously
   - Faster than sequential processing
   
3. SCALABILITY
   - Easy to add new specialist agents
   - Each agent can be optimized independently
   
4. RELIABILITY
   - If one agent fails, others continue
   - Graceful degradation
   
5. COST OPTIMIZATION
   - Use small models (SLMs) for specialists
   - Only use large models when needed
    """)


if __name__ == "__main__":
    asyncio.run(demo())

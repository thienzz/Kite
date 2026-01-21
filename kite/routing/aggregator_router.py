from kite.agent import Agent
from kite.llm_providers import LLMFactory
import os
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json

load_dotenv()


# ============================================================================
# AGENT RESPONSE
# ============================================================================

@dataclass
class AgentResponse:
    """Response from a specialist agent."""
    agent_name: str
    subtask: str
    response: str
    success: bool
    metadata: Dict = field(default_factory=dict)


class SpecialistAgent(Agent):
    """Base class for specialist agents using LLM logic."""
    
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
            # Use the Agent's run method
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
        self.agents_info = {}
    
    def decompose(self, query: str) -> List[Subtask]:
        """
        Decompose query into subtasks.
        
        Args:
            query: User's complex query
            
        Returns:
            List of subtasks with assigned agents
        """
        if not self.agents_info:
            return [Subtask(description=query, assigned_agent="DefaultAgent", priority=1)]

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
  {{"description": "...", "agent": "SelectedAgentName", "priority": 1}},
  {{"description": "...", "agent": "AnotherAgentName", "priority": 2}}
]"""
        
        if self.llm:
            response = self.llm.complete(prompt, temperature=0.3)
            content = response.strip()
        else:
            # Fallback if no LLM provided (should not happen in production)
            return [Subtask(description=query, assigned_agent=list(self.agents_info.keys())[0], priority=1)]
        
        # Robust JSON extraction
        try:
            # Find the first '[' and last ']'
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx + 1]
            
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
            # Fallback
            return [Subtask(
                description=query,
                assigned_agent=list(self.agents_info.keys())[0],
                priority=1
            )]


# ============================================================================
# AGGREGATOR ROUTER
# ============================================================================

class AggregatorRouter:
    """
    The Router (Aggregator Agent) that orchestrates specialist agents.
    
    Responsibilities:
    1. Analyze user intent
    2. Decompose into subtasks
    3. Route to specialist agents
    4. Execute in parallel
    5. Merge results
    6. Present unified response
    """
    
    def __init__(self, llm = None):
        self.llm = llm or LLMFactory.auto_detect()
        self.agents: Dict[str, Any] = {}
        self.decomposer = TaskDecomposer(llm=self.llm)
        self.conversation_history: List[Dict] = []
        
        print("[OK] Aggregator Router initialized")
    
    def register_agent(self, name: str, agent: Any, description: Optional[str] = None):
        """Register a new specialist agent."""
        self.agents[name] = agent
        
        # Update decomposer's info
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
            return AgentResponse(
                agent_name="Router",
                subtask=subtask.description,
                response=f"Error: Agent {subtask.assigned_agent} not available",
                success=False
            )
        
        # Support both SpecialistAgent (has handle method) and generic Agent (has run method)
        if hasattr(agent, 'handle'):
            return await agent.handle(subtask.description)
        elif hasattr(agent, 'run'):
            resp = await agent.run(subtask.description)
            return AgentResponse(
                agent_name=subtask.assigned_agent,
                subtask=subtask.description,
                response=resp,
                success=True
            )
        else:
            return AgentResponse(
                agent_name=subtask.assigned_agent,
                subtask=subtask.description,
                response=f"Error: Agent {subtask.assigned_agent} is not compatible",
                success=False
            )
    
    async def _execute_parallel(self, subtasks: List[Subtask]) -> List[AgentResponse]:
        """Execute multiple subtasks in parallel."""
        print(f"\n  Executing {len(subtasks)} subtasks in parallel...")
        
        tasks = [self._execute_subtask(task) for task in subtasks]
        responses = await asyncio.gather(*tasks)
        
        print(f"  [OK] All {len(responses)} agents completed")
        return responses
    
    def _merge_responses(self, responses: List[AgentResponse], query: str) -> str:
        """Merge multiple agent responses into unified answer."""
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
        """Main routing method."""
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
            "route": subtasks[0].assigned_agent if len(subtasks) == 1 else "multi",
            "subtasks_count": len(subtasks),
            "agents_used": list(set(r.agent_name for r in responses)),
            "workers": list(set(r.agent_name for r in responses)),
            "parallel": True,
            "response": final_response,
            "answer": final_response,
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

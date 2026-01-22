"""
Conversation Manager
Handles multi-turn dialogue, context switching, and collaboration between agents.
"""

import asyncio
import json
from typing import List, Dict, Optional, Any, Callable
from .agent import Agent

class ConversationManager:
    """
    Orchestrates multi-turn dialogue between multiple agents.
    
    Features:
    - Multi-turn history management.
    - Termination conditions (max_turns, consensus).
    - Collaboration patterns (round-robin).
    """
    
    def __init__(self, 
                 agents: List[Agent], 
                 framework,
                 max_turns: int = 10,
                 min_turns: int = 3,
                 termination_condition: str = "consensus"):
        if not agents or len(agents) == 0:
            raise ValueError("Conversation requires at least one agent")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")
            
        self.agents = agents
        self.framework = framework
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.termination_condition = termination_condition.lower()
        self.history = []
        self.logger = framework.logger

    async def run(self, initial_input: str) -> Dict[str, Any]:
        """
        Run the multi-turn conversation.
        """
        print(f"\n[Conversation] Starting with input: {initial_input}")
        
        current_input = initial_input
        turn_count = 0
        consensus_reached = False
        
        while turn_count < self.max_turns and not consensus_reached:
            turn_count += 1
            # Round-robin turn taking (simplified for now)
            agent_idx = (turn_count - 1) % len(self.agents)
            current_agent = self.agents[agent_idx]
            
            print(f"  [Turn {turn_count}] Agent: {current_agent.name}")
            
            # Prepare context with full history
            context = {
                "conversation_history": self.history,
                "current_turn": turn_count,
                "max_turns": self.max_turns
            }
            
            # Run agent
            res = await current_agent.run(current_input, context=context)
            
            if not res['success']:
                print(f"  [Error] Agent {current_agent.name} failed: {res.get('error')}")
                break
                
            response = res['response']
            self.history.append({
                "turn": turn_count,
                "agent": current_agent.name,
                "content": response
            })
            
            # Check for termination
            if self.termination_condition == "consensus":
                consensus_reached = await self._check_consensus(response, turn_count)
                if consensus_reached:
                    print(f"  [Consensus] Reached by {current_agent.name}")
            
            # Set input for next agent (usually the response of the current one)
            current_input = response
            
        print(f"\n[Conversation] Finished after {turn_count} turns.")
        
        return {
            "success": True,
            "turns": turn_count,
            "history": self.history,
            "final_response": self.history[-1]['content'] if self.history else "No history",
            "termination": "consensus" if consensus_reached else "max_turns"
        }

    async def _check_consensus(self, last_response: str, turn_count: int) -> bool:
        """
        Check if consensus or a final conclusion has been reached.
        Only valid after min_turns.
        """
        if turn_count < self.min_turns:
            return False
            
        # Stricter declarative markers
        keywords = [
            "final consensus reached", 
            "consensus reached", 
            "final conclusion:", 
            "i agree, let's reach a consensus",
            "we have reached an agreement"
        ]
        
        response_lower = last_response.lower()
        for kw in keywords:
            if kw in response_lower:
                return True
        
        return False

    def get_summary(self) -> str:
        """Synthesize a summary of the conversation."""
        summary = "\n".join([f"{h['agent']}: {h['content'][:100]}..." for h in self.history])
        return summary

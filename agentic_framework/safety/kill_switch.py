"""
KillSwitch - Safety limits for autonomous agents.
"""

import time
from typing import Dict, Tuple, Optional, Any


class KillSwitch:
    """
    Safety limits for autonomous agents to prevent infinite loops, 
    excessive costs, or hanging processes.
    """
    
    def __init__(self, 
                 max_iterations: int = 10, 
                 max_cost: float = 1.0, 
                 max_same_action: int = 2, 
                 max_time: int = 300):
        """
        Initialize KillSwitch with safety limits.
        
        Args:
            max_iterations: Maximum number of steps/loops allowed.
            max_cost: Maximum total cost in USD allowed.
            max_same_action: Maximum number of times the same action can be repeated consecutively.
            max_time: Maximum time in seconds allowed.
        """
        self.max_iterations = max_iterations
        self.max_cost_usd = max_cost
        self.max_same_action = max_same_action
        self.max_time_seconds = max_time
    
    def check(self, state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check all kill switch limits against the current agent state.
        
        Args:
            state: Dictionary containing current agent state:
                - steps (int): Current iteration count
                - total_cost (float): Accumulated cost
                - start_time (float): Time when the process started (time.time())
                - actions (list): List of actions taken
                - completed (bool): Whether the goal is achieved
                
        Returns:
            Tuple (should_stop, reason)
        """
        
        # Limit 1: Iteration cap
        if state.get('steps', 0) >= self.max_iterations:
            return True, f"Max iterations ({self.max_iterations}) reached."
        
        # Limit 2: Budget cap
        if state.get('total_cost', 0.0) >= self.max_cost_usd:
            return True, f"Budget exceeded (${self.max_cost_usd})."
        
        # Limit 3: Time limit
        if 'start_time' in state:
            elapsed = time.time() - state['start_time']
            if elapsed >= self.max_time_seconds:
                return True, f"Time limit ({self.max_time_seconds}s) exceeded."
        
        # Limit 4: Stupidity check (repeated actions)
        actions = state.get('actions', [])
        if len(actions) >= self.max_same_action:
            recent = [a.get('type') for a in actions[-self.max_same_action:]]
            if len(set(recent)) == 1 and recent[0] is not None:
                return True, f"Stuck in loop (same action '{recent[0]}' repeated {self.max_same_action} times)."
        
        # Limit 5: Goal completed
        if state.get('completed', False):
            return True, "Goal achieved."
        
        return False, None

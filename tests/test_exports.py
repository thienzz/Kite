import os
import sys

# Ensure we are testing the local version
sys.path.append(os.getcwd())

try:
    from kite import (
        Kite, Agent, Tool, 
        CircuitBreaker, CircuitState, 
        ReActAgent, PlanExecuteAgent,
        __version__
    )
    print(f"Kite v{__version__} imports successful!")
    print(f"Kite: {Kite}")
    print(f"Agent: {Agent}")
    print(f"ReActAgent: {ReActAgent}")
    print(f"CircuitState: {CircuitState}")
    print("[OK] All top-level exports verified.")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

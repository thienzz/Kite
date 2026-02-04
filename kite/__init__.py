"""
Kite Framework - Production-Ready Agentic AI

A lightweight, safety-first framework for building intelligent AI agents
with enterprise-grade reliability.
"""

__version__ = "0.1.0"
__author__ = "Thien Nguyen"
__license__ = "MIT"

from .core import Kite
from .agent import Agent
from .tool import Tool

# Safety components
from .safety.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .safety.idempotency_manager import IdempotencyManager
from .safety.kill_switch import KillSwitch

# Agent patterns
from .agents.react_agent import ReActAgent
from .agents.plan_execute import PlanExecuteAgent
from .agents.rewoo import ReWOOAgent
from .agents.tot import TreeOfThoughtsAgent

__all__ = [
    # Core
    "Kite",
    "Agent",
    "Tool",
    "__version__",
    
    # Safety
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "IdempotencyManager",
    "KillSwitch",
    
    # Agents
    "ReActAgent",
    "PlanExecuteAgent",
    "ReWOOAgent",
    "TreeOfThoughtsAgent",
]

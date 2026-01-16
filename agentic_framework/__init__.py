"""
AgenticAI Framework v1.0
Complete production-ready framework with ALL implementations integrated.

Single import, everything works:
    from agentic_framework import AgenticAI
    
    ai = AgenticAI()
    result = ai.process("customer query")
"""

__version__ = "1.0.0"

from .core import AgenticAI

__all__ = ['AgenticAI']

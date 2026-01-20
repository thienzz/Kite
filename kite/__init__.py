"""
Kite Framework v1.0
Complete production-ready framework with ALL implementations integrated.

Single import, everything works:
    from kite import Kite
    
    ai = Kite()
    result = ai.process("customer query")
"""

__version__ = "1.0.0"

from .core import Kite

# Maintain backward compatibility
AgenticAI = Kite

__all__ = ['Kite', 'AgenticAI']

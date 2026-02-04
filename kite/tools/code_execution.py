"""
Python Execution Tool
====================
Allows agents to run Python code for data analysis, math, or visualization.
WARNING: This uses `exec`. In a real production environment, this should run in a Docker container or Firecracker VM.
"""

from typing import Any, Dict
import sys
import io
import traceback
from kite.tool import Tool

class PythonReplTool(Tool):
    def __init__(self):
        super().__init__(
            name="python",
            func=self.execute,
            description="Executes Python code. Use this for data analysis (pandas), visualization (matplotlib), or complex calculations. Input must be valid python code string."
        )
        self.globals = {}
        self.locals = {}

    async def execute(self, code: str, **kwargs) -> str:
        """
        Executes the provided Python code and returns standout/stderr.
        """
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        
        try:
            # Dangerous in real prod, but okay for local demo with user consent
            # We strictly namespace it
            
            # Pre-import common libs if available
            try:
                import pandas as pd
                import matplotlib.pyplot as plt
                self.globals['pd'] = pd
                self.globals['plt'] = plt
            except ImportError:
                pass
                
            exec(code, self.globals, self.locals)
            
            output = redirected_output.getvalue()
            return output if output.strip() else "[Code executed successfully with no output]"
            
        except Exception as e:
            return f"Error executing code:\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

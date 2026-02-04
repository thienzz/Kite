"""
System & Shell Tools
===================
Tools for interacting with the operating system.
Designed for DevOps agents.
"""

import subprocess
import shlex
from kite.tool import Tool

class ShellTool(Tool):
    def __init__(self, allowed_commands=None):
        """
        Args:
            allowed_commands: List of allowed command prefixes (e.g. ['ls', 'grep']). 
                              If None, allows everything (DANGEROUS).
        """
        super().__init__(
            name="shell_execute",
            func=self.execute,
            description="Execute a shell command. Use for system inspection, git operations, or file checks."
        )
        self.allowed_commands = allowed_commands or ["ls", "cat", "grep", "echo", "git", "pwd", "whoami"]

    async def execute(self, command: str, **kwargs) -> str:
        """Executes shell command."""
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return "Empty command."
            
        base_cmd = cmd_parts[0]
        
        # Security Check
        if base_cmd not in self.allowed_commands:
            return f"Security Alert: Command '{base_cmd}' is not allowed. Allowed: {self.allowed_commands}"
            
        if "rm" in cmd_parts and "-rf" in cmd_parts:
             return "Security Alert: 'rm -rf' is strictly forbidden."

        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip() or "[Command executed with no output]"
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Execution Failed: {e}"

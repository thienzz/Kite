"""
Tool Registry - Register and manage tools.
"""

from typing import Dict
import logging


class ToolRegistry:
    """
    Registry for all tools.
    
    Example:
        ai.tools.register("my_tool", tool)
        tool = ai.tools.get("my_tool")
        all_tools = ai.tools.list()
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._tools = {}
        
        # Optionally initialize MCP servers
        self._init_mcp_servers()
    
    def _init_mcp_servers(self):
        """Initialize MCP servers if credentials available."""
        try:
            from .tools_manager import MCPServers
            self.mcp = MCPServers(self.config, self.logger)
            self.logger.info("  [OK] Tools (MCP)")
        except Exception as e:
            self.logger.info("    Tools (manual registration)")
            self.mcp = None
    
    def register(self, name: str, tool):
        """Register a tool."""
        self._tools[name] = tool
        self.logger.info(f"  [OK] Registered tool: {name}")
    
    def get(self, name: str):
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list(self):
        """List all registered tools."""
        return list(self._tools.keys())
    
    def get_all(self):
        """Get all tools."""
        return self._tools

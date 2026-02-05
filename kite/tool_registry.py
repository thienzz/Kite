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
        
    def load_standard_tools(self, framework):
        """Automatically load and register all standard contrib tools."""
        from .tool import Tool
        from .tools.contrib import (
            get_current_datetime,
        )
        
        # Try to import optional dependencies
        try:
            from .tools.web_search import web_search
            has_web_search = True
        except:
            has_web_search = False
            
        try:
            from .tools.contrib.calculator import calculator
            has_calculator = True
        except:
            has_calculator = False
        
        # Basic tools that are always available
        standard_tools = [
            ("get_datetime", get_current_datetime, "Get current date and time"),
        ]
        
        # Add optional tools if available
        if has_web_search:
            standard_tools.append(("web_search", web_search, "Search the web for information"))
        if has_calculator:
            standard_tools.append(("calculator", calculator, "Evaluate mathematical expressions"))
        
        for name, func, desc in standard_tools:
            if name not in self._tools:
                self.register(name, Tool(name, func, desc))

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

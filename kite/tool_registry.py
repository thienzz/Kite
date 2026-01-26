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
            web_search, 
            calculator, 
            get_current_datetime,
            search_linkedin_posts,
            get_linkedin_profile_details,
            get_linkedin_company_details,
            create_linkedin_session
        )
        
        standard_tools = [
            ("web_search", web_search, "Search the web for information"),
            ("calculator", calculator, "Evaluate mathematical expressions"),
            ("get_datetime", get_current_datetime, "Get current date and time"),
            ("search_linkedin", search_linkedin_posts, "Search for LinkedIn posts"),
            ("get_profile", get_linkedin_profile_details, "Get detailed profile information"),
            ("get_company", get_linkedin_company_details, "Get detailed company information"),
            ("create_session", create_linkedin_session, "Create a LinkedIn session by logging in manually")
        ]
        
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

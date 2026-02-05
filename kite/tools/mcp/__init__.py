"""
MCP (Model Context Protocol) Server Implementations

This package contains MCP server implementations for external integrations.
MCP servers provide standardized interfaces to external services that can be
used by AI agents through the Kite framework.

Available MCP Servers:
- SlackMCPServer: Slack workspace integration
- StripeMCPServer: Stripe payment processing
- GmailMCPServer: Gmail email management
- GDriveMCPServer: Google Drive file management
- PostgresMCPServer: PostgreSQL database access
- DatabaseMCP: Multi-database connector (SQLite, MySQL, Redis, Neo4j, etc.)
"""

from .slack_mcp_server import SlackMCPServer
from .stripe_mcp_server import StripeMCPServer
from .gmail_mcp_server import GmailMCPServer
from .gdrive_mcp_server import GDriveMCPServer
from .postgres_mcp_server import PostgresMCPServer
from .database_mcp import DatabaseMCP

__all__ = [
    'SlackMCPServer',
    'StripeMCPServer', 
    'GmailMCPServer',
    'GDriveMCPServer',
    'PostgresMCPServer',
    'DatabaseMCP'
]

"""Tools (MCP Servers) Manager"""

class MCPServers:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._init_servers()
    
    def _init_servers(self):
        """Initialize MCP servers from the tools.mcp package."""
        from .tools.mcp import (
            PostgresMCPServer, 
            SlackMCPServer, 
            StripeMCPServer,
            GmailMCPServer, 
            GDriveMCPServer
        )
        
        self.postgres = PostgresMCPServer(connection_string=self.config.get('postgres_url'))
        self.slack = SlackMCPServer(bot_token=self.config.get('slack_token'))
        self.stripe = StripeMCPServer(api_key=self.config.get('stripe_key'))
        self.gmail = GmailMCPServer(credentials_path=self.config.get('gmail_credentials'))
        self.gdrive = GDriveMCPServer(credentials_path=self.config.get('gdrive_credentials'))
        
        self.logger.info("  [OK] Tools (MCP)")



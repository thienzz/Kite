"""Tools (MCP Servers) Manager"""

class MCPServers:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._init_servers()
    
    def _init_servers(self):
        try:
            from .tools import (PostgresMCPServer, SlackMCPServer, StripeMCPServer,
                               GmailMCPServer, GoogleDriveMCPServer)
            self.postgres = PostgresMCPServer(connection_string=self.config.get('postgres_url'))
            self.slack = SlackMCPServer(bot_token=self.config.get('slack_token'))
            self.stripe = StripeMCPServer(api_key=self.config.get('stripe_key'))
            self.gmail = GmailMCPServer(credentials_path=self.config.get('gmail_credentials'))
            self.gdrive = GoogleDriveMCPServer(credentials_path=self.config.get('gdrive_credentials'))
            self.logger.info("  [OK] Tools")
        except Exception as e:
            self.logger.warning(f"    Tools mock: {e}")
            from .mocks import MockPostgres, MockSlack, MockStripe, MockGmail, MockGDrive
            self.postgres = MockPostgres()
            self.slack = MockSlack()
            self.stripe = MockStripe()
            self.gmail = MockGmail()
            self.gdrive = MockGDrive()

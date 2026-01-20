"""
Slack MCP Server Implementation
Based on Chapter 4: MCP (Model Context Protocol)

Allows AI agents to interact with Slack workspaces.

Tools provided:
- send_message: Send messages to channels
- read_channel: Read recent messages
- search_messages: Search message history
- get_user_info: Get user details

Run: python slack_mcp_server.py
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(".kite.env")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SlackConfig:
    """Configuration for Slack MCP server."""
    bot_token: str = ""
    workspace_id: str = ""
    
    # Safety limits
    max_message_length: int = 4000
    rate_limit_per_minute: int = 60
    allowed_channels: Optional[List[str]] = None  # None = all channels
    
    # Features
    enable_send: bool = True
    enable_search: bool = True
    enable_user_lookup: bool = True


# ============================================================================
# MOCK SLACK CLIENT (Replace with real slack_sdk in production)
# ============================================================================

class MockSlackClient:
    """
    Mock Slack client for demonstration.
    
    In production, use:
    from slack_sdk import WebClient
    client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    """
    
    def __init__(self, token: str):
        self.token = token
        self.messages_db = {}
        self.users_db = {
            "U123": {"id": "U123", "name": "John Doe", "email": "john@company.com"},
            "U456": {"id": "U456", "name": "Jane Smith", "email": "jane@company.com"},
        }
        self.channels_db = {
            "C001": {"id": "C001", "name": "general", "topic": "General discussions"},
            "C002": {"id": "C002", "name": "engineering", "topic": "Engineering team"},
            "C003": {"id": "C003", "name": "support", "topic": "Customer support"},
        }
        
        # Seed some messages
        self._seed_messages()
    
    def _seed_messages(self):
        """Add some demo messages."""
        now = datetime.now()
        
        self.messages_db = {
            "C001": [
                {
                    "user": "U123",
                    "text": "Good morning team!",
                    "ts": (now - timedelta(hours=2)).timestamp()
                },
                {
                    "user": "U456",
                    "text": "Morning! Ready for the sprint planning?",
                    "ts": (now - timedelta(hours=1, minutes=30)).timestamp()
                }
            ],
            "C002": [
                {
                    "user": "U123",
                    "text": "The deployment to production was successful",
                    "ts": (now - timedelta(hours=3)).timestamp()
                },
                {
                    "user": "U456",
                    "text": "Great! All tests passing?",
                    "ts": (now - timedelta(hours=2, minutes=45)).timestamp()
                },
                {
                    "user": "U123",
                    "text": "Yes, 100% test coverage maintained",
                    "ts": (now - timedelta(hours=2, minutes=30)).timestamp()
                }
            ]
        }
    
    def chat_postMessage(self, channel: str, text: str) -> Dict:
        """Send a message to channel."""
        message = {
            "user": "BOT",
            "text": text,
            "ts": datetime.now().timestamp()
        }
        
        if channel not in self.messages_db:
            self.messages_db[channel] = []
        
        self.messages_db[channel].append(message)
        
        return {
            "ok": True,
            "channel": channel,
            "ts": message["ts"],
            "message": message
        }
    
    def conversations_history(self, channel: str, limit: int = 10) -> Dict:
        """Get channel message history."""
        messages = self.messages_db.get(channel, [])
        
        # Sort by timestamp (most recent first)
        messages = sorted(messages, key=lambda x: x["ts"], reverse=True)
        
        return {
            "ok": True,
            "messages": messages[:limit]
        }
    
    def search_messages(self, query: str) -> Dict:
        """Search messages across channels."""
        results = []
        
        for channel, messages in self.messages_db.items():
            for msg in messages:
                if query.lower() in msg["text"].lower():
                    results.append({
                        **msg,
                        "channel": channel
                    })
        
        return {
            "ok": True,
            "messages": {"matches": results}
        }
    
    def users_info(self, user: str) -> Dict:
        """Get user information."""
        user_data = self.users_db.get(user)
        
        if user_data:
            return {
                "ok": True,
                "user": user_data
            }
        else:
            return {
                "ok": False,
                "error": "user_not_found"
            }


# ============================================================================
# SLACK MCP SERVER
# ============================================================================

class SlackMCPServer:
    """
    MCP Server for Slack integration.
    
    Provides tools for AI agents to interact with Slack:
    - Send messages to channels
    - Read channel history
    - Search messages
    - Get user information
    
    Example:
        config = SlackConfig(bot_token="xoxb-...")
        server = SlackMCPServer(config)
        
        # Send message
        result = server.send_message("C001", "Hello team!")
        
        # Read channel
        messages = server.read_channel("C001", limit=5)
    """
    
    def __init__(self, config: SlackConfig = None, bot_token: str = None):
        self.config = config or SlackConfig()
        if bot_token:
            self.config.bot_token = bot_token
        
        # Initialize Slack client (mock for demo)
        self.client = MockSlackClient(self.config.bot_token)
        
        # Rate limiting
        self.request_count = 0
        self.window_start = datetime.now()
        
        print(f"[OK] Slack MCP Server initialized")
        print(f"  Workspace: {self.config.workspace_id or 'demo'}")
        print(f"  Rate limit: {self.config.rate_limit_per_minute}/min")
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows request."""
        now = datetime.now()
        
        # Reset counter if window expired
        if (now - self.window_start).seconds >= 60:
            self.request_count = 0
            self.window_start = now
        
        # Check limit
        if self.request_count >= self.config.rate_limit_per_minute:
            return False
        
        self.request_count += 1
        return True
    
    def _is_channel_allowed(self, channel: str) -> bool:
        """Check if channel is in allowed list."""
        if self.config.allowed_channels is None:
            return True
        
        return channel in self.config.allowed_channels
    
    def send_message(self, channel: str, text: str) -> Dict[str, Any]:
        """
        Send message to Slack channel.
        
        Args:
            channel: Channel ID (e.g., "C001") or name
            text: Message text
            
        Returns:
            Result dictionary
        """
        print(f"\n  Sending message to {channel}")
        
        # Safety checks
        if not self.config.enable_send:
            return {
                "success": False,
                "error": "Send messages is disabled"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        if not self._is_channel_allowed(channel):
            return {
                "success": False,
                "error": f"Channel {channel} not in allowed list"
            }
        
        if len(text) > self.config.max_message_length:
            return {
                "success": False,
                "error": f"Message too long ({len(text)} > {self.config.max_message_length})"
            }
        
        # Send message
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text
            )
            
            if response.get("ok"):
                print(f"  [OK] Message sent successfully")
                return {
                    "success": True,
                    "channel": channel,
                    "timestamp": response["ts"]
                }
            else:
                print(f"    Failed: {response.get('error')}")
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            print(f"    Exception: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def read_channel(self, channel: str, limit: int = 10) -> Dict[str, Any]:
        """
        Read recent messages from channel.
        
        Args:
            channel: Channel ID
            limit: Number of messages to retrieve
            
        Returns:
            Messages list
        """
        print(f"\n  Reading {limit} messages from {channel}")
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        if not self._is_channel_allowed(channel):
            return {
                "success": False,
                "error": f"Channel {channel} not in allowed list"
            }
        
        try:
            response = self.client.conversations_history(
                channel=channel,
                limit=min(limit, 100)  # Cap at 100
            )
            
            if response.get("ok"):
                messages = response["messages"]
                print(f"  [OK] Retrieved {len(messages)} messages")
                
                return {
                    "success": True,
                    "channel": channel,
                    "messages": messages,
                    "count": len(messages)
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_messages(self, query: str) -> Dict[str, Any]:
        """
        Search messages across workspace.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        print(f"\n  Searching for: {query}")
        
        if not self.config.enable_search:
            return {
                "success": False,
                "error": "Search is disabled"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            response = self.client.search_messages(query=query)
            
            if response.get("ok"):
                matches = response["messages"]["matches"]
                print(f"  [OK] Found {len(matches)} matches")
                
                return {
                    "success": True,
                    "query": query,
                    "results": matches,
                    "count": len(matches)
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get user information.
        
        Args:
            user_id: User ID
            
        Returns:
            User info
        """
        print(f"\n  Getting info for user {user_id}")
        
        if not self.config.enable_user_lookup:
            return {
                "success": False,
                "error": "User lookup is disabled"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            response = self.client.users_info(user=user_id)
            
            if response.get("ok"):
                user = response["user"]
                print(f"  [OK] Found user: {user.get('name')}")
                
                return {
                    "success": True,
                    "user": user
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error")
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get MCP tool definitions for AI agents.
        
        These follow the MCP standard format.
        """
        tools = []
        
        if self.config.enable_send:
            tools.append({
                "name": "slack_send_message",
                "description": "Send a message to a Slack channel",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel ID or name (e.g., 'C001' or 'general')"
                        },
                        "text": {
                            "type": "string",
                            "description": "Message text to send"
                        }
                    },
                    "required": ["channel", "text"]
                }
            })
        
        tools.append({
            "name": "slack_read_channel",
            "description": "Read recent messages from a Slack channel",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID or name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to retrieve (max 100)",
                        "default": 10
                    }
                },
                "required": ["channel"]
            }
        })
        
        if self.config.enable_search:
            tools.append({
                "name": "slack_search_messages",
                "description": "Search messages across Slack workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            })
        
        if self.config.enable_user_lookup:
            tools.append({
                "name": "slack_get_user_info",
                "description": "Get information about a Slack user",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Slack user ID"
                        }
                    },
                    "required": ["user_id"]
                }
            })
        
        return tools


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("SLACK MCP SERVER DEMO")
    print("=" * 70)
    print("\nBased on Chapter 4: Model Context Protocol")
    print("Allows AI agents to interact with Slack workspaces\n")
    print("=" * 70)
    
    # Initialize server
    config = SlackConfig(
        bot_token="xoxb-demo-token",
        workspace_id="demo-workspace",
        allowed_channels=["C001", "C002", "C003"]
    )
    
    server = SlackMCPServer(config)
    
    # Demo 1: Read channel
    print(f"\n{'='*70}")
    print("DEMO 1: Read channel messages")
    print('='*70)
    
    result = server.read_channel("C002", limit=5)
    if result["success"]:
        print(f"\nMessages in {result['channel']}:")
        for msg in result["messages"]:
            user = msg.get("user", "Unknown")
            text = msg.get("text", "")
            print(f"  [{user}] {text}")
    
    # Demo 2: Search messages
    print(f"\n{'='*70}")
    print("DEMO 2: Search messages")
    print('='*70)
    
    result = server.search_messages("deployment")
    if result["success"]:
        print(f"\nFound {result['count']} messages matching '{result['query']}':")
        for msg in result["results"]:
            channel = msg.get("channel", "Unknown")
            user = msg.get("user", "Unknown")
            text = msg.get("text", "")
            print(f"  [#{channel}] [{user}] {text}")
    
    # Demo 3: Send message
    print(f"\n{'='*70}")
    print("DEMO 3: Send message")
    print('='*70)
    
    result = server.send_message(
        "C001",
        "Hello team! This is a message from the AI agent."
    )
    print(f"\nSend result: {result}")
    
    # Demo 4: Get user info
    print(f"\n{'='*70}")
    print("DEMO 4: Get user info")
    print('='*70)
    
    result = server.get_user_info("U123")
    if result["success"]:
        user = result["user"]
        print(f"\nUser information:")
        print(f"  ID: {user['id']}")
        print(f"  Name: {user['name']}")
        print(f"  Email: {user.get('email', 'N/A')}")
    
    # Show tool definitions
    print(f"\n{'='*70}")
    print("MCP TOOL DEFINITIONS")
    print('='*70)
    
    tools = server.get_tool_definitions()
    print(f"\nAvailable tools: {len(tools)}")
    for tool in tools:
        print(f"\n    {tool['name']}")
        print(f"    {tool['description']}")
    
    print("\n" + "="*70)
    print("USAGE WITH AI AGENT")
    print("="*70)
    print("""
# In your agent code:
from slack_mcp_server import SlackMCPServer, SlackConfig

# Initialize
config = SlackConfig(bot_token=os.getenv("SLACK_BOT_TOKEN"))
slack = SlackMCPServer(config)

# Get tools for agent
tools = slack.get_tool_definitions()

# When agent calls tool:
if tool_name == "slack_send_message":
    result = slack.send_message(args["channel"], args["text"])
elif tool_name == "slack_read_channel":
    result = slack.read_channel(args["channel"], args.get("limit", 10))
elif tool_name == "slack_search_messages":
    result = slack.search_messages(args["query"])
    """)


if __name__ == "__main__":
    demo()

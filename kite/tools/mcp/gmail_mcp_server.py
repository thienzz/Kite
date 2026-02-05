"""
Gmail MCP Server Implementation
Based on Chapter 4: MCP (Model Context Protocol)

Allows AI agents to interact with Gmail.

Tools provided:
- search_emails: Search emails by query
- read_email: Read specific email
- send_email: Send new email
- list_labels: Get Gmail labels

Run: python gmail_mcp_server.py
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GmailConfig:
    """Configuration for Gmail MCP server."""
    credentials_path: str = ""
    
    # Safety limits
    max_emails_per_search: int = 50
    max_email_size_kb: int = 1024
    rate_limit_per_minute: int = 60
    
    # Allowed operations
    enable_read: bool = True
    enable_send: bool = False  # Disabled by default for safety
    enable_search: bool = True
    
    # Content filters
    max_attachment_size_mb: int = 10
    blocked_domains: List[str] = field(default_factory=list)


# ============================================================================
# GMAIL MCP SERVER
# ============================================================================

class GmailMCPServer:
    """
    MCP Server for Gmail integration.
    
    Provides tools for AI agents to search and manage emails.
    
    Requires: google-api-python-client, google-auth-httplib2, google-auth-oauthlib
    Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
    
    Setup:
        1. Enable Gmail API in Google Cloud Console
        2. Download credentials.json
        3. Run authentication flow to get token
    
    Example:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        
        config = GmailConfig(credentials_path='~/.gmail_credentials.json')
        server = GmailMCPServer(config)
        
        # Search emails
        results = server.search_emails("Project Zeus")
        
        # Read email
        email = server.read_email("email001")
    """
    
    def __init__(self, config: GmailConfig = None, credentials_path: str = "", **kwargs):
        self.config = config or GmailConfig()
        if credentials_path:
            self.config.credentials_path = credentials_path
        
        # Initialize Gmail client (requires Google API client)
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            import os
            import pickle
            
            SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
                      'https://www.googleapis.com/auth/gmail.send']
            
            creds = None
            token_path = os.path.expanduser('~/.gmail_token.pickle')
            
            # Load existing token
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif self.config.credentials_path and os.path.exists(self.config.credentials_path):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                else:
                    raise Exception("Gmail credentials not found. Please provide credentials_path")
                
                # Save token
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            self.gmail = build('gmail', 'v1', credentials=creds)
            
        except ImportError:
            raise ImportError(
                "Google API client is required for GmailMCPServer. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Gmail client: {e}")

        
        # Rate limiting
        self.request_count = 0
        self.window_start = datetime.now()
        
        print(f"[OK] Gmail MCP Server initialized")
        print(f"  Rate limit: {self.config.rate_limit_per_minute}/min")
        print(f"  Send enabled: {self.config.enable_send}")
    
    def _check_rate_limit(self) -> bool:
        """Check rate limit."""
        now = datetime.now()
        
        if (now - self.window_start).seconds >= 60:
            self.request_count = 0
            self.window_start = now
        
        if self.request_count >= self.config.rate_limit_per_minute:
            return False
        
        self.request_count += 1
        return True
    
    def _extract_header(self, headers: List[Dict], name: str) -> str:
        """Extract header value."""
        for header in headers:
            if header["name"].lower() == name.lower():
                return header["value"]
        return ""
    
    def search_emails(
        self,
        query: str,
        max_results: int = 10,
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search emails by query.
        
        Args:
            query: Search query (keywords, sender, subject, etc.)
            max_results: Maximum results to return
            label: Filter by label (INBOX, SENT, etc.)
            
        Returns:
            Search results
        """
        print(f"\n  Searching emails: {query}")
        
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
            # Search
            label_ids = [label] if label else None
            results = self.gmail.users_messages_list(
                userId="me",
                q=query,
                maxResults=min(max_results, self.config.max_emails_per_search),
                labelIds=label_ids
            )
            
            # Get email details
            emails = []
            for msg in results.get("messages", []):
                email = self.gmail.users_messages_get("me", msg["id"])
                
                headers = email["payload"]["headers"]
                emails.append({
                    "id": email["id"],
                    "from": self._extract_header(headers, "From"),
                    "to": self._extract_header(headers, "To"),
                    "subject": self._extract_header(headers, "Subject"),
                    "date": self._extract_header(headers, "Date"),
                    "snippet": email["snippet"],
                    "labels": email["labelIds"]
                })
            
            print(f"  [OK] Found {len(emails)} emails")
            
            return {
                "success": True,
                "query": query,
                "emails": emails,
                "count": len(emails)
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def read_email(self, email_id: str) -> Dict[str, Any]:
        """
        Read specific email.
        
        Args:
            email_id: Gmail message ID
            
        Returns:
            Email content
        """
        print(f"\n  Reading email: {email_id}")
        
        if not self.config.enable_read:
            return {
                "success": False,
                "error": "Read is disabled"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            email = self.gmail.users_messages_get("me", email_id)
            
            headers = email["payload"]["headers"]
            body = email["payload"]["body"]["data"]
            
            print(f"  [OK] Read email from: {self._extract_header(headers, 'From')}")
            
            return {
                "success": True,
                "email_id": email_id,
                "from": self._extract_header(headers, "From"),
                "to": self._extract_header(headers, "To"),
                "subject": self._extract_header(headers, "Subject"),
                "date": self._extract_header(headers, "Date"),
                "body": body,
                "labels": email["labelIds"]
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send email.
        
        Args:
            to: Recipient email
            subject: Email subject
            body: Email body
            cc: CC recipients (optional)
            
        Returns:
            Send result
        """
        print(f"\n  Sending email to: {to}")
        
        if not self.config.enable_send:
            return {
                "success": False,
                "error": "Send is disabled (safety)"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        # Check blocked domains
        domain = to.split("@")[-1]
        if domain in self.config.blocked_domains:
            return {
                "success": False,
                "error": f"Domain blocked: {domain}"
            }
        
        try:
            # Create message
            message = {
                "payload": {
                    "headers": [
                        {"name": "To", "value": to},
                        {"name": "Subject", "value": subject}
                    ],
                    "body": {"data": body}
                }
            }
            
            if cc:
                message["payload"]["headers"].append({"name": "Cc", "value": cc})
            
            # Send
            result = self.gmail.users_messages_send("me", message)
            
            print(f"  [OK] Email sent: {result['id']}")
            
            return {
                "success": True,
                "message_id": result["id"],
                "to": to,
                "subject": subject
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_labels(self) -> Dict[str, Any]:
        """
        List Gmail labels.
        
        Returns:
            List of labels
        """
        print(f"\n    Listing labels")
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            result = self.gmail.users_labels_list("me")
            labels = result.get("labels", [])
            
            print(f"  [OK] Found {len(labels)} labels")
            
            return {
                "success": True,
                "labels": labels,
                "count": len(labels)
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get MCP tool definitions for AI agents."""
        tools = []
        
        if self.config.enable_search:
            tools.append({
                "name": "gmail_search_emails",
                "description": "Search emails by keywords, sender, subject, or content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'from:sarah@company.com', 'subject:Project Zeus')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        },
                        "label": {
                            "type": "string",
                            "description": "Filter by label (INBOX, SENT, etc.)",
                            "enum": ["INBOX", "SENT", "DRAFT", "IMPORTANT", "STARRED"]
                        }
                    },
                    "required": ["query"]
                }
            })
        
        if self.config.enable_read:
            tools.append({
                "name": "gmail_read_email",
                "description": "Read the full content of a specific email",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Gmail message ID"
                        }
                    },
                    "required": ["email_id"]
                }
            })
        
        if self.config.enable_send:
            tools.append({
                "name": "gmail_send_email",
                "description": "Send a new email",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body text"
                        },
                        "cc": {
                            "type": "string",
                            "description": "CC recipients (optional)"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            })
        
        tools.append({
            "name": "gmail_list_labels",
            "description": "List all Gmail labels (folders)",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        })
        
        return tools


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("GMAIL MCP SERVER DEMO")
    print("=" * 70)
    print("\nBased on Chapter 4: Model Context Protocol")
    print("Allows AI agents to search and manage Gmail\n")
    print("=" * 70)
    
    # Initialize server
    config = GmailConfig(
        enable_read=True,
        enable_send=True,  # Enabled for demo
        enable_search=True
    )
    
    server = GmailMCPServer(config)
    
    # Demo 1: Search emails
    print(f"\n{'='*70}")
    print("DEMO 1: Search for emails about Project Zeus")
    print('='*70)
    
    result = server.search_emails("Project Zeus")
    if result["success"]:
        print(f"\nFound {result['count']} emails:")
        for email in result["emails"]:
            print(f"\n    {email['subject']}")
            print(f"     From: {email['from']}")
            print(f"     Date: {email['date']}")
            print(f"     Preview: {email['snippet'][:60]}...")
    
    # Demo 2: Read specific email
    print(f"\n{'='*70}")
    print("DEMO 2: Read full email")
    print('='*70)
    
    result = server.read_email("email002")
    if result["success"]:
        print(f"\n  Email Details:")
        print(f"   From: {result['from']}")
        print(f"   To: {result['to']}")
        print(f"   Subject: {result['subject']}")
        print(f"\n   Body:")
        print("   " + " " * 66)
        for line in result['body'].split('\n'):
            print(f"   {line}")
        print("   " + " " * 66)
    
    # Demo 3: List labels
    print(f"\n{'='*70}")
    print("DEMO 3: List Gmail labels")
    print('='*70)
    
    result = server.list_labels()
    if result["success"]:
        print(f"\n    Labels ({result['count']}):")
        for label in result["labels"]:
            print(f"     {label['name']}")
    
    # Demo 4: Send email (simulated)
    print(f"\n{'='*70}")
    print("DEMO 4: Send email")
    print('='*70)
    
    result = server.send_email(
        to="team@company.com",
        subject="AI Agent Test Email",
        body="This is a test email sent by the AI agent via Gmail MCP server."
    )
    
    if result["success"]:
        print(f"\n[OK] Email sent successfully")
        print(f"  Message ID: {result['message_id']}")
        print(f"  To: {result['to']}")
        print(f"  Subject: {result['subject']}")
    
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
from gmail_mcp_server import GmailMCPServer, GmailConfig

# Initialize
server = GmailMCPServer(GmailConfig())

# Get tools for agent
tools = server.get_tool_definitions()

# When agent calls tool:
if tool_name == "gmail_search_emails":
    result = server.search_emails(args["query"])
elif tool_name == "gmail_read_email":
    result = server.read_email(args["email_id"])
elif tool_name == "gmail_send_email":
    result = server.send_email(args["to"], args["subject"], args["body"])

# Agent can now:
# - "Find all emails from Sarah about Project Zeus"
# - "Read the latest email from AlphaCorp"
# - "Send status update to the team"
# - "Search for budget approval emails from last month"
    """)


if __name__ == "__main__":
    demo()

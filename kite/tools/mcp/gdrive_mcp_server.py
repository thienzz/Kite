"""
Google Drive MCP Server Implementation
Based on Chapter 4: MCP (Model Context Protocol)

Allows AI agents to interact with Google Drive.

Tools provided:
- search_files: Search for files
- read_file: Read file content
- list_folder: List folder contents
- get_file_metadata: Get file details

Run: python gdrive_mcp_server.py
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GDriveConfig:
    """Configuration for Google Drive MCP server."""
    credentials_path: str = ""
    
    # Safety limits
    max_file_size_mb: int = 10
    max_search_results: int = 50
    rate_limit_per_minute: int = 100
    
    # Allowed operations
    enable_read: bool = True
    enable_search: bool = True
    enable_list: bool = True
    
    # File type restrictions
    allowed_mime_types: Optional[List[str]] = None


# ============================================================================
# GOOGLE DRIVE MCP SERVER
# ============================================================================

class GoogleDriveMCPServer:
    """
    MCP Server for Google Drive integration.
    
    Provides tools for AI agents to search and read Google Drive files.
    
    Requires: google-api-python-client, google-auth-httplib2, google-auth-oauthlib
    Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
    
    Setup:
        1. Enable Google Drive API in Google Cloud Console
        2. Download credentials.json
        3. Run authentication flow to get token
    
    Example:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        
        config = GDriveConfig(credentials_path='~/.gdrive_credentials.json')
        server = GoogleDriveMCPServer(config)
        
        # Search files
        results = server.search_files("Q4 strategy")
        
        # Read file
        content = server.read_file("file1")
    """
    
    def __init__(self, config: GDriveConfig = None, credentials_path: str = "", **kwargs):
        self.config = config or GDriveConfig()
        if credentials_path:
            self.config.credentials_path = credentials_path
        
        # Initialize Drive client (requires Google API client)
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            import os
            import pickle
            
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            
            creds = None
            token_path = os.path.expanduser('~/.gdrive_token.pickle')
            
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
                    raise Exception("Google Drive credentials not found. Please provide credentials_path")
                
                # Save token
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            self.drive = build('drive', 'v3', credentials=creds)
            
        except ImportError:
            raise ImportError(
                "Google API client is required for GoogleDriveMCPServer. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Google Drive client: {e}")

        
        # Rate limiting
        self.request_count = 0
        self.window_start = datetime.now()
        
        print(f"[OK] Google Drive MCP Server initialized")
        print(f"  Rate limit: {self.config.rate_limit_per_minute}/min")
        print(f"  Max file size: {self.config.max_file_size_mb}MB")
    
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
    
    def _is_mime_type_allowed(self, mime_type: str) -> bool:
        """Check if MIME type is allowed."""
        if self.config.allowed_mime_types is None:
            return True
        
        return mime_type in self.config.allowed_mime_types
    
    def search_files(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for files in Google Drive.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            Search results
        """
        print(f"\n  Searching Drive: {query}")
        
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
            results = self.drive.files_list(
                q=query,
                pageSize=min(max_results, self.config.max_search_results)
            )
            
            files = results.get("files", [])
            
            # Filter by allowed MIME types
            filtered_files = []
            for file in files:
                if self._is_mime_type_allowed(file.get("mimeType", "")):
                    filtered_files.append({
                        "id": file["id"],
                        "name": file["name"],
                        "mimeType": file["mimeType"],
                        "size": file.get("size", 0),
                        "modifiedTime": file.get("modifiedTime"),
                        "owners": file.get("owners", [])
                    })
            
            print(f"  [OK] Found {len(filtered_files)} files")
            
            return {
                "success": True,
                "query": query,
                "files": filtered_files,
                "count": len(filtered_files)
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def read_file(self, file_id: str) -> Dict[str, Any]:
        """
        Read file content from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            File content
        """
        print(f"\n  Reading file: {file_id}")
        
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
            # Get file metadata first
            metadata = self.drive.files_get(fileId=file_id)
            
            # Check MIME type
            if not self._is_mime_type_allowed(metadata.get("mimeType", "")):
                return {
                    "success": False,
                    "error": f"MIME type not allowed: {metadata.get('mimeType')}"
                }
            
            # Check file size
            size_mb = metadata.get("size", 0) / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                return {
                    "success": False,
                    "error": f"File too large: {size_mb:.1f}MB > {self.config.max_file_size_mb}MB"
                }
            
            # Read content
            content = self.drive.files_get_media(fileId=file_id)
            
            print(f"  [OK] Read {len(content)} characters")
            
            return {
                "success": True,
                "file_id": file_id,
                "name": metadata.get("name"),
                "content": content,
                "metadata": {
                    "mimeType": metadata.get("mimeType"),
                    "size": metadata.get("size"),
                    "modifiedTime": metadata.get("modifiedTime"),
                    "owners": metadata.get("owners", [])
                }
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get file metadata without reading content.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            File metadata
        """
        print(f"\n[CHART] Getting metadata: {file_id}")
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            metadata = self.drive.files_get(fileId=file_id)
            
            print(f"  [OK] Retrieved metadata for: {metadata.get('name')}")
            
            return {
                "success": True,
                "file_id": file_id,
                "metadata": {
                    "name": metadata.get("name"),
                    "mimeType": metadata.get("mimeType"),
                    "size": metadata.get("size"),
                    "createdTime": metadata.get("createdTime"),
                    "modifiedTime": metadata.get("modifiedTime"),
                    "owners": metadata.get("owners", []),
                    "parents": metadata.get("parents", [])
                }
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
                "name": "gdrive_search_files",
                "description": "Search for files in Google Drive by keyword or content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords, file names, content)"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            })
        
        if self.config.enable_read:
            tools.append({
                "name": "gdrive_read_file",
                "description": "Read the content of a Google Drive file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Google Drive file ID"
                        }
                    },
                    "required": ["file_id"]
                }
            })
        
        tools.append({
            "name": "gdrive_get_metadata",
            "description": "Get metadata about a Google Drive file without reading its content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Google Drive file ID"
                    }
                },
                "required": ["file_id"]
            }
        })
        
        return tools


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("GOOGLE DRIVE MCP SERVER DEMO")
    print("=" * 70)
    print("\nBased on Chapter 4: Model Context Protocol")
    print("Allows AI agents to search and read Google Drive files\n")
    print("=" * 70)
    
    # Initialize server
    config = GDriveConfig(
        max_file_size_mb=10,
        enable_read=True,
        enable_search=True
    )
    
    server = GDriveMCPServer(config)
    
    # Demo 1: Search files
    print(f"\n{'='*70}")
    print("DEMO 1: Search for files")
    print('='*70)
    
    result = server.search_files("Project Zeus")
    if result["success"]:
        print(f"\nFound {result['count']} files matching '{result['query']}':")
        for file in result["files"]:
            print(f"\n    {file['name']}")
            print(f"     ID: {file['id']}")
            print(f"     Type: {file['mimeType']}")
            print(f"     Modified: {file.get('modifiedTime', 'N/A')}")
    
    # Demo 2: Read file content
    print(f"\n{'='*70}")
    print("DEMO 2: Read file content")
    print('='*70)
    
    result = server.read_file("file1")
    if result["success"]:
        print(f"\n  File: {result['name']}")
        print(f"   Size: {result['metadata']['size']} bytes")
        print(f"\n   Content preview:")
        print("   " + " " * 66)
        preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
        for line in preview.split('\n'):
            print(f"   {line}")
        print("   " + " " * 66)
    
    # Demo 3: Get metadata only
    print(f"\n{'='*70}")
    print("DEMO 3: Get file metadata")
    print('='*70)
    
    result = server.get_file_metadata("file2")
    if result["success"]:
        meta = result["metadata"]
        print(f"\n[CHART] Metadata for: {meta['name']}")
        print(f"   Type: {meta['mimeType']}")
        print(f"   Size: {meta['size']} bytes")
        print(f"   Created: {meta['createdTime']}")
        print(f"   Modified: {meta['modifiedTime']}")
        print(f"   Owners: {', '.join(o['displayName'] for o in meta['owners'])}")
    
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
from gdrive_mcp_server import GDriveMCPServer, GDriveConfig

# Initialize
server = GDriveMCPServer(GDriveConfig())

# Get tools for agent
tools = server.get_tool_definitions()

# When agent calls tool:
if tool_name == "gdrive_search_files":
    result = server.search_files(args["query"])
elif tool_name == "gdrive_read_file":
    result = server.read_file(args["file_id"])
elif tool_name == "gdrive_get_metadata":
    result = server.get_file_metadata(args["file_id"])

# Agent can now:
# - Search: "Find our Q4 strategy document"
# - Read: "What does the AlphaCorp agreement say about IP?"
# - Analyze: "Summarize all engineering documents from last month"
    """)


if __name__ == "__main__":
    demo()

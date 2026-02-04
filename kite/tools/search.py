"""
Standard Web Search Tool using DuckDuckGo Lite.
"""
import requests
import asyncio
import re
import time
from typing import Dict, Any

class WebSearchTool:
    """Real web search using DuckDuckGo Lite (No API key required)."""
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the real web for current information. Returns snippets."
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
    
    def get_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
        
    def to_schema(self):
        return self.get_definition()
        
    async def execute(self, query: str = None, **kwargs):
        if not query:
            return "Error: Please provide a search query."
            
        print(f"      [Tool] Real Web Search for: '{query}'...")
        await asyncio.sleep(1) # Be polite (Async sleep)
        
        try:
            # DuckDuckGo Lite (HTML version)
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query}
            
            # Using asyncio toThread for blocking request if needed, but requests is sync.
            # In a real async framework we might use aiohttp, but here we wrap or accept block for now.
            # Or use run_in_executor to avoid blocking the loop
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: self.session.post(url, data=data, timeout=10))
            
            if resp.status_code != 200:
                return f"Error: Search failed with status {resp.status_code}"
                
            # Regex scrape (Simple but effective for DDG Lite)
            results = []
            
            # Extract result blocks (approximate)
            # Try BS4 if available, else regex
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = soup.find_all('a', class_='result__a', limit=5)
                for link in links:
                    title = link.get_text(strip=True)
                    href = link.get('href')
                    results.append(f"Title: {title}\nURL: {href}")
            except ImportError:
                 # Fallback regex
                 titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text)
                 for i in range(min(len(titles), 5)):
                     results.append(f"Title: {titles[i]}")
            
            if not results:
                # Regex fallback if BS4 logic missed or wasn't used
                titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text)
                for i in range(min(len(titles), 5)):
                    results.append(f"Title: {titles[i]}")

            output = "\n---\n".join(results[:5])
            if not output: return "No results found."
            return output

        except Exception as e:
            return f"Search error: {e}"

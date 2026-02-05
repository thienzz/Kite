"""
LLM-based Router Implementation
Uses LLM/SLM to classify user intent with reasoning.
More accurate than embeddings but slower and more expensive.
"""

import json
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class LLMRoute:
    name: str
    description: str
    handler: Callable

class LLMRouter:
    """
    Routes user queries using LLM classification.
    """
    
    def __init__(self, llm=None):
        self.routes: Dict[str, LLMRoute] = {}
        self.llm = llm
        
    def add_route(self, name: str, examples: List[str] | str = None, samples: List[str] | str = None, description: str = "", handler: Callable = None):
        """Add a new route. Examples serve as context. 'samples' is an alias for 'examples'."""
        final_examples = examples or samples
        self.routes[name] = LLMRoute(
            name=name,
            description=description or f"Handle queries related to {name}",
            handler=handler
        )
        print(f"[OK] Added LLM route: {name}")

    async def route(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Route query to appropriate specialist agent using LLM."""
        if not self.routes:
            raise RuntimeError("No routes configured in LLMRouter")

        # Prepare prompt
        routes_desc = ""
        for route in self.routes.values():
            routes_desc += f"- {route.name}: {route.description}\n"

        prompt = f"""Classify the user query into one of the following categories.
Available Categories:
{routes_desc}
- none: Use this if the query doesn't fit any of the above.

User Query: "{query}"

Respond ONLY with a JSON object:
{{"category": "category_name", "confidence": 0.0-1.0, "reasoning": "why?"}}"""

        try:
            # Use LLM for classification
            response = await asyncio.to_thread(self.llm.complete, prompt, temperature=0.1)
            
            # Clean and parse JSON
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[-1].split("```")[0].strip()
            
            # Robust JSON parsing with fallback
            try:
                # Try to find JSON in text
                import re
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                
                data = json.loads(content)
                category = data.get("category", "none")
                confidence = data.get("confidence", 0.0)
                reasoning = data.get("reasoning", "No reasoning")
                
            except (json.JSONDecodeError, AttributeError) as e:
                # Fallback: keyword matching
                print(f"[WARN] Router JSON parse failed creating fallback")
                content_lower = response.lower()
                category = "none"
                confidence = 0.5
                reasoning = "Fallback text classification"
                
                # Check for route keywords
                for route_name in self.routes.keys():
                    if route_name.replace("_", " ") in content_lower:
                        category = route_name
                        break
            print(f"\n  LLM Intent Classification:")
            print(f"   Query: {query}")
            print(f"   Category: {category} (confidence: {confidence:.0%})")
            print(f"   Reasoning: {reasoning}")

            if category == "none" or category not in self.routes:
                return {
                    "route": "none",
                    "confidence": confidence,
                    "response": "I'm not sure how to help with that. Could you be more specific?",
                    "needs_clarification": True
                }

            # Execute handler
            route = self.routes[category]
            print(f"[OK] Routing to {route.name}")
            
            # Use context if provided
            try:
                resp = route.handler(query, context)
            except TypeError:
                 # Fallback for handlers that don't accept context
                resp = route.handler(query)

            if asyncio.iscoroutine(resp):
                resp = await resp
            
            # Extract response text
            if isinstance(resp, dict) and 'response' in resp:
                response_text = resp['response']
            else:
                response_text = str(resp)

            return {
                "route": route.name,
                "confidence": confidence,
                "response": response_text,
                "needs_clarification": False
            }

        except Exception as e:
            print(f"[ERROR] LLM Routing failed: {e}")
            return {
                "route": "error",
                "confidence": 0,
                "response": f"Routing error: {str(e)}",
                "needs_clarification": False
            }

    def get_stats(self) -> Dict:
        return {
            "total_routes": len(self.routes),
            "confidence_threshold": 0.0,
            "cache_hit_rate": 0.0,
            "type": "LLM"
        }

"""
Semantic Router Implementation
Based on Chapter 6: The Semantic Router

A router that classifies user intent and routes to specialist agents.
Uses embeddings for fast, cached classification.

Run: python semantic_router.py
"""

import os
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# ROUTING CONFIGURATION
# ============================================================================

@dataclass
class Route:
    """Define a route with examples and handler."""
    name: str
    description: str
    examples: List[str]
    handler: Callable
    embedding: Optional[np.ndarray] = None


# ============================================================================
# SPECIALIST AGENTS (Mock implementations)
# ============================================================================

class TechnicalSupportAgent:
    """Handles technical issues."""
    
    def handle(self, query: str) -> str:
        print(f"    Technical Support Agent handling: {query[:50]}...")
        return f"Technical Support: I can help with that. Let me check the system logs..."


class BillingAgent:
    """Handles billing and refunds."""
    
    def handle(self, query: str) -> str:
        print(f"    Billing Agent handling: {query[:50]}...")
        return f"Billing: I'll look into your account and billing history..."


class ProductInfoAgent:
    """Handles product questions."""
    
    def handle(self, query: str) -> str:
        print(f"    Product Info Agent handling: {query[:50]}...")
        return f"Product Info: Let me find information about that product..."


class GeneralAgent:
    """Fallback for unclear queries."""
    
    def handle(self, query: str) -> str:
        print(f"  [CHAT] General Agent handling: {query[:50]}...")
        return f"I can help with that. Could you provide more details?"


# ============================================================================
# SEMANTIC ROUTER
# ============================================================================

class SemanticRouter:
    """
    Routes user queries to specialist agents based on semantic similarity.
    
    Features:
    - Embedding-based classification (fast, cheap)
    - LRU caching for repeated queries
    - Confidence thresholds
    - Ambiguity handling
    
    Example:
        router = SemanticRouter()
        router.add_route("technical", tech_agent.handle, [
            "my internet is down",
            "can't connect to wifi",
            "server not responding"
        ])
        
        response = router.route("my connection keeps dropping")
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.75,
                 embedding_provider = None):
        self.routes: List[Route] = []
        self.confidence_threshold = confidence_threshold
        self.embedding_provider = embedding_provider
        
        # Initialize specialist agents
        self.tech_agent = TechnicalSupportAgent()
        self.billing_agent = BillingAgent()
        self.product_agent = ProductInfoAgent()
        self.general_agent = GeneralAgent()
        
        # Setup default routes
        self._setup_default_routes()
        
        # Precompute route embeddings
        self._compute_route_embeddings()
    
    def _setup_default_routes(self):
        """Define default routing rules."""
        
        # Technical Support route
        self.routes.append(Route(
            name="technical_support",
            description="Technical issues, connectivity, errors",
            examples=[
                "my internet is down",
                "can't connect to wifi",
                "server not responding",
                "error message when logging in",
                "app keeps crashing",
                "slow connection",
                "website won't load"
            ],
            handler=self.tech_agent.handle
        ))
        
        # Billing route
        self.routes.append(Route(
            name="billing",
            description="Billing, refunds, charges, invoices",
            examples=[
                "I want a refund",
                "wrong charge on my card",
                "didn't authorize this payment",
                "cancel my subscription",
                "invoice is incorrect",
                "double charged",
                "need receipt"
            ],
            handler=self.billing_agent.handle
        ))
        
        # Product Info route
        self.routes.append(Route(
            name="product_info",
            description="Product features, specifications, availability",
            examples=[
                "do you sell laptops",
                "what's the price of",
                "product specifications",
                "is this in stock",
                "compare models",
                "which version should I buy",
                "product features"
            ],
            handler=self.product_agent.handle
        ))
    
    @lru_cache(maxsize=10000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with LRU caching."""
        if self.embedding_provider:
            return np.array(self.embedding_provider.embed(text))
            
        raise RuntimeError("No embedding provider configured for SemanticRouter")
    
    def _compute_route_embeddings(self):
        """Precompute embeddings for all route examples."""
        print("[CHART] Computing route embeddings...")
        
        for route in self.routes:
            # Combine all examples into one text for the route
            combined_text = " | ".join(route.examples)
            route.embedding = self._get_embedding(combined_text)
        
        print(f"[OK] Computed embeddings for {len(self.routes)} routes")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def classify_intent(self, query: str) -> tuple[Route, float]:
        """
        Classify query intent using semantic similarity.
        
        Args:
            query: User query
            
        Returns:
            (best_route, confidence_score)
        """
        # Get query embedding (cached if seen before)
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity to each route
        scores = []
        for route in self.routes:
            similarity = self._cosine_similarity(query_embedding, route.embedding)
            scores.append((route, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_route, best_score = scores[0]
        
        print(f"\n  Intent Classification:")
        print(f"   Query: {query}")
        print(f"   Best Match: {best_route.name} (confidence: {best_score:.2%})")
        
        # Show runner-ups
        if len(scores) > 1:
            print(f"   Runner-up: {scores[1][0].name} ({scores[1][1]:.2%})")
        
        return best_route, best_score
    
    def route(self, query: str) -> Dict:
        """
        Route query to appropriate specialist agent.
        
        Args:
            query: User query
            
        Returns:
            Response dictionary with routing info
        """
        # Classify intent
        route, confidence = self.classify_intent(query)
        
        # Handle low confidence (ambiguity)
        if confidence < self.confidence_threshold:
            print(f"[WARN]  Low confidence ({confidence:.2%} < {self.confidence_threshold:.2%})")
            print(f"   Routing to General Agent for clarification")
            
            response = self.general_agent.handle(query)
            
            return {
                "route": "general",
                "confidence": confidence,
                "response": response,
                "needs_clarification": True,
                "suggested_routes": [r.name for r, _ in self._get_top_routes(query, n=3)]
            }
        
        # Route to specialist
        print(f"[OK] Routing to {route.name}")
        response = route.handler(query)
        
        return {
            "route": route.name,
            "confidence": confidence,
            "response": response,
            "needs_clarification": False
        }
    
    def _get_top_routes(self, query: str, n: int = 3) -> List[tuple[Route, float]]:
        """Get top N routes by similarity."""
        query_embedding = self._get_embedding(query)
        
        scores = [
            (route, self._cosine_similarity(query_embedding, route.embedding))
            for route in self.routes
        ]
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
    
    def add_route(self, name: str, examples: List[str] | str, description: str = "", handler: Callable = None):
        """Add a new route dynamically."""
        if isinstance(examples, str):
            examples = [examples]
            
        route = Route(
            name=name,
            description=description,
            examples=examples,
            handler=handler or (lambda x: f"Default response for {name}")
        )
        
        # Compute embedding
        combined_text = " | ".join(examples)
        route.embedding = self._get_embedding(combined_text)
        
        self.routes.append(route)
        print(f"[OK] Added route: {name}")
    
    def get_stats(self) -> Dict:
        """Get router statistics."""
        cache_info = self._get_embedding.cache_info()
        
        return {
            "total_routes": len(self.routes),
            "confidence_threshold": self.confidence_threshold,
            "cache_size": cache_info.currsize,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_hit_rate": (
                cache_info.hits / (cache_info.hits + cache_info.misses)
                if (cache_info.hits + cache_info.misses) > 0
                else 0
            )
        }


# ============================================================================
# DEMO
# ============================================================================

def main():
    print("=" * 70)
    print("SEMANTIC ROUTER DEMO")
    print("=" * 70)
    print("\nFeatures:")
    print("    Embedding-based classification (fast, cheap)")
    print("    LRU caching (10,000 queries)")
    print("    Confidence thresholds")
    print("    Ambiguity handling")
    print("=" * 70)
    
    # Initialize router
    router = SemanticRouter(confidence_threshold=0.75)
    
    # Test queries
    test_queries = [
        # Clear technical
        "my internet keeps disconnecting",
        "can't log into the app",
        
        # Clear billing
        "I was charged twice",
        "need a refund for order 12345",
        
        # Clear product
        "do you have wireless keyboards",
        "what's the price of the pro model",
        
        # Ambiguous
        "I have a problem",
        "help me",
        
        # Variation (should use cache)
        "my internet keeps disconnecting",  # Repeat
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print('='*70)
        
        result = router.route(query)
        
        print(f"\n  Response:")
        print(f"   {result['response']}")
        
        if result['needs_clarification']:
            print(f"\n  Suggested routes: {', '.join(result['suggested_routes'])}")
    
    # Show stats
    print(f"\n{'='*70}")
    print("ROUTER STATISTICS")
    print('='*70)
    stats = router.get_stats()
    print(f"Total routes: {stats['total_routes']}")
    print(f"Confidence threshold: {stats['confidence_threshold']:.0%}")
    print(f"Cache size: {stats['cache_size']:,} embeddings")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"    Hits: {stats['cache_hits']:,}")
    print(f"    Misses: {stats['cache_misses']:,}")
    
    print(f"\n  The repeat query used cached embedding (instant, free!)")
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print("""
Approach 1: Generalist GPT-4
    Latency: 2-4 seconds
    Cost: $0.03 per request
    Handles: All queries (but slower)

Approach 2: Semantic Router + Specialists
    Router latency: 50ms (with cache: ~0ms)
    Specialist latency: 200-500ms (SLM)
    Total: 250-550ms
    Cost: $0.001 per request (30x cheaper!)
    Handles: 10,000 requests/second
    """)


if __name__ == "__main__":
    main()

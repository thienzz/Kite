"""
Semantic Router Implementation
Based on Chapter 6: The Semantic Router

A router that classifies user intent and routes to specialists.
Uses embeddings for fast, cached classification.
"""

import os
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from dotenv import load_dotenv

load_dotenv(".kite.env")


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
        router = SemanticRouter(embedding_provider=my_embedder)
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
        
        # Precompute route embeddings if routes exist (none initially)
        self._compute_route_embeddings()
    
    @lru_cache(maxsize=10000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with LRU caching."""
        if self.embedding_provider:
            return np.array(self.embedding_provider.embed(text))
            
        raise RuntimeError("No embedding provider configured for SemanticRouter")
    
    def _compute_route_embeddings(self):
        """Precompute embeddings for all route examples."""
        if not self.routes:
            return

        print("[CHART] Computing route embeddings...")
        for route in self.routes:
            if route.embedding is None:
                # Combine all examples into one text for the route
                combined_text = " | ".join(route.examples)
                route.embedding = self._get_embedding(combined_text)
    
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
        if not self.routes:
            raise RuntimeError("No routes configured in SemanticRouter")

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
            
            return {
                "route": "none",
                "confidence": confidence,
                "response": "I'm not sure how to help with that. Could you be more specific?",
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
        
        # Compute embedding if provider exists
        if self.embedding_provider:
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

"""
Intelligent Caching System
Reduces latency and costs by caching LLM responses.
"""

import time
import json
import hashlib
from typing import Dict, Any, Optional, List
from ..memory.vector_memory import VectorMemory

class CacheEntry:
    def __init__(self, value: Any, ttl: int = 3600):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        if self.ttl == -1: return False
        return time.time() > self.timestamp + self.ttl

class SemanticCache:
    """
    Caches responses based on query similarity.
    """
    def __init__(self, vector_memory: VectorMemory, threshold: float = 0.7):
        self.memory = vector_memory
        self.threshold = threshold # Lowered for better hit rate in demo
        self.cache = {} # id -> CacheEntry

    def _get_query_id(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        # 1. Exact match first
        query_id = self._get_query_id(query)
        if query_id in self.cache:
            entry = self.cache[query_id]
            if not entry.is_expired():
                print(f"[OK] Exact cache hit: {query[:30]}...")
                return entry.value
            else:
                del self.cache[query_id]

        # 2. Semantic match
        results = self.memory.search(query, k=1)
        if results:
            doc_id, text, distance = results[0]
            # Strip chunk suffix if present
            base_doc_id = doc_id.split('_chunk_')[0]
            
            similarity = 1 - distance
            
            if similarity >= self.threshold:
                # Retrieve from cache using base_doc_id
                if base_doc_id in self.cache:
                    entry = self.cache[base_doc_id]
                    if not entry.is_expired():
                        print(f"[OK] Semantic cache hit ({similarity:.2f}): {query[:30]}...")
                        return entry.value
        
        return None

    def set(self, query: str, value: Any, ttl: int = 3600):
        query_id = self._get_query_id(query)
        self.cache[query_id] = CacheEntry(value, ttl)
        
        # Also index in vector memory for semantic search
        self.memory.add_document(query_id, query, metadata={"type": "cache_query"})
        print(f"[OK] Cached: {query[:30]}...")

class CacheManager:
    """
    Facade for multiple caching strategies.
    """
    def __init__(self, vector_memory: Optional[VectorMemory] = None):
        self.semantic_cache = SemanticCache(vector_memory) if vector_memory else None
        self.memory_cache = {} # Simple in-memory KV cache

    def get(self, key: str, semantic: bool = False) -> Optional[Any]:
        if semantic and self.semantic_cache:
            return self.semantic_cache.get(key)
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.value
            del self.memory_cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 3600, semantic: bool = False):
        if semantic and self.semantic_cache:
            self.semantic_cache.set(key, value, ttl)
        else:
            self.memory_cache[key] = CacheEntry(value, ttl)

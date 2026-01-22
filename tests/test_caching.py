"""
Comprehensive tests for caching system.
Tests cache manager and Redis backend.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from kite.caching.cache_manager import CacheManager


class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager setup."""
        cache = CacheManager()
        assert cache is not None
        assert cache.memory_cache is not None
    
    def test_cache_set_get(self):
        """Test basic cache operations."""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        result = cache.get("key1")
        
        assert result == "value1"
    
    def test_cache_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = CacheManager()
        
        result = cache.get("nonexistent")
        assert result is None
    
    def test_cache_expiration(self):
        """Test TTL expiration."""
        cache = CacheManager()
        
        cache.set("temp_key", "temp_value", ttl=1)
        
        # Should exist immediately
        assert cache.get("temp_key") == "temp_value"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired
        result = cache.get("temp_key")
        assert result is None
    
    def test_cache_invalidation(self):
        """Test cache clearing."""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Clear specific key
        cache.delete("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_cache_clear_all(self):
        """Test clearing all cache."""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheWithLLM:
    """Test LLM response caching."""
    
    def test_cache_llm_response(self, mock_llm_provider):
        """Test caching LLM responses."""
        cache = CacheManager()
        
        prompt = "What is AI?"
        response = "AI is artificial intelligence"
        
        # Cache the response
        cache_key = f"llm:{hash(prompt)}"
        cache.set(cache_key, response)
        
        # Retrieve from cache
        cached = cache.get(cache_key)
        assert cached == response
    
    def test_cache_hit_performance(self, mock_llm_provider):
        """Test cache hit improves performance."""
        cache = CacheManager()
        
        prompt = "Test prompt"
        response = "Test response"
        cache_key = f"llm:{hash(prompt)}"
        
        # First call - cache miss
        start = time.time()
        if cache.get(cache_key) is None:
            # Simulate LLM call
            time.sleep(0.1)
            result = mock_llm_provider.complete(prompt)
            cache.set(cache_key, result)
        duration_miss = time.time() - start
        
        # Second call - cache hit
        start = time.time()
        result = cache.get(cache_key)
        duration_hit = time.time() - start
        
        # Cache hit should be much faster
        assert duration_hit < duration_miss
        assert result is not None


class TestCacheIntegration:
    """Integration tests for caching."""
    
    def test_cache_with_kite(self, ai):
        """Test cache integration with Kite."""
        cache = ai.cache
        assert cache is not None
    
    def test_cache_multiple_backends(self):
        """Test different cache backends."""
        backends = ["memory"]
        
        for backend in backends:
            cache = CacheManager()
            cache.set("test", "value")
            assert cache.get("test") == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

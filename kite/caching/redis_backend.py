"""
Redis Backend for CacheManager
"""

import redis
import json
import time
from typing import Any, Optional

class RedisCache:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            print(f"[OK] Connected to Redis at {host}:{port}")
        except Exception as e:
            print(f"  Failed to connect to Redis: {e}")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.client: return None
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        if not self.client: return
        self.client.set(key, json.dumps(value), ex=ttl)

    def delete(self, key: str):
        if not self.client: return
        self.client.delete(key)

    def clear(self):
        if not self.client: return
        self.client.flushdb()

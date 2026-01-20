"""
Idempotency Manager for AI Agent Operations

Ensures that operations are executed exactly once, even if called multiple times.
Critical for preventing duplicate actions like refunds, emails, database writes.

Author: Agentic AI Systems
License: MIT
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyConfig:
    """Configuration for idempotency behavior."""
    ttl_seconds: int = 3600  # How long to remember operations (1 hour default)
    storage_backend: str = "memory"  # Options: memory, redis, database


class IdempotencyManager:
    """
    Manages idempotency for agent operations.
    
    Use this to ensure operations like refunds, emails, or database
    writes only execute once, even if the agent retries.
    
    Example:
        manager = IdempotencyManager()
        
        # Generate idempotency key
        key = manager.generate_id(
            operation="refund",
            params={"order_id": "123", "amount": 99.99}
        )
        
        # Check if already executed
        if manager.is_duplicate(key):
            return manager.get_result(key)
        
        # Execute operation
        result = process_refund(...)
        manager.store_result(key, result)
    """
    
    def __init__(self, config: Optional[IdempotencyConfig] = None):
        self.config = config or IdempotencyConfig()
        self._storage: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Idempotency manager initialized: {self.config}")
    
    def generate_id(self, operation: str, params: Dict[str, Any]) -> str:
        """
        Generate deterministic idempotency key.
        
        The same operation + params will always produce the same key.
        This is critical for detecting duplicates.
        
        Args:
            operation: Name of the operation (e.g., "refund", "send_email")
            params: Parameters that uniquely identify this operation
            
        Returns:
            Deterministic hash string
        """
        # Sort params for deterministic hashing
        sorted_params = json.dumps(params, sort_keys=True)
        unique_string = f"{operation}:{sorted_params}"
        
        # Generate hash
        return hashlib.sha256(unique_string.encode()).hexdigest()
    
    def is_duplicate(self, idempotency_key: str) -> bool:
        """
        Check if this operation has already been executed.
        
        Args:
            idempotency_key: The idempotency key from generate_id()
            
        Returns:
            True if operation already executed, False otherwise
        """
        if idempotency_key not in self._storage:
            return False
        
        # Check if TTL expired
        entry = self._storage[idempotency_key]
        expiry = entry["expires_at"]
        
        if datetime.now() > expiry:
            # Expired, remove and return False
            del self._storage[idempotency_key]
            logger.info(f"Idempotency key expired and removed: {idempotency_key}")
            return False
        
        logger.warning(f"Duplicate operation detected: {idempotency_key}")
        return True
    
    def get_result(self, idempotency_key: str) -> Optional[Any]:
        """
        Get the cached result of a previous execution.
        
        Args:
            idempotency_key: The idempotency key
            
        Returns:
            Cached result or None if not found
        """
        if idempotency_key not in self._storage:
            return None
        
        entry = self._storage[idempotency_key]
        
        # Check expiry
        if datetime.now() > entry["expires_at"]:
            del self._storage[idempotency_key]
            return None
        
        logger.info(f"Returning cached result for key: {idempotency_key}")
        return entry["result"]
    
    def store_result(
        self,
        idempotency_key: str,
        result: Any,
        ttl_seconds: Optional[int] = None
    ):
        """
        Store the result of an operation.
        
        Args:
            idempotency_key: The idempotency key
            result: The result to cache
            ttl_seconds: Override default TTL
        """
        ttl = ttl_seconds or self.config.ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self._storage[idempotency_key] = {
            "result": result,
            "created_at": datetime.now(),
            "expires_at": expires_at
        }
        
        logger.info(
            f"Stored result for key: {idempotency_key}, "
            f"expires: {expires_at.isoformat()}"
        )
    
    def clear_expired(self):
        """Remove all expired entries (cleanup)."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._storage.items()
            if entry["expires_at"] < now
        ]
        
        for key in expired_keys:
            del self._storage[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired entries")
    
    def clear_all(self):
        """Clear all cached results."""
        self._storage.clear()
        logger.info("All idempotency cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached operations."""
        now = datetime.now()
        active = sum(
            1 for entry in self._storage.values()
            if entry["expires_at"] > now
        )
        
        return {
            "total_cached": len(self._storage),
            "active": active,
            "expired": len(self._storage) - active
        }


# Decorator for easy use
def idempotent(manager: IdempotencyManager, operation_name: str):
    """
    Decorator to make a function idempotent.
    
    Example:
        manager = IdempotencyManager()
        
        @idempotent(manager, "process_refund")
        def process_refund(order_id: str, amount: float):
            # This will only execute once for the same order_id + amount
            return stripe.Refund.create(...)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate idempotency key from function args
            params = {
                "args": str(args),
                "kwargs": str(kwargs)
            }
            key = manager.generate_id(operation_name, params)
            
            # Check if already executed
            if manager.is_duplicate(key):
                logger.info(f"Returning cached result for {operation_name}")
                return manager.get_result(key)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            manager.store_result(key, result)
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=== Idempotency Manager Demo ===\n")
    
    manager = IdempotencyManager(
        config=IdempotencyConfig(ttl_seconds=60)
    )
    
    # Simulate a refund operation
    def process_refund(order_id: str, amount: float):
        """Simulate processing a refund."""
        print(f"    Processing refund: ${amount} for order {order_id}")
        time.sleep(0.5)  # Simulate API call
        return {
            "success": True,
            "refund_id": f"ref_{int(time.time())}",
            "amount": amount
        }
    
    # Test idempotency
    print("1. First refund request:")
    key1 = manager.generate_id(
        "refund",
        {"order_id": "12345", "amount": 299.99}
    )
    
    if not manager.is_duplicate(key1):
        result1 = process_refund("12345", 299.99)
        manager.store_result(key1, result1)
        print(f"  [OK] Result: {result1}\n")
    
    print("2. Duplicate refund request (should use cache):")
    key2 = manager.generate_id(
        "refund",
        {"order_id": "12345", "amount": 299.99}
    )
    
    if manager.is_duplicate(key2):
        cached = manager.get_result(key2)
        print(f"    Cached result: {cached}\n")
    
    print("3. Different refund (should execute):")
    key3 = manager.generate_id(
        "refund",
        {"order_id": "67890", "amount": 199.99}
    )
    
    if not manager.is_duplicate(key3):
        result3 = process_refund("67890", 199.99)
        manager.store_result(key3, result3)
        print(f"  [OK] Result: {result3}\n")
    
    # Stats
    print("Statistics:")
    print(f"  {manager.get_stats()}\n")
    
    # Decorator example
    print("\n=== Decorator Example ===\n")
    
    @idempotent(manager, "send_email")
    def send_email(to: str, subject: str):
        print(f"    Sending email to {to}: {subject}")
        return {"sent": True, "message_id": f"msg_{int(time.time())}"}
    
    # First call - executes
    print("1. First email:")
    result = send_email("user@example.com", "Welcome!")
    print(f"  Result: {result}\n")
    
    # Duplicate call - cached
    print("2. Duplicate email (cached):")
    result = send_email("user@example.com", "Welcome!")
    print(f"  Result: {result}\n")

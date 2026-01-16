"""
Circuit Breaker Pattern for AI Agent Operations

Prevents cascading failures when an agent repeatedly attempts failed operations.
This is critical for write-access AI systems where retries can cause real damage.

Author: [Your Name]
License: MIT
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Callable, Any
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 3          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: int = 60           # Time before attempting recovery
    half_open_max_calls: int = 1        # Max concurrent calls in half-open


@dataclass
class CircuitStats:
    """Statistics for monitoring circuit breaker health."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for AI agent operations.
    
    Use this to wrap any operation that:
    - Makes external API calls
    - Modifies state
    - Costs money
    - Could fail repeatedly
    
    Example:
        circuit_breaker = CircuitBreaker(
            name="stripe_refunds",
            config=CircuitBreakerConfig(failure_threshold=3)
        )
        
        @circuit_breaker.protected
        def process_refund(order_id: str, amount: float):
            return stripe.Refund.create(...)
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        
        # Callbacks for state changes
        self.on_open = on_open
        self.on_close = on_close
        
        # Failure tracking
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.open_until: Optional[datetime] = None
        self.half_open_calls = 0
        
        logger.info(f"Circuit breaker '{name}' initialized: {config}")
    
    def _change_state(self, new_state: CircuitState, reason: str):
        """Change circuit state and trigger callbacks."""
        if new_state == self.state:
            return
            
        old_state = self.state
        self.state = new_state
        self.stats.state_changes += 1
        
        logger.warning(
            f"Circuit '{self.name}': {old_state.value}   {new_state.value} "
            f"(Reason: {reason})"
        )
        
        if new_state == CircuitState.OPEN and self.on_open:
            self.on_open(self.name, self.stats)
        elif new_state == CircuitState.CLOSED and self.on_close:
            self.on_close(self.name, self.stats)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.state != CircuitState.OPEN:
            return False
            
        if self.open_until and datetime.now() >= self.open_until:
            return True
        
        return False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result from function
            
        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails
        """
        self.stats.total_calls += 1
        
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self._should_attempt_reset():
            self._change_state(CircuitState.HALF_OPEN, "Timeout expired")
            self.half_open_calls = 0
        
        # Block calls if circuit is OPEN
        if self.state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            raise CircuitBreakerError(
                f"Circuit '{self.name}' is OPEN. "
                f"Reset at {self.open_until}"
            )
        
        # Limit concurrent calls in HALF_OPEN state
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.stats.rejected_calls += 1
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is HALF_OPEN and at capacity"
                )
            self.half_open_calls += 1
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
        
        finally:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls -= 1
    
    def _on_success(self):
        """Handle successful function call."""
        self.stats.successful_calls += 1
        self.stats.last_success_time = datetime.now()
        self.consecutive_failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes += 1
            
            if self.consecutive_successes >= self.config.success_threshold:
                self._change_state(
                    CircuitState.CLOSED,
                    f"{self.consecutive_successes} consecutive successes"
                )
                self.consecutive_successes = 0
        
        logger.debug(f"Circuit '{self.name}': Call succeeded")
    
    def _on_failure(self):
        """Handle failed function call."""
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.now()
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        
        logger.warning(
            f"Circuit '{self.name}': Call failed "
            f"({self.consecutive_failures}/{self.config.failure_threshold})"
        )
        
        # Open circuit if threshold exceeded
        if self.consecutive_failures >= self.config.failure_threshold:
            self.open_until = (
                datetime.now() + 
                timedelta(seconds=self.config.timeout_seconds)
            )
            self._change_state(
                CircuitState.OPEN,
                f"{self.consecutive_failures} consecutive failures"
            )
    
    def protected(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.
        
        Example:
            @circuit_breaker.protected
            def risky_operation():
                return external_api.call()
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def reset(self):
        """Manually reset circuit to closed state."""
        self._change_state(CircuitState.CLOSED, "Manual reset")
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.open_until = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "rejected_calls": self.stats.rejected_calls,
            "success_rate": (
                self.stats.successful_calls / self.stats.total_calls
                if self.stats.total_calls > 0 else 0
            ),
            "consecutive_failures": self.consecutive_failures,
            "open_until": self.open_until.isoformat() if self.open_until else None,
            "last_failure": (
                self.stats.last_failure_time.isoformat() 
                if self.stats.last_failure_time else None
            )
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker blocks an operation."""
    pass


class CircuitBreakerRegistry:
    """
    Global registry for managing multiple circuit breakers.
    
    Use this when you have multiple operations that need separate circuits.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
registry = CircuitBreakerRegistry()


# Convenience function
def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker from global registry."""
    return registry.get_or_create(name, config)


if __name__ == "__main__":
    # Example usage
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create circuit breaker
    breaker = CircuitBreaker(
        name="example_api",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=5
        )
    )
    
    # Simulate API calls
    call_count = 0
    
    def flaky_api_call():
        """Simulates an API that fails sometimes."""
        global call_count
        call_count += 1
        
        # Fail first 5 calls, then succeed
        if call_count <= 5:
            raise Exception("API Error")
        return {"status": "success", "data": "..."}
    
    # Test circuit breaker
    print("\n=== Testing Circuit Breaker ===\n")
    
    for i in range(10):
        print(f"\nAttempt {i+1}:")
        try:
            result = breaker.call(flaky_api_call)
            print(f"[OK] Success: {result}")
        except CircuitBreakerError as e:
            print(f"  Circuit Open: {e}")
        except Exception as e:
            print(f"  API Error: {e}")
        
        time.sleep(1)
    
    # Show final stats
    print("\n=== Final Statistics ===")
    print(breaker.get_stats())

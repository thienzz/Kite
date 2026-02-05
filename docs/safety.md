# Safety Guide

**Production safety mechanisms in Kite**

This guide covers Kite's safety features: circuit breakers, idempotency, and kill switches.

---

## Table of Contents

- [Circuit Breakers](#circuit-breakers)
- [Idempotency](#idempotency)
- [Kill Switches](#kill-switches)
- [Best Practices](#best-practices)

---

## Circuit Breakers

### Overview

Circuit breakers prevent cascading failures when operations fail repeatedly. They're essential for production systems that interact with external services.

**States:**
- **CLOSED**: Normal operation, all requests pass through
- **OPEN**: Blocking requests after threshold failures
- **HALF_OPEN**: Testing if service has recovered

### Configuration

```python
from kite import Kite
from kite.safety.circuit_breaker import CircuitBreakerConfig

ai = Kite()

# Configure circuit breaker
ai.circuit_breaker.config = CircuitBreakerConfig(
    failure_threshold=3,      # Open after 3 failures
    success_threshold=2,      # Close after 2 successes in half-open
    timeout_seconds=60,       # Wait 60s before testing recovery
    half_open_max_calls=1     # Allow 1 concurrent call in half-open
)
```

### How It Works

```
CLOSED (Normal)
    │
    ├─ Success ──▶ Stay CLOSED
    │
    └─ Failure ──▶ Increment counter
                    │
                    └─ Threshold reached? ──▶ OPEN
                    
OPEN (Blocking)
    │
    └─ Timeout expired? ──▶ HALF_OPEN
    
HALF_OPEN (Testing)
    │
    ├─ Success ──▶ Increment success counter
    │               │
    │               └─ Threshold reached? ──▶ CLOSED
    │
    └─ Failure ──▶ OPEN
```

### Usage

Circuit breakers are **automatically applied** to all LLM calls:

```python
# Automatically protected
response = ai.complete("Hello")

# If circuit opens, raises CircuitBreakerError
from kite.safety.circuit_breaker import CircuitBreakerError

try:
    response = ai.complete("Hello")
except CircuitBreakerError as e:
    print(f"Circuit open: {e}")
```

### Manual Protection

Protect custom operations:

```python
@ai.circuit_breaker.protected
def call_external_api():
    return requests.get("https://api.example.com")

# Or use call() method
result = ai.circuit_breaker.call(expensive_operation, arg1, arg2)
```

### Monitoring

```python
# Get statistics
stats = ai.circuit_breaker.get_stats()

print(f"State: {stats['state']}")
print(f"Total calls: {stats['total_calls']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Failed calls: {stats['failed_calls']}")
print(f"Rejected calls: {stats['rejected_calls']}")

# Manually reset if needed
ai.circuit_breaker.reset()
```

### Callbacks

Execute custom logic on state changes:

```python
def on_open(name, stats):
    print(f"Circuit {name} opened!")
    # Send alert, log to monitoring system, etc.

def on_close(name, stats):
    print(f"Circuit {name} closed!")

from kite.safety.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(
    name="my_service",
    on_open=on_open,
    on_close=on_close
)
```

---

## Idempotency

### Overview

Idempotency ensures that executing the same operation multiple times has the same effect as executing it once. Critical for:
- Payment processing
- Order creation
- Email sending
- Database writes

### Configuration

```python
from kite import Kite

ai = Kite()

# Configure idempotency
ai.idempotency.config.ttl_seconds = 3600  # 1 hour
ai.idempotency.config.storage_backend = 'redis'  # or 'memory'
```

### Usage

```python
# Execute with idempotency
result = ai.idempotency.execute(
    operation_id="process_order_12345",
    func=process_payment,
    args=(order_id, amount),
    kwargs={"currency": "USD"}
)

# Same operation_id returns cached result
result2 = ai.idempotency.execute(
    operation_id="process_order_12345",
    func=process_payment,
    args=(order_id, amount),
    kwargs={"currency": "USD"}
)  # Returns cached result, doesn't execute again
```

### Operation IDs

**Best Practices:**

```python
# Good: Unique, deterministic IDs
operation_id = f"order_{order_id}_{timestamp}"
operation_id = f"email_{user_id}_{email_type}"
operation_id = f"refund_{transaction_id}"

# Bad: Random or non-deterministic IDs
operation_id = str(uuid.uuid4())  # Different every time
operation_id = f"op_{time.time()}"  # Different every time
```

### Storage Backends

#### Memory (Development)

```python
ai.idempotency.config.storage_backend = 'memory'
```

**Pros:**
- Fast
- No external dependencies

**Cons:**
- Not persistent
- Not shared across instances

#### Redis (Production)

```python
ai.idempotency.config.storage_backend = 'redis'
ai.config['redis_url'] = 'redis://localhost:6379'
```

**Pros:**
- Persistent
- Shared across instances
- Automatic expiration (TTL)

**Cons:**
- Requires Redis

### TTL (Time-to-Live)

```python
# Default: 1 hour
ai.idempotency.config.ttl_seconds = 3600

# Custom TTL per operation
result = ai.idempotency.execute(
    operation_id="temp_op",
    func=some_func,
    ttl=300  # 5 minutes
)
```

### Monitoring

```python
# Check if operation was cached
result = ai.idempotency.execute(
    operation_id="op_123",
    func=my_func
)

if ai.idempotency.was_cached("op_123"):
    print("Result was cached")
```

---

## Kill Switches

### Overview

Kill switches provide emergency stop functionality for runaway agents or operations.

### Global Kill Switch

```python
from kite import Kite

ai = Kite()

# Activate kill switch
ai.kill_switch.activate("Emergency: Runaway agent detected")

# All operations will raise KillSwitchError
try:
    ai.complete("Hello")
except KillSwitchError as e:
    print(f"Kill switch active: {e}")

# Deactivate
ai.kill_switch.deactivate()
```

### Per-Agent Kill Switch

```python
import asyncio

agent = ai.create_agent("Assistant")

# Activate for specific agent
agent.kill_switch.activate("Agent misbehaving")

# Only this agent's operations are blocked
async def main():
    try:
        await agent.run("Hello")
    except KillSwitchError:
        print("Agent kill switch active")

asyncio.run(main())
```

```python
import asyncio

async def main():
    # Other agents still work
    other_agent = ai.create_agent("Other")
    await other_agent.run("Hello")  # Works fine

asyncio.run(main())
```

### Monitoring

```python
# Check kill switch status
if ai.kill_switch.is_active():
    print(f"Reason: {ai.kill_switch.reason}")
    print(f"Activated at: {ai.kill_switch.activated_at}")
```

---

## Best Practices

### 1. Use Circuit Breakers for External Calls

```python
# Protect all external API calls
@ai.circuit_breaker.protected
def call_stripe_api():
    return stripe.Charge.create(...)

@ai.circuit_breaker.protected
def call_sendgrid():
    return sendgrid.send(...)
```

### 2. Implement Idempotency for Write Operations

```python
# Payment processing
result = ai.idempotency.execute(
    operation_id=f"payment_{order_id}",
    func=process_payment,
    args=(order_id, amount)
)

# Order creation
result = ai.idempotency.execute(
    operation_id=f"order_{cart_id}_{user_id}",
    func=create_order,
    args=(cart_id, user_id)
)
```

### 3. Monitor Circuit Breaker Health

```python
# Regular health checks
stats = ai.circuit_breaker.get_stats()

if stats['state'] == 'open':
    # Alert operations team
    send_alert(f"Circuit breaker open: {stats}")

if stats['success_rate'] < 0.95:
    # Investigate degraded performance
    log_warning(f"Low success rate: {stats['success_rate']:.2%}")
```

### 4. Set Appropriate Thresholds

```python
# For critical operations (payments, orders)
critical_breaker = CircuitBreaker(
    name="payments",
    config=CircuitBreakerConfig(
        failure_threshold=2,      # Open quickly
        timeout_seconds=300,      # Wait longer before retry
        success_threshold=3       # Require more successes to close
    )
)

# For non-critical operations (analytics, logging)
non_critical_breaker = CircuitBreaker(
    name="analytics",
    config=CircuitBreakerConfig(
        failure_threshold=5,      # More tolerant
        timeout_seconds=30,       # Retry sooner
        success_threshold=1       # Close quickly
    )
)
```

### 5. Use Redis for Production Idempotency

```python
# Development
ai.idempotency.config.storage_backend = 'memory'

# Production
ai.idempotency.config.storage_backend = 'redis'
ai.config['redis_url'] = os.getenv('REDIS_URL')
```

### 6. Implement Kill Switch Monitoring

```python
# Check kill switch status in health endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy" if not ai.kill_switch.is_active() else "degraded",
        "kill_switch": {
            "active": ai.kill_switch.is_active(),
            "reason": ai.kill_switch.reason if ai.kill_switch.is_active() else None
        }
    }
```

### 7. Log Safety Events

```python
import logging

logger = logging.getLogger(__name__)

# Log circuit breaker events
def on_circuit_open(name, stats):
    logger.error(f"Circuit {name} opened: {stats}")

def on_circuit_close(name, stats):
    logger.info(f"Circuit {name} closed: {stats}")

# Log idempotency cache hits
if ai.idempotency.was_cached(operation_id):
    logger.info(f"Idempotency cache hit: {operation_id}")
```

---

## Production Checklist

- [ ] Circuit breakers configured for all external calls
- [ ] Idempotency enabled for write operations
- [ ] Redis configured for production idempotency
- [ ] Circuit breaker thresholds tuned for your workload
- [ ] Monitoring and alerting setup for circuit breaker state
- [ ] Kill switch accessible for emergency use
- [ ] Safety events logged to monitoring system
- [ ] Health checks include safety status

---

For more information, see:
- [Architecture Guide](architecture.md)
- [Deployment Guide](deployment.md)
- [API Reference](api_reference.md)

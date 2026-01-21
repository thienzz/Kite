# Deployment Guide

**Production deployment patterns for Kite**

This guide covers deploying Kite to production environments.

---

## Table of Contents

- [Docker Deployment](#docker-deployment)
- [Environment Configuration](#environment-configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Security](#security)
- [Best Practices](#best-practices)

---

## Docker Deployment

### Docker Compose Setup

The included `docker-compose.yml` provides a complete production stack:

```yaml
version: '3.8'

services:
  kite:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=openai
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:password@postgres:5432/kite
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=kite
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f kite

# Stop services
docker-compose down
```

---

## Environment Configuration

### Required Variables

```bash
# LLM Provider (required)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-your-key-here

# Or use Anthropic
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-5-sonnet-20241022
# ANTHROPIC_API_KEY=sk-ant-your-key

# Or use Groq (fast)
# LLM_PROVIDER=groq
# LLM_MODEL=llama-3.3-70b-versatile
# GROQ_API_KEY=gsk_your-key

# Embedding Provider (required)
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Optional Variables

```bash
# Safety Configuration
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT=60
IDEMPOTENCY_TTL=3600

# Memory Configuration
VECTOR_BACKEND=faiss
VECTOR_DIMENSION=384

# Redis (for caching and idempotency)
REDIS_URL=redis://localhost:6379

# PostgreSQL (for session storage)
POSTGRES_URL=postgresql://user:pass@localhost:5432/kite

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

---

## Monitoring

### Prometheus Metrics

Kite exposes Prometheus metrics at `/metrics`:

```python
from kite.monitoring import metrics

# LLM metrics
metrics.llm_calls.inc()
metrics.llm_latency.observe(duration)
metrics.llm_errors.inc()

# Circuit breaker metrics
metrics.circuit_breaker_state.set(state)
metrics.circuit_breaker_failures.inc()

# Agent metrics
metrics.agent_calls.inc()
metrics.agent_success_rate.set(rate)
```

### Grafana Dashboards

Import the included Grafana dashboard:

1. Open Grafana at `http://localhost:3000`
2. Login with `admin/admin`
3. Import `grafana/kite_dashboard.json`

**Key Metrics:**
- LLM call rate and latency
- Circuit breaker state
- Agent success rate
- Memory usage
- Error rate

---

## Scaling

### Horizontal Scaling

Run multiple Kite instances behind a load balancer:

```yaml
# docker-compose.yml
services:
  kite:
    deploy:
      replicas: 3
    # ... rest of config
```

### Load Balancing

Use Nginx or HAProxy:

```nginx
upstream kite_backend {
    least_conn;
    server kite1:8000;
    server kite2:8000;
    server kite3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://kite_backend;
    }
}
```

### Caching Strategy

Use Redis for distributed caching:

```python
# Enable Redis caching
ai.config['cache_backend'] = 'redis'
ai.config['redis_url'] = 'redis://redis:6379'

# Cache LLM responses
result = ai.cache.get_or_set(
    key=f"llm_{hash(prompt)}",
    func=lambda: ai.complete(prompt),
    ttl=3600
)
```

---

## Security

### API Key Management

**Never commit API keys to version control.**

Use environment variables or secret management:

```bash
# .env (gitignored)
OPENAI_API_KEY=sk-...

# Or use Docker secrets
docker secret create openai_key openai_key.txt
```

### Rate Limiting

Implement rate limiting to prevent abuse:

```python
from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

@app.post("/complete")
@limiter.limit("10/minute")
async def complete(request: Request):
    # ... handle request
```

### Input Validation

Validate all inputs:

```python
from pydantic import BaseModel, validator

class CompletionRequest(BaseModel):
    prompt: str
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v) > 10000:
            raise ValueError("Prompt too long")
        return v
```

---

## Best Practices

### 1. Use Circuit Breakers

Protect against cascading failures:

```python
ai.circuit_breaker.config.failure_threshold = 3
ai.circuit_breaker.config.timeout_seconds = 60
```

### 2. Enable Idempotency

Prevent duplicate operations:

```python
result = ai.idempotency.execute(
    operation_id=f"order_{order_id}",
    func=process_payment,
    args=(order_id, amount)
)
```

### 3. Monitor Performance

Track key metrics:

```python
# Agent performance
print(f"Success rate: {agent.success_count / agent.call_count:.2%}")

# Circuit breaker health
stats = ai.circuit_breaker.get_stats()
print(f"Circuit state: {stats['state']}")
```

### 4. Use Async Operations

Maximize throughput:

```python
# Concurrent agent calls
results = await asyncio.gather(
    agent1.run(query1),
    agent2.run(query2),
    agent3.run(query3)
)
```

### 5. Implement Graceful Shutdown

Handle shutdown signals:

```python
import signal
import sys

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    # Close connections, save state
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### 6. Use Health Checks

Implement health check endpoints:

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "circuit_breaker": ai.circuit_breaker.state.value,
        "llm_provider": ai.config['llm_provider']
    }
```

### 7. Log Structured Data

Use structured logging:

```python
import logging
import json

logger = logging.getLogger(__name__)

logger.info(json.dumps({
    "event": "agent_call",
    "agent": agent.name,
    "duration": duration,
    "success": success
}))
```

---

## Troubleshooting

### High Latency

**Symptoms**: Slow response times

**Solutions**:
- Enable Redis caching
- Use faster LLM provider (Groq)
- Reduce `max_iterations` for planning agents
- Implement request batching

### Circuit Breaker Opens Frequently

**Symptoms**: Many rejected calls

**Solutions**:
- Increase `failure_threshold`
- Increase `timeout_seconds`
- Check LLM provider status
- Review error logs

### Memory Issues

**Symptoms**: High memory usage

**Solutions**:
- Use FAISS instead of ChromaDB for vector memory
- Limit vector memory size
- Clear session memory periodically
- Use Redis for caching instead of in-memory

### Connection Pool Exhaustion

**Symptoms**: Connection errors

**Solutions**:
- Increase connection pool size
- Implement connection pooling
- Use async operations
- Add connection timeouts

---

## Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured (not in code)
- [ ] Circuit breakers enabled
- [ ] Idempotency configured
- [ ] Monitoring setup (Prometheus + Grafana)
- [ ] Logging configured
- [ ] Health checks implemented
- [ ] Rate limiting enabled
- [ ] Input validation added
- [ ] Error handling implemented
- [ ] Graceful shutdown configured
- [ ] Backup strategy defined
- [ ] Scaling plan documented

---

For more information, see:
- [Architecture Guide](architecture.md)
- [API Reference](api_reference.md)
- [Safety Guide](safety.md)

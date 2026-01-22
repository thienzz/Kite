# 🪁 Kite

**Production-Ready Agentic AI Framework**  
*High-Performance • Lightweight • Simple*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Kite is a **production-grade framework** for building intelligent AI agents and workflows. Designed for real-world applications, it bridges the gap between probabilistic LLM outputs and deterministic business logic with enterprise-grade safety, memory, and observability.

---

## 🎯 Why Kite?

**Built for Production from Day One**

- ⚡ **High Performance**: Lazy-loaded architecture with minimal startup overhead (~50ms)
- 🛡️ **Enterprise Safety**: Circuit breakers, idempotency, and self-healing validation
- 🚀 **Multi-Provider**: Seamless switching between OpenAI, Anthropic, Groq, Together AI, and local models
- 🧠 **Advanced Memory**: Vector RAG, Graph RAG, and session memory out-of-the-box
- 📊 **Observable**: Built-in monitoring, metrics, and cost tracking
- 🔧 **Simple API**: Intuitive design that scales from prototypes to production

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/thienzz/Kite.git
cd Kite
pip install -r requirements.txt
```

### 30-Second Example

```python
from kite import Kite

# Initialize framework (lazy-loads components)
ai = Kite()

# Create a specialized agent
analyst = ai.create_agent(
    name="DataAnalyst",
    system_prompt="You are a data analyst. Provide actionable insights.",
    tools=[ai.tools.get("web_search")]
)

# Run with automatic safety and retry logic
result = await analyst.run("Analyze Q4 revenue trends")
print(result['response'])
```

**That's it.** No complex configuration. Production-ready by default.

---

## 🏗️ Architecture

### Core Components

```
kite/
├── agents/          # 4 reasoning patterns (ReAct, Plan-Execute, ReWOO, ToT)
├── memory/          # Vector, Graph RAG, Session memory
├── safety/          # Circuit breakers, idempotency, kill switches
├── pipeline/        # Deterministic workflows with HITL support
├── routing/         # Semantic and aggregator routers
├── tools/           # Built-in tools + MCP integrations
└── monitoring/      # Real-time metrics and dashboards
```

### Design Philosophy

1. **Lazy Loading**: Components initialize only when needed
2. **Fail-Safe Defaults**: Circuit breakers and retries enabled automatically
3. **Provider Agnostic**: Switch LLMs without changing code
4. **Observable**: Every operation is logged and metered

---

## 🚀 Key Features

### 1. Multi-Provider LLM Support

```python
# Use any provider with the same API
ai = Kite()  # Auto-detects from .env

# OpenAI
ai.config['llm_provider'] = 'openai'
ai.config['llm_model'] = 'gpt-4'

# Anthropic
ai.config['llm_provider'] = 'anthropic'
ai.config['llm_model'] = 'claude-3-5-sonnet-20241022'

# Local (Ollama)
ai.config['llm_provider'] = 'ollama'
ai.config['llm_model'] = 'qwen2.5:1.5b'

# Groq (ultra-fast)
ai.config['llm_provider'] = 'groq'
ai.config['llm_model'] = 'llama-3.3-70b-versatile'
```

### 2. Production Safety

**Circuit Breakers** prevent cascading failures:

```python
# Automatic protection for all LLM calls
ai.circuit_breaker.config.failure_threshold = 3
ai.circuit_breaker.config.timeout_seconds = 60

# Circuit opens after 3 failures, blocks requests for 60s
# Automatically transitions to half-open for testing recovery
```

**Idempotency** prevents duplicate operations:

```python
# Deduplicate requests within 1 hour window
result = ai.idempotency.execute(
    operation_id="process_order_12345",
    func=process_payment,
    args=(order_id, amount)
)
```

### 3. Advanced Memory Systems

**Vector Memory** for semantic search:

```python
# Add documents
ai.vector_memory.add_document("doc1", "Kite is a production-ready framework...")

# Semantic search
results = ai.vector_memory.search("What is Kite?", top_k=3)
```

**Graph RAG** for relationship-aware retrieval:

```python
# Build knowledge graph
ai.graph_rag.add_entity("Kite", "framework", {"type": "software"})
ai.graph_rag.add_relationship("Kite", "uses", "Circuit Breakers")

# Query with context
answer = ai.graph_rag.query("How does Kite ensure reliability?")
```

### 4. Agent Reasoning Patterns

**ReAct** (Reasoning + Acting):
```python
agent = ai.create_agent("Assistant", tools=[search_tool, calculator])
result = await agent.run("What's the GDP of France in 2024?")
```

**Plan-and-Execute** with replanning:
```python
planner = ai.create_planning_agent(strategy="plan-and-execute")
result = await planner.run("Research AI market and suggest pricing")
```

**Tree-of-Thoughts** for complex reasoning:
```python
tot = ai.create_planning_agent(strategy="tot", max_iterations=3)
result = await tot.run("Evaluate 3 mitigation strategies for...")
```

**ReWOO** (parallel execution):
```python
rewoo = ai.create_planning_agent(strategy="rewoo")
result = await rewoo.run("Search news for LangChain, CrewAI, AutoGPT")
```

### 5. Human-in-the-Loop Workflows

```python
# Create workflow with checkpoints
workflow = ai.create_workflow("approval_flow")
workflow.add_step("research", research_func)
workflow.add_checkpoint("research", approval_required=True)
workflow.add_step("execute", execute_func)

# Execute (pauses at checkpoint)
state = await workflow.execute_async({"query": "..."})

# Resume after approval
state = await workflow.resume_async(state.task_id, feedback="Approved")
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Startup Time** | ~50ms (lazy loading) |
| **Memory Footprint** | <100MB (base) |
| **LLM Latency** | Provider-dependent (Groq: ~500ms, OpenAI: ~2s) |
| **Throughput** | 100+ requests/sec (with caching) |
| **Concurrent Agents** | Limited by LLM provider rate limits |

**Benchmarks** (on M1 Mac, local Ollama):
- Simple completion: 50-200ms
- Agent with tools: 500ms-2s
- Plan-and-execute (3 steps): 3-5s

---

## 🛠️ Production Deployment

### Docker

```bash
docker-compose up -d
```

Includes:
- Kite API server (FastAPI)
- Redis (caching + idempotency)
- PostgreSQL (session storage)
- Prometheus + Grafana (monitoring)

### Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Embedding Provider
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Safety
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT=60
IDEMPOTENCY_TTL=3600

# Memory
VECTOR_BACKEND=faiss
VECTOR_DIMENSION=384
```

See [`.env.example`](.env.example) for full configuration.

---

## 📚 Examples

The [`examples/`](examples/) directory contains 9 production-ready case studies:

1. **Invoice Pipeline** - Deterministic 4-step processing with validation
2. **Semantic Router** - Intent classification using LLMs
3. **SQL Analytics** - Natural language to SQL with safety checks
4. **Research Assistant** - Multi-step search and synthesis
5. **Enterprise System** - Full integration of all features
6. **E-commerce Support** - Customer service automation
7. **Advanced Planning** - Plan-Execute, ReWOO, Tree-of-Thoughts
8. **Conversational Agents** - Multi-agent dialogue with consensus
9. **Human-in-the-Loop** - Approval workflows with checkpoints

Run any example:
```bash
python examples/case1_invoice_pipeline_framework.py
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Specific test suites
pytest tests/verify_planning.py
pytest tests/verify_conversation.py
pytest tests/verify_hitl.py

# With coverage
pytest --cov=kite tests/
```

---

## 📖 Documentation

Comprehensive guides for building production-grade AI agents:

- **[Quick Start](docs/quickstart.md)** - Get started in 5 minutes
- **[Architecture Guide](docs/architecture.md)** - System design and components
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Deployment Guide](docs/deployment.md)** - Docker, monitoring, scaling
- **[Safety Guide](docs/safety.md)** - Circuit breakers, idempotency, kill switches
- **[Memory Systems](docs/memory.md)** - Vector, Graph RAG, session memory

---

## 🤝 Contributing

We welcome contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

**Key Areas:**
- New agent reasoning patterns
- Additional LLM provider integrations
- Performance optimizations
- Documentation improvements

---

## 🗺️ Roadmap

- [ ] **v1.1**: Streaming responses, async batch processing
- [ ] **v1.2**: Multi-agent orchestration patterns
- [ ] **v1.3**: Fine-tuning integration for custom models
- [ ] **v2.0**: Distributed agent execution (Ray/Celery)

---

## 📜 License

MIT License - see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [FastEmbed](https://github.com/qdrant/fastembed) - Fast embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [ChromaDB](https://www.trychroma.com/) - Vector database

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/thienzz/Kite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/thienzz/Kite/discussions)
- **Email**: [your-email@example.com]

---

<p align="center">
  <strong>Built for developers who ship to production.</strong><br>
  Star ⭐ this repo if you find it useful!
</p>


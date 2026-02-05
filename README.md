# 🪁 Kite

**Build Production-Ready AI Agents That Actually Work**

*Fast • Safe • Simple • Powerful*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/thienzz/Kite/releases)

> 🚀 **What is Kite?** A lightweight Python framework that turns LLMs into reliable AI agents you can deploy with confidence. No PhD required.

## 📦 Installation

**Via pip (recommended):**
```bash
pip install kite-agent
```

**From source:**
```bash
git clone https://github.com/thienzz/Kite.git
cd Kite
pip install -e .
```

[Quick Start](#-quick-start) • [Examples](#-production-examples) • [Features](#-core-features) • [Documentation](#-documentation)

---

## ✨ Why Developers Choose Kite

Most AI frameworks overwhelm you with complexity. Kite gives you **production-grade reliability** with **dead-simple APIs**:

```python
from kite import Kite

# Initialize once
ai = Kite()

# Create a specialist agent
support_agent = ai.create_agent(
    name="CustomerSupport",
    system_prompt="You are a helpful e-commerce support agent.",
    tools=[search_orders, process_refunds],
    agent_type="react"  # Autonomous reasoning loop
)

# Run it
result = await support_agent.run("Where is order ORD-12345?")
print(result['response'])
```

**That's it.** Behind the scenes, Kite handles:
- ✅ Circuit breakers (prevent cascading failures)
- ✅ Retry logic (auto-recovery from API errors)
- ✅ Memory management (RAG, sessions, graph knowledge)
- ✅ Multi-provider support (OpenAI, Anthropic, Groq, local models)
- ✅ Cost tracking & monitoring

---

## 🎯 Built for Real-World Problems

Stop building MVP demos. Start shipping production systems:

| Your Challenge | Kite's Solution |
|---------------|----------------|
| "LLMs hallucinate in production" | **Vector RAG + Graph RAG** for grounded responses |
| "API failures crash my agents" | **Circuit breakers** auto-pause failing services |
| "Too slow & expensive" | **Smart/Fast model routing** - use cheap models when possible |
| "Can't track what agents are doing" | **Event bus + metrics** for full observability |
| "Hard to prevent dangerous actions" | **Guardrails & shell whitelisting** built-in |
| "Need human approval for critical tasks" | **HITL workflows** with checkpoints |

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/thienzz/Kite.git
cd Kite
pip install -r requirements.txt
```

### Setup Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Minimum config:**
```bash
LLM_PROVIDER=openai  # or anthropic, groq, ollama
OPENAI_API_KEY=sk-...
```

### Your First Agent (30 seconds)

```python
import asyncio
from kite import Kite

async def main():
    # Auto-loads from .env
    ai = Kite()
    
    # Create a tool
    def get_weather(city: str) -> str:
        return f"Sunny, 72°F in {city}"
    
    weather_tool = ai.create_tool("get_weather", get_weather, 
                                   "Get current weather for a city")
    
    # Create agent
    agent = ai.create_agent(
        name="WeatherBot",
        system_prompt="You help users check weather. Always use the tool.",
        tools=[weather_tool]
    )
    
    # Run
    result = await agent.run("What's the weather in San Francisco?")
    print(result['response'])

asyncio.run(main())
```

---

## 🏗️ Architecture Overview

Kite's modular design lets you use what you need:

```
kite/
├── agents/          # 🤖 Reasoning patterns (ReAct, ReWOO, ToT, Plan-Execute)
├── memory/          # 🧠 Vector RAG, Graph RAG, Session Memory
├── safety/          # 🛡️ Circuit Breakers, Idempotency, Kill Switches
├── routing/         # 🧭 Semantic Routing, Aggregator Routing, Smart/Fast Model Selection
├── tools/           # 🔧 Built-in utilities (Web Search, Code Execution, Shell, MCP integrations)
├── pipeline/        # ⚙️ Deterministic workflows with HITL support
└── monitoring/      # 📊 Metrics, Tracing, Event Bus
```

### Core Components (Lazy-Loaded)

```python
ai = Kite()

# These initialize only when accessed:
ai.llm                  # LLM provider (OpenAI, Anthropic, Groq, Ollama)
ai.embeddings           # Embedding provider (FastEmbed, OpenAI)
ai.vector_memory        # Vector similarity search (FAISS, ChromaDB, or in-memory)
ai.graph_rag            # Knowledge graph for relationships
ai.session_memory       # Conversation history
ai.semantic_router      # Intent-based routing
ai.circuit_breaker      # Fault tolerance
ai.idempotency          # Duplicate request prevention
ai.tools                # Tool registry
ai.pipeline             # Workflow manager
```

---

## 🚀 Core Features

### 1️⃣ Multiple Reasoning Patterns

Choose the right "brain" for your task:

```python
# ReAct: Standard loop (Think → Act → Observe → Repeat)
agent = ai.create_agent(..., agent_type="react")

# ReWOO: Plan everything upfront, execute in parallel (FAST!)
agent = ai.create_agent(..., agent_type="rewoo")

# Tree-of-Thoughts: Explore multiple solutions (creative tasks)
agent = ai.create_agent(..., agent_type="tot")

# Plan-Execute: Classic two-phase planning
agent = ai.create_agent(..., agent_type="plan_execute")
```

**See them in action:** [examples/case6_reasoning_architectures.py](examples/case6_reasoning_architectures.py)

---

### 2️⃣ Production Safety Mechanisms

**Circuit Breakers** prevent cascading failures:

```python
ai.circuit_breaker.config.failure_threshold = 3  # Open after 3 failures
ai.circuit_breaker.config.timeout_seconds = 60   # Cool-down period

# Circuit auto-opens if LLM/tool fails 3x, preventing waste
```

**Idempotency** prevents duplicate operations:

```python
# Same operation_id within TTL returns cached result
result = ai.idempotency.execute(
    operation_id="order_123_refund",
    func=process_refund,
    args=(order_id,)
)
```

**Guardrails** for dangerous operations:

```python
from kite.tools.system_tools import ShellTool

# Whitelist safe commands only
shell = ShellTool(allowed_commands=["ls", "git", "df", "uptime"])

# Blocks 'rm -rf', 'sudo', etc. automatically
```

---

### 3️⃣ Advanced Memory Systems

**Vector Memory** for semantic search:

```python
# Add knowledge
ai.vector_memory.add_document("policy_001", "Returns accepted within 30 days...")

# Semantic search
results = ai.vector_memory.search("What's the return policy?", top_k=3)
```

**Graph RAG** for relationship-aware knowledge:

```python
ai.graph_rag.add_entity("Kite", "framework", {"language": "Python"})
ai.graph_rag.add_relationship("Kite", "uses", "OpenAI")

# Query walks the graph
answer = ai.graph_rag.query("What providers does Kite support?")
```

**Session Memory** for conversations:

```python
ai.session_memory.add_message(session_id="user_123", role="user", content="Hi!")
history = ai.session_memory.get_history(session_id="user_123")
```

---

### 4️⃣ Smart Multi-Provider Support

Switch between providers without changing code:

```python
# OpenAI
ai.config['llm_provider'] = 'openai'
ai.config['llm_model'] = 'gpt-4o'

# Anthropic
ai.config['llm_provider'] = 'anthropic'
ai.config['llm_model'] = 'claude-3-5-sonnet-20241022'

# Groq (ultra-fast inference)
ai.config['llm_provider'] = 'groq'
ai.config['llm_model'] = 'llama-3.3-70b-versatile'

# Local with Ollama
ai.config['llm_provider'] = 'ollama'
ai.config['llm_model'] = 'qwen2.5:1.5b'
```

**Cost Optimization**: Use resource-aware routing:

```python
from kite.optimization.resource_router import ResourceAwareRouter

router = ResourceAwareRouter(ai.config)

# Automatically uses:
# - FAST model (cheap) for routing, simple tasks
# - SMART model (powerful) for complex reasoning
analyst = ai.create_agent(
    name="Analyst",
    model=router.smart_model,  # gpt-4o for hard problems
    ...
)
```

---

### 5️⃣ Human-in-the-Loop Workflows

Build approval workflows for critical operations:

```python
from kite.pipeline import DeterministicPipeline

# Define workflow
def draft_email(state):
    return {"draft": "Dear Customer, ..."}

def send_email(state):
    return {"status": "sent"}

# Create pipeline with checkpoint
pipeline = ai.pipeline.create("approval_flow")
pipeline.add_step("draft", draft_email)
pipeline.add_checkpoint("draft")  # Pauses here for approval
pipeline.add_step("send", send_email)

# Execute (stops at checkpoint)
state = await pipeline.execute_async({"to": "user@example.com"})

# Human reviews, then resume
final = await pipeline.resume_async(state.task_id, approved=True)
```

**Real example:** [case4_multi_agent_collab.py](examples/case4_multi_agent_collab.py)

---

## 📊 Production Examples

We built **6 real-world case studies** to show you exactly how to use Kite:

| Case | Scenario | Key Concepts | Difficulty |
|------|----------|--------------|-----------|
| **[Case 1](examples/case1_ecommerce_support.py)** | E-commerce Support Bot | LLM Routing, Tools, Multi-Agent | 🟢 Beginner |
| **[Case 2](examples/case2_enterprise_analytics.py)** | Data Analyst Agent | SQL + Python Execution, Charts | 🟡 Intermediate |
| **[Case 3](examples/case3_research_assistant.py)** | Deep Research System | Web Scraping, Multi-Step Planning | 🟡 Intermediate |
| **[Case 4](examples/case4_multi_agent_collab.py)** | Multi-Agent Collaboration | Supervisor Pattern, HITL, Iterative Refinement | 🔴 Advanced |
| **[Case 5](examples/case5_devops_automation.py)** | DevOps Automation | Shell Tools, Safety Guardrails | 🟡 Intermediate |
| **[Case 6](examples/case6_reasoning_architectures.py)** | Reasoning Pattern Comparison | ReAct vs ReWOO vs ToT | 🔴 Advanced |

### Run an Example

```bash
# E-commerce support demo
PYTHONPATH=. python3 examples/case1_ecommerce_support.py

# Data analyst with charts
PYTHONPATH=. python3 examples/case2_enterprise_analytics.py
```

**👉 [See detailed tutorials for each case →](examples/README.md)**

---

## 📈 Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Framework Startup** | ~50ms (lazy loading) |
| **Memory Footprint** | <100MB (base) |
| **Agent Latency** | 500ms - 2s (depends on LLM provider) |
| **Throughput** | 100+ req/s with caching |

**Real data** (M1 Mac, Ollama qwen2.5:1.5b):
- Simple completion: **50-200ms**
- ReAct agent (3 tool calls): **800ms-1.5s**
- Plan-Execute (5 steps): **3-5s**

---

## 🛠️ Production Deployment

### Docker Compose (Recommended)

```bash
docker-compose up -d
```

**Includes:**
- Kite API server (FastAPI)
- Redis (caching)
- PostgreSQL (session storage)
- Prometheus + Grafana (monitoring)

### Environment Variables

See [.env.example](.env.example) for all options. Key configs:

```bash
# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Embeddings
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Safety
CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT_SECONDS=60
IDEMPOTENCY_TTL=3600

# Memory
VECTOR_BACKEND=faiss
VECTOR_DIMENSION=384

# Optimization
FAST_LLM_MODEL=groq/llama-3.1-8b-instant  # Cheap routing
SMART_LLM_MODEL=openai/gpt-4o             # Complex tasks
```

---

## 📖 Documentation

### Guides

- **[Quick Start Guide](docs/quickstart.md)** - Get running in 5 minutes
- **[Architecture Deep Dive](docs/architecture.md)** - How Kite works internally
- **[API Reference](docs/api_reference.md)** - Complete API docs
- **[Deployment Guide](docs/deployment.md)** - Docker, scaling, monitoring
- **[Safety Patterns](docs/safety.md)** - Circuit breakers, guardrails, idempotency
- **[Memory Systems](docs/memory.md)** - Vector, Graph RAG, sessions

### Examples

All examples include detailed inline comments and step-by-step walkthroughs:

- [E-commerce Support](examples/case1_ecommerce_support.py) - Multi-agent routing
- [Enterprise Analytics](examples/case2_enterprise_analytics.py) - SQL + Python
- [Research Assistant](examples/case3_research_assistant.py) - Web research
- [Multi-Agent Workflow](examples/case4_multi_agent_collab.py) - Supervisor pattern
- [DevOps Automation](examples/case5_devops_automation.py) - Safe shell execution
- [Reasoning Patterns](examples/case6_reasoning_architectures.py) - ReAct/ReWOO/ToT

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Specific suites
pytest tests/test_framework.py      # Core functionality
pytest tests/test_async_concurrency.py  # Async patterns
pytest tests/test_exports.py        # Module exports

# With coverage
pytest --cov=kite tests/
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- 🐛 Bug reports & feature requests
- 📝 Documentation improvements
- 🔧 New reasoning patterns
- 🌐 Additional LLM integrations
- ⚡ Performance optimizations

**Priority areas:**
- More agent architectures (LATS, Reflexion)
- Streaming response support
- Multi-agent orchestration patterns
- Integration tests for all examples

---

## 🗺️ Roadmap

- [x] **v0.1.0**: Core framework, ReAct/ReWOO/ToT agents
- [ ] **v0.2.0**: Streaming responses, async batch processing
- [ ] **v0.3.0**: Multi-agent coordination primitives
- [ ] **v0.4.0**: Fine-tuning integration
- [ ] **v1.0.0**: Production-ready release with full test coverage

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

**TLDR:** Use it however you want. Commercial use welcome. No warranty.

---

## 🙏 Acknowledgments

Built with amazing open-source tools:

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [FastEmbed](https://github.com/qdrant/fastembed) - Lightning-fast embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook's vector search
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://python.langchain.com/) - Inspiration for tool abstractions

---

## 💬 Community & Support

- **🐛 Bug Reports:** [GitHub Issues](https://github.com/thienzz/Kite/issues)
- **💡 Feature Requests:** [GitHub Discussions](https://github.com/thienzz/Kite/discussions)
- **📧 Contact:** thien@beevr.ai

---

<p align="center">
  <strong>Stop building demos. Start shipping AI agents to production.</strong><br>
  ⭐ Star this repo if Kite helps you build better AI systems!
</p>

<p align="center">
  Made with ❤️ by developers who ship production AI
</p>

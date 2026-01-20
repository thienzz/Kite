# 🪁 Kite: High-Performance Agentic Framework

Kite is a production-ready, modular framework designed for building robust AI agents and high-performance pipelines. It bridges the gap between probabilistic LLM outputs and deterministic business logic with built-in safety, memory, and monitoring systems.

---

## ✨ Key Features

- **🚀 Multi-Engine Support**: Native integration with OpenAI, Anthropic, Groq, Together AI, and local models via Ollama.
- **⚡ SLM Optimization**: Specialized support for "Small Language Models" (SLMs) to handle high-frequency, low-latency tasks.
- **🧠 Advanced Memory**: Structured **Semantic Memory** (Vector DB) and **Short-term Memory** for context-aware agents.
- **🛡️ Production Safety**: Built-in **Circuit Breakers**, **Idempotency Managers**, and **Self-Healing** validation.
- **📈 Observability**: Real-time monitoring and metrics for agent performance, cost, and success rates.
- **🔄 Deterministic Pipelines**: Orchestrate complex workflows with state management and error handling.

---

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thienzz/Kite.git
cd Kite

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from kite import AgenticAI

# 1. Initialize Framework
ai = AgenticAI()

# 2. Extract structured data with built-in safety
text = "Invoice #INV-001 from Acme Corp for $1,250.00"
prompt = f"Extract JSON from this text: {text}"

# Automatically handles circuit breaking and retries
response = ai.complete(prompt)
print(f"Extracted: {response}")

# 3. Create a specialized Agent
analyst = ai.create_agent(
    name="DataAnalyst",
    system_prompt="You are a data analyst. provide insights.",
    tools=[] # Add your tools here
)

# analyst.run("Analyze the latest trends...")
```

---

## 📂 Core Components

| Component | Description |
| :--- | :--- |
| `kite` | Core package of the framework. |
| **`Agent`** | High-level persona-driven entities that use tools and models. |
| **`Pipeline`** | Linear workflows for deterministic data processing. |
| **`VectorMemory`** | RAG-ready semantic storage and retrieval system. |
| **`Safety`** | Circuit breakers and idempotency to protect production systems. |

---

## 📖 Case Studies & Examples

The `examples/` directory contains complete implementations of common patterns:

1. [**Invoice Pipeline**]: Deterministic 4-step processing with self-healing validation.
2. [**Semantic Router**]: Efficient intent routing using SLMs.
3. [**SQL Analytics**]: Natural language to SQL with safety checks.
4. [**Research Assistant**]: Complex multi-step search and synthesis.
5. [**Enterprise System**]: Full integration of all framework features.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE] file for details.

# 🪁 Kite

**Reliable AI Agents for Python**

*Fast • Safe • Simple • Powerful*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/kite-agent)](https://pypi.org/project/kite-agent/)

---

### Introduction

Kite is a lightweight Python framework for building production-grade AI agents. 

Most agent frameworks rely heavily on prompt engineering for safety. Kite takes a different approach: it treats the LLM as an untrusted reasoning engine and enforces safety through code.

It provides a "kernel" that validates every action the agent proposes before it executes. This means you get strict control over permissions, blast radius, and failure handling.

[Read the Architecture →](ARCHITECTURE.md)

---

## ⚡ Quick Start

### Installation

```bash
pip install kite-agent
```

### Your First Agent

Here is a simple agent that proposes refunds but is supervised by the framework.

```python
import asyncio
from kite import Kite

async def main():
    ai = Kite()

    # 1. Define Tools (The Execution Layer)
    # The only way the agent affects the world.
    def refund_order(order_id: str):
        print(f"💰 Refunding {order_id}...")
        return "Refunded"
        
    # Explicitly register the tool
    refund_tool = ai.create_tool(name="refund", func=refund_order)

    # 2. Define Policy (The Safety Layer)
    # Stop if it fails 3 times in a row.
    ai.circuit_breaker.config.failure_threshold = 3

    # 3. Create Agent (The Cognition Layer)
    agent = ai.create_agent(
        name="SupportBot",
        tools=[refund_tool],
        system_prompt="You are a support agent. Propose refunds if asked."
    )

    # 4. Run (The Kernel Supervision)
    # The framework validates the tool call before execution.
    result = await agent.run("Please refund order #999")
    print(result['response'])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Core Philosophy

*   **Code over Prompts**: Safety logic belongs in Python, not in English prompts.
*   **Whitelisting**: Agents should only have access to tools you explicitly whitelist.
*   **Circuit Breakers**: Prevent infinite loops and cascading API failures.
*   **Auditable**: Every decision and action is traced.

---

## ⚡ The "Kernel" Pattern

In Kite, the agent doesn't execute code directly. It proposes actions to the Kernel.

```python
# ❌ BAD: Logic in Prompt
# "Please check if the user is admin before deleting"
# Result: LLM might ignore this.

# ✅ GOOD: Logic in Code (Kite)
def delete_user(user_id: str):
    if not current_user.is_admin:
        raise SecurityError("Permission Denied")
    db.delete(user_id)
```

## 🚀 Key Features

*   **Multiple Reasoning Patterns**: Support for ReAct, ReWOO, Tree-of-Thoughts.
*   **Production Safety**: Built-in circuit breakers, idempotency keys, and shell access whitelists.
*   **Memory Systems**: Vector RAG, Graph RAG, and persistent session memory.
*   **Multi-Provider**: Switch between OpenAI, Anthropic, Groq, and Ollama purely via config.
*   **Human-in-the-loop**: Pause execution for human approval on sensitive actions.

---

## 📦 Architecture Overview

Kite is modular. You can use the components independently.

```
kite/
├── agents/          # Reasoning engines (ReAct, Plan-Execute)
├── memory/          # Vector & Graph memory
├── safety/          # Circuit breakers & Guardrails
├── routing/         # Model routing (Fast vs Smart)
└── tools/           # Standard tool library
```

---

## 📈 Performance

*   **Startup**: ~50ms (lazy loading)
*   **Memory**: <100MB base footprint
*   **Throughput**: Designed for high-concurrency async workloads.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

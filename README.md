# 🪁 Kite

> **The "Boring" Architecture for Reliable Agents**

**Stable • Deterministic • Auditable**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/kite-agent)](https://pypi.org/project/kite-agent/)

---

### Hypothesis 0

Most agent frameworks sell "magic". Kite is built on a boring premise:
> **The Agent is an inherently UNTRUSTED component.**

If a system relies on the LLM "being smart" to be safe, it is broken.
Kite is an **enforcement kernel** that treats LLM outputs as *proposals*, not instructions.

[Read the Architecture →](ARCHITECTURE.md)

---

## 🎯 Why Kite?

Kite doesn't optimize for "demo wow factor". It optimizes for **production sleep**.

| Feature | The Hype Way | The Kite Way |
| :--- | :--- | :--- |
| **Philosophy** | "Let the agent decide" | **"Code is Law"** |
| **Safety** | "Please don't delete files" (Prompt) | **Regex Kernel + Whitelist** (Code) |
| **Errors** | Infinite retry loops | **Circuit Breakers** |
| **Hallucination** | "Prompt Engineering" | **Blast Radius Containment** |

## ⚡ The "Kernel" Pattern

In Kite, the agent has **zero authority**. It can only suggest actions to the Kernel.

```python
# ❌ BAD: The "Agent" way (Logic in Prompt)
# "Please check if the user is admin before deleting"
# Result: LLM ignores you, deletes DB.

# ✅ GOOD: The "Kite" way (Logic in Code)
def delete_user(user_id: str):
    # The Kernel enforces policy. The LLM cannot bypass this.
    if not current_user.is_admin:
        raise SecurityError("Permission Denied: Agent attempted admin action")
    
    db.delete(user_id)

# The agent PROPOSES: {"tool": "delete_user", "id": "123"}
# The Kernel VALIDATES. 
# If validation fails, the tool never runs.
```

## 📦 Installation

```bash
pip install kite-agent
```

## 🚀 Quick Start: The "Safe" Agent

A Kite agent isn't just a loop; it's a managed process with supervision.

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

    # 2. Define Policy (The Safety Layer)
    # Circuit breaker: Stop if it fails 3 times.
    ai.circuit_breaker.configure(threshold=3)

    # 3. Create Agent (The Cognition Layer)
    agent = ai.create_agent(
        name="SupportBot",
        tools=[refund_order],
        system_prompt="You are a support agent. Propose refunds if asked."
    )

    # 4. Run (The Kernel Supervision)
    # If the agent hallucinates a tool that doesn't exist, Kite blocks it.
    # If the agent tries to call 'refund' 50 times/sec, Circuit Breaker trips.
    result = await agent.run("Please refund order #999")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏗️ Architecture Overview

Kite's modular design lets you use what you need. It is not just "one agent loop", but a suite of reliability tools.

```
kite/
├── agents/          # 🤖 Reasoning patterns (ReAct, ReWOO, ToT, Plan-Execute)
├── memory/          # 🧠 Vector RAG, Graph RAG, Session Memory
├── safety/          # 🛡️ Circuit Breakers, Idempotency, Kill Switches
├── routing/         # 🧭 Semantic Routing, Aggregator Routing
├── tools/           # 🔧 Execution Kernel (Shell, Code, HTTP)
├── pipeline/        # ⚙️ Deterministic workflows with HITL support
└── monitoring/      # 📊 Metrics, Tracing, Event Bus
```

---

## 🛡️ Production Primitives (The "Boring" Stuff)

This is what makes Kite unique. We prioritize these over fancy new models.

### 1. Circuit Breakers
Prevent cascading failures and "infinite loops of death".
```python
# If usage spikes or errors spike, the agent is grounded.
ai.circuit_breaker.trip() 
```

### 2. Guardrails & Whitelists
Don't give the agent a shell. Give it a **jail**.
```python
# Whitelist specific commands. Everything else is rejected at the parser level.
safe_shell = ShellTool(allowed_commands=["grep", "cat", "ls"])
```

### 3. Human-in-the-Loop (HITL)
For high-stakes actions, the agent is just a form-filler. You sign the check.
```python
pipeline.add_checkpoint("approval_required")
# Agent pauses. Waits for API call from human dashboard.
```

---

## � Core Features

### 1️⃣ Multiple Reasoning Patterns
Choose the right "brain" for your task:
```python
# ReAct: Standard loop (Think → Act → Observe → Repeat)
agent = ai.create_agent(..., agent_type="react")

# ReWOO: Plan everything upfront, execute in parallel (FAST!)
agent = ai.create_agent(..., agent_type="rewoo")

# Tree-of-Thoughts: Explore multiple solutions (creative tasks)
agent = ai.create_agent(..., agent_type="tot")
```

### 2️⃣ Advanced Memory Systems
*   **Vector Memory**: For semantic search ("Find similar policy documents").
*   **Graph RAG**: For relationship-aware knowledge ("How is Entity A related to B?").
*   **Session Memory**: For robust conversation history management.

### 3️⃣ Smart Multi-Provider Support
Switch between providers without changing code, or route based on cost/complexity.
```python
# Automatic routing: Simple tasks go to fast models, complex ones to smart models.
router = ResourceAwareRouter(ai.config)
model = router.smart_model  # e.g., GPT-4o
```

---

## 📊 Production Examples

We built **6 real-world case studies** to show you exactly how to use Kite:

| Case | Scenario | Key Concepts | Difficulty |
|------|----------|--------------|-----------|
| **[Case 1](examples/case1_ecommerce_support.py)** | E-commerce Support Bot | LLM Routing, Tools, Multi-Agent | 🟢 Beginner |
| **[Case 2](examples/case2_enterprise_analytics.py)** | Data Analyst Agent | SQL + Python Execution, Charts | 🟡 Intermediate |
| **[Case 3](examples/case3_research_assistant.py)** | Deep Research System | Web Scraping, Multi-Step Planning | 🟡 Intermediate |
| **[Case 4](examples/case4_multi_agent_collab.py)** | Multi-Agent Collaboration | Supervisor Pattern, HITL | 🔴 Advanced |
| **[Case 5](examples/case5_devops_automation.py)** | DevOps Automation | Shell Tools, Safety Guardrails | 🟡 Intermediate |
| **[Case 6](examples/case6_reasoning_architectures.py)** | Reasoning Pattern Comparison | ReAct vs ReWOO vs ToT | 🔴 Advanced |

---

## 📈 Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Framework Startup** | ~50ms (lazy loading) |
| **Memory Footprint** | <100MB (base) |
| **Throughput** | 100+ req/s with caching |

---

## 🤝 Contributing

We want "Boring" code.
*   Reliability > Features
*   Explicit > Implicit
*   Types > Strings

See [CONTRIBUTING.md](CONTRIBUTING.md).

---
<p align="center">
  <strong>Kite: Because "I'm sorry, I can't do that" is better than a lawsuit.</strong>
</p>

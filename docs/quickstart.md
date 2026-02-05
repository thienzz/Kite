# Quick Start Guide

**Get started with Kite in 5 minutes**

This guide will help you build your first AI agent with Kite.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Kite

```bash
# Clone the repository
git clone https://github.com/thienzz/Kite.git
cd Kite

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file in the project root:

```bash
# Copy example configuration
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# LLM Provider (choose one)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-your-key-here

# Or use local Ollama (no API key needed)
# LLM_PROVIDER=ollama
# LLM_MODEL=qwen2.5:1.5b

# Embedding Provider
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## Your First Agent

### 1. Simple Completion

```python
from kite import Kite

# Initialize framework
ai = Kite()

# Simple completion
response = ai.complete("What is the capital of France?")
print(response)
```

### 2. Agent with Tools

```python
from kite import Kite

# Initialize
ai = Kite()

# Define a custom tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # In production, call a real weather API
    return f"Weather in {city}: Sunny, 72°F"

# Register tool
ai.create_tool("get_weather", get_weather, "Get current weather for a city")

# Create agent
agent = ai.create_agent(
    name="WeatherAssistant",
    system_prompt="You are a helpful weather assistant. Use the get_weather tool to answer questions.",
    tools=[ai.tools.get("get_weather")]
)

# Run agent
import asyncio

async def main():
    result = await agent.run("What's the weather in Paris?")
    print(result['response'])

asyncio.run(main())
```

### 3. Plan-and-Execute Agent

```python
from kite import Kite

ai = Kite()

# Create planning agent
planner = ai.create_planning_agent(
    strategy="plan-and-execute",
    name="ResearchAgent",
    system_prompt="You are a research analyst."
)

# Run complex task
async def main():
    result = await planner.run(
        "Research the AI agent market size in 2024 and suggest a pricing model"
    )
    print(result['answer'])

import asyncio
asyncio.run(main())
```

### 4. Memory-Enabled Agent

```python
from kite import Kite

ai = Kite()

# Load documents into vector memory
ai.load_document("docs/product_manual.pdf", doc_id="manual")

# Create agent
agent = ai.create_agent(
    name="SupportAgent",
    system_prompt="You are a support agent. Use the knowledge base to answer questions."
)

# Search memory
results = ai.vector_memory.search("How do I reset my password?", top_k=3)
print(results)

# Agent with context
async def main():
    context = {"knowledge_base": results}
    result = await agent.run("How do I reset my password?", context=context)
    print(result['response'])

import asyncio
asyncio.run(main())
```

---

## Common Patterns

### Async Agent Execution

```python
import asyncio
from kite import Kite

ai = Kite()
agent = ai.create_agent(
    name="Assistant",
    system_prompt="You are a helpful assistant."
)

# Run multiple queries concurrently
async def main():
    results = await asyncio.gather(
        agent.run("What is Python?"),
        agent.run("What is AI?"),
        agent.run("What is Kite?")
    )
    for result in results:
        print(result['response'])

asyncio.run(main())
```

### Circuit Breaker Protection

```python
from kite import Kite

ai = Kite()

# Configure circuit breaker
ai.circuit_breaker.config.failure_threshold = 3
ai.circuit_breaker.config.timeout_seconds = 60

# All LLM calls are automatically protected
response = ai.complete("Hello")  # Protected by circuit breaker
```

### Idempotency

```python
from kite import Kite

ai = Kite()

# Define an expensive operation
def process_payment(order_id, amount):
    # Simulate payment processing
    return {"order_id": order_id, "amount": amount, "status": "paid"}

# Prevent duplicate operations
result = ai.idempotency.execute(
    operation_id="process_order_12345",
    func=process_payment,
    args=("ORD-12345", 99.99)
)

print(result)  # Processes payment

# Same operation_id returns cached result
result2 = ai.idempotency.execute(
    operation_id="process_order_12345",
    func=process_payment,
    args=("ORD-12345", 99.99)
)

print(result2)  # Returns cached result, doesn't execute again
```

### Semantic Routing

```python
from kite import Kite

ai = Kite()

# Create semantic router
router = ai.semantic_router

# Add routes
router.add_route("support", "help issue problem bug customer service")
router.add_route("sales", "pricing purchase buy cost payment billing")
router.add_route("technical", "API integration code developer documentation")

# Route user input
intent = router.classify("How do I integrate the API?")
print(intent)  # "technical"
```

### Human-in-the-Loop Workflow

```python
from kite import Kite

ai = Kite()

# Define workflow steps
def research_step(data):
    return {"findings": "Market research results..."}

def execute_step(data):
    return {"result": "Executed based on research"}

# Create workflow
workflow = ai.create_workflow("approval_flow")
workflow.add_step("research", research_step)
workflow.add_checkpoint("research", approval_required=True)
workflow.add_step("execute", execute_step)

# Execute
async def main():
    state = await workflow.execute_async({"query": "Research AI market"})
    print(f"Status: {state.status}")  # AWAITING_APPROVAL
    
    # Resume after approval
    state = await workflow.resume_async(state.task_id, feedback="Approved")
    print(f"Final result: {state.data}")

import asyncio
asyncio.run(main())
```

---

## Next Steps

### Explore Examples

Run the example case studies:

```bash
# Invoice processing pipeline
python examples/case1_invoice_pipeline_framework.py

# Semantic routing
python examples/case2_semantic_router_framework.py

# SQL analytics
python examples/case3_sql_analytics_framework.py

# Research assistant
python examples/case4_research_assistant_framework.py

# Advanced planning patterns
python examples/case7_advanced_planning.py

# Human-in-the-loop
python examples/case9_human_in_loop.py
```

### Read Documentation

- [Architecture Guide](architecture.md) - System design and components
- [API Reference](api_reference.md) - Complete API documentation
- [Deployment Guide](deployment.md) - Production deployment
- [Safety Guide](safety.md) - Circuit breakers and idempotency
- [Memory Systems](memory.md) - Vector, Graph RAG, session memory

### Join the Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/thienzz/Kite/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/thienzz/Kite/discussions)

---

## Troubleshooting

### LLM Provider Issues

**Problem**: `LLM provider not initialized`

**Solution**: Check your `.env` file has the correct provider and API key:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

### Ollama Connection Issues

**Problem**: `Connection refused to localhost:11434`

**Solution**: Make sure Ollama is running:

```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull qwen2.5:1.5b
```

### Memory Issues

**Problem**: `Vector memory not initialized`

**Solution**: Vector memory loads lazily. Access it once to initialize:

```python
ai.vector_memory.add_document("test", "content")
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'kite'`

**Solution**: Make sure you're in the Kite directory and have installed dependencies:

```bash
cd Kite
pip install -r requirements.txt
```

---

## Tips for Success

1. **Start Simple**: Begin with basic completions before moving to agents
2. **Use Async**: Always use `async/await` for agent operations
3. **Monitor Performance**: Check `agent.call_count` and circuit breaker stats
4. **Test Locally**: Use Ollama for development, switch to cloud providers for production
5. **Read Examples**: The `examples/` directory has production-ready patterns

---

## Getting Help

If you're stuck:

1. Check the [documentation](architecture.md)
2. Search [GitHub Issues](https://github.com/thienzz/Kite/issues)
3. Ask in [Discussions](https://github.com/thienzz/Kite/discussions)
4. Review the [examples](../examples/)

---

**Ready to build production-grade AI agents? Let's go! 🚀**

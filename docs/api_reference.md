# 📖 Kite API Reference

**Comprehensive guide to all Kite methods and APIs**

This document covers every public API in the Kite framework with examples, parameters, and use cases.

---

## Table of Contents

1. [Core Kite Class](#1-core-kite-class)
2. [Agent APIs](#2-agent-apis)
3. [Tool System](#3-tool-system)
4. [Memory Systems](#4-memory-systems)
5. [Safety Mechanisms](#5-safety-mechanisms)
6. [Routing](#6-routing)
7. [Workflows & Pipelines](#7-workflows--pipelines)
8. [Monitoring & Metrics](#8-monitoring--metrics)
9. [Configuration](#9-configuration)

---

## 1. Core Kite Class

### Initialization

#### `Kite(config: Optional[Dict] = None)`

Initialize the Kite framework.

**Parameters:**
- `config` (Dict, optional): Configuration overrides. Auto-loads from `.env` if not provided.

**Returns:** Kite instance

**Example:**
```python
from kite import Kite

# Auto-load from .env
ai = Kite()

# With custom config
ai = Kite(config={
    'llm_provider': 'openai',
    'llm_model': 'gpt-4o',
    'circuit_breaker_threshold': 3,
    'max_iterations': 15
})
```

---

### LLM Methods

#### `chat(messages: List[Dict], **kwargs) -> str`

Chat with the configured LLM using message format.

**Parameters:**
- `messages`: List of message dicts with `role` and `content`
- `**kwargs`: Additional LLM parameters (temperature, max_tokens, etc.)

**Returns:** String response

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is quantum computing?"}
]

response = ai.chat(messages, temperature=0.7)
print(response)
```

---

#### `complete(prompt: str, **kwargs) -> str`

Simple text completion.

**Parameters:**
- `prompt`: Text prompt
- `**kwargs`: LLM parameters

**Returns:** String response

**Example:**
```python
response = ai.complete("Write a haiku about coding:")
```

---

### Embedding Methods

#### `embed(text: str) -> List[float]`

Generate embedding vector for text.

**Example:**
```python
vector = ai.embed("Machine learning is fascinating")
```

---

#### `embed_batch(texts: List[str]) -> List[List[float]]`

Generate embeddings for multiple texts efficiently.

**Example:**
```python
vectors = ai.embed_batch(["Hello world", "Python is great"])
```

---

### Agent Creation

#### `create_agent(...) -> Agent`

Create a custom agent.

**Parameters:**
- `name` (str): Agent identifier
- `system_prompt` (str): Instructions defining agent behavior
- `tools` (List, optional): List of Tool objects
- `model` (str, optional): Model specifier (e.g., `"groq/llama-3.3-70b"`)
- `agent_type` (str, optional): `"base" | "react" | "plan_execute" | "rewoo" | "tot"`
- `verbose` (bool, optional): Enable detailed logging

**Example:**
```python
import asyncio
from kite import Kite

ai = Kite()

# Define a search tool
def web_search(query: str) -> str:
    return f"Search results for: {query}"

search_tool = ai.create_tool("web_search", web_search)

async def main():
    agent = ai.create_agent(
        name="Researcher",
        system_prompt="You research topics thoroughly.",
        tools=[search_tool],
        agent_type="react",
        model="groq/llama-3.3-70b-versatile",
        verbose=True
    )
    
    result = await agent.run("Research quantum computing")
    print(result['response'])

asyncio.run(main())
```

---

#### `create_tool(name: str, func: Callable, description: str = None) -> Tool`

Wrap a Python function as a tool.

**Example:**
```python
def get_weather(city: str) -> str:
    return f"Sunny, 72°F in {city}"

tool = ai.create_tool("get_weather", get_weather, "Get current weather")
```

---

### Event System

#### `event_bus.subscribe(event_name: str, callback: Callable)`

Subscribe to framework events.

**Example:**
```python
def on_action(event, data):
    print(f"Tool: {data.get('tool')}")

ai.event_bus.subscribe("action", on_action)
```

---

#### `enable_tracing(filename: str = "process_trace.json")`

Enable JSON trace logging.

**Example:**
```python
ai.enable_tracing("my_trace.json")
```

---

### Knowledge Management

#### `load_document(path: str, doc_id: Optional[str] = None) -> bool`

Load document(s) into vector memory.

**Supported Formats:** `.txt`, `.md`, `.pdf`, `.docx`, `.json`, `.csv`

**Example:**
```python
ai.load_document("docs/manual.pdf", "manual")
ai.load_document("docs/", "knowledge")
```

---

### Properties

#### `ai.llm`

Configured LLM provider (lazy-loaded).

---

#### `ai.embeddings`

Embedding provider (lazy-loaded).

---

#### `ai.vector_memory`

Vector similarity search engine.

**Methods:**
- `add_document(doc_id, text, metadata=None)`
- `search(query, top_k=5)`
- `delete_document(doc_id)`

**Example:**
```python
ai.vector_memory.add_document("doc1", "Kite is a Python framework...")
results = ai.vector_memory.search("What is Kite?", top_k=3)
```

---

#### `ai.session_memory`

Conversation history manager.

**Methods:**
- `add_message(session_id, role, content)`
- `get_history(session_id, max_messages=None)`
- `clear_session(session_id)`

---

#### `ai.graph_rag`

Knowledge graph for relationship-aware RAG.

**Methods:**
- `add_entity(name, type, properties=None)`
- `add_relationship(entity1, relation, entity2)`
- `query(question)`

---

#### `ai.circuit_breaker`

Fault tolerance manager.

**Methods:**
- `execute(func, *args, **kwargs)`
- `get_stats()`
- `reset()`

---

#### `ai.idempotency`

Prevents duplicate operations.

**Methods:**
- `execute(operation_id, func, args=None, ttl=3600)`

---

## 2. Agent APIs

### Agent.run()

#### `run(user_input: str, context: Optional[Dict] = None) -> Dict`

Execute agent on input (async).

**Returns:** Dict with `response`, `total_tokens`, `tool_calls`

**Example:**
```python
import asyncio

async def main():
    result = await agent.run("What is quantum computing?")
    print(result['response'])

asyncio.run(main())
```

---

### Agent.get_metrics()

Get agent performance metrics.

**Example:**
```python
metrics = agent.get_metrics()
```

---

## 3. Tool System

### Tool Class

#### `Tool(name: str, func: Callable, description: str)`

Wrap a function as a tool.

---

### Built-in Tools

#### WebSearchTool

```python
from kite.tools import WebSearchTool
search = WebSearchTool()
```

---

#### PythonReplTool

Safe Python code execution.

```python
from kite.tools.code_execution import PythonReplTool
python = PythonReplTool()
```

---

#### ShellTool

Execute shell commands (with whitelisting).

```python
from kite.tools.system_tools import ShellTool
shell = ShellTool(allowed_commands=["ls", "git", "df"])
```

---

## 4. Memory Systems

### Vector Memory

#### `add_document(doc_id: str, text: str, metadata: Dict = None)`

Add document to vector index.

---

#### `search(query: str, top_k: int = 5) -> List[Tuple]`

Semantic search.

---

### Session Memory

#### `add_message(session_id: str, role: str, content: str)`

Store conversation message.

---

### Graph RAG

#### `add_entity(name: str, type: str, properties: Dict = None)`

Add entity to knowledge graph.

---

## 5. Safety Mechanisms

### Circuit Breaker

**Configuration:**
```python
ai.circuit_breaker.config.failure_threshold = 3
ai.circuit_breaker.config.timeout_seconds = 60
```

---

### Idempotency Manager

**Example:**
```python
result = ai.idempotency.execute(
    operation_id="process_payment_12345",
    func=charge_card,
    args=(card_number, amount)
)
```

---

## 6. Routing

### LLM Router

```python
import asyncio
from kite import Kite
from kite.routing.llm_router import LLMRouter

ai = Kite()

# Define handler function
async def billing_handler(query: str):
    return {"response": f"Billing query handled: {query}"}

async def main():
    router = LLMRouter(llm=ai.llm)
    router.add_route("billing", "Handle billing queries", billing_handler)
    result = await router.route("Where's my invoice?")
    print(result)

asyncio.run(main())
```

---

## 7. Workflows & Pipelines

### Creating Pipelines

```python
workflow = ai.pipeline.create("approval_flow")
workflow.add_step("draft", draft_function)
workflow.add_checkpoint("draft")
workflow.add_step("send", send_function)

state = await workflow.execute_async({"data": "..."})
```

---

## 8. Monitoring & Metrics

### Enable Monitoring

```python
ai.enable_verbose_monitoring()
ai.enable_tracing("trace.json")
```

---

### Metrics Collection

```python
metrics = ai.get_metrics()
ai.print_summary()
```

---

## 9. Configuration

### Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Embedding Provider
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Safety
CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT_SECONDS=60

# Memory
VECTOR_BACKEND=faiss
VECTOR_DIMENSION=384
```

---

## Complete Example

```python
import asyncio
from kite import Kite
from kite.tools import WebSearchTool

async def main():
    # Initialize
    ai = Kite()
    ai.enable_verbose_monitoring()
    
    # Create tools
    search = WebSearchTool()
    
    # Create agent
    agent = ai.create_agent(
        name="Researcher",
        system_prompt="Research topics thoroughly.",
        tools=[search],
        agent_type="react"
    )
    
    # Run
    result = await agent.run("Research quantum computing")
    print(result['response'])

asyncio.run(main())
```

---

For more information:
- [Main README](../README.md)
- [Examples](../examples/README.md)
- [Architecture Guide](architecture.md)

# API Reference

**Complete API documentation for Kite framework**

This document provides detailed API documentation for all Kite components.

---

## Table of Contents

- [Kite Core](#kite-core)
- [Agents](#agents)
- [Tools](#tools)
- [Memory](#memory)
- [Safety](#safety)
- [Pipelines](#pipelines)
- [Routing](#routing)

---

## Kite Core

### `class Kite(config: Optional[Dict] = None)`

Main framework class.

**Parameters:**
- `config` (Optional[Dict]): Configuration dictionary. If not provided, loads from `.env`

**Attributes:**
- `llm`: LLM provider instance
- `embeddings`: Embedding provider instance
- `vector_memory`: Vector memory instance
- `session_memory`: Session memory instance
- `graph_rag`: Graph RAG instance
- `circuit_breaker`: Circuit breaker instance
- `idempotency`: Idempotency manager instance
- `tools`: Tool registry

**Example:**
```python
ai = Kite()
ai = Kite(config={"llm_provider": "openai", "llm_model": "gpt-4"})
```

---

### LLM Operations

#### `complete(prompt: str) -> str`

Synchronous completion.

**Parameters:**
- `prompt` (str): Input prompt

**Returns:**
- `str`: LLM response

**Example:**
```python
response = ai.complete("What is AI?")
```

#### `chat(messages: List[Dict]) -> str`

Synchronous chat completion.

**Parameters:**
- `messages` (List[Dict]): List of message dicts with `role` and `content`

**Returns:**
- `str`: LLM response

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
]
response = ai.chat(messages)
```

#### `complete_async(prompt: str) -> str`

Async completion.

**Parameters:**
- `prompt` (str): Input prompt

**Returns:**
- `str`: LLM response

**Example:**
```python
response = await ai.complete_async("What is AI?")
```

#### `chat_async(messages: List[Dict]) -> str`

Async chat completion.

**Parameters:**
- `messages` (List[Dict]): List of message dicts

**Returns:**
- `str`: LLM response

**Example:**
```python
response = await ai.chat_async(messages)
```

---

### Agent Creation

#### `create_agent(name: str, system_prompt: str = "", tools: List = None, **kwargs) -> Agent`

Create a general-purpose agent.

**Parameters:**
- `name` (str): Agent name
- `system_prompt` (str): System prompt for the agent
- `tools` (List): List of tools available to the agent
- `**kwargs`: Additional configuration

**Returns:**
- `Agent`: Agent instance

**Example:**
```python
agent = ai.create_agent(
    name="Assistant",
    system_prompt="You are a helpful assistant",
    tools=[search_tool, calculator]
)
```

#### `create_planning_agent(strategy: str, name: Optional[str] = None, system_prompt: Optional[str] = None, tools: List = None, max_iterations: int = 10, **kwargs) -> Agent`

Create a planning agent with specific reasoning strategy.

**Parameters:**
- `strategy` (str): Reasoning strategy: `"plan-and-execute"`, `"rewoo"`, or `"tot"`
- `name` (Optional[str]): Agent name (auto-generated if not provided)
- `system_prompt` (Optional[str]): System prompt (defaults based on strategy)
- `tools` (List): List of tools
- `max_iterations` (int): Maximum iterations/depth
- `**kwargs`: Additional configuration (e.g., `branches` for ToT)

**Returns:**
- `Agent`: Planning agent instance

**Example:**
```python
# Plan-and-Execute
planner = ai.create_planning_agent(strategy="plan-and-execute")

# ReWOO (parallel execution)
rewoo = ai.create_planning_agent(strategy="rewoo")

# Tree-of-Thoughts
tot = ai.create_planning_agent(strategy="tot", max_iterations=3, branches=3)
```

---

### Tool Management

#### `create_tool(name: str, func: Callable, description: str) -> Tool`

Register a custom tool.

**Parameters:**
- `name` (str): Tool name
- `func` (Callable): Tool function
- `description` (str): Tool description

**Returns:**
- `Tool`: Tool instance

**Example:**
```python
def my_tool(arg: str) -> str:
    return f"Processed {arg}"

tool = ai.create_tool("my_tool", my_tool, "Process input")
```

---

### Workflow Management

#### `create_workflow(name: str) -> DeterministicPipeline`

Create a deterministic workflow.

**Parameters:**
- `name` (str): Workflow name

**Returns:**
- `DeterministicPipeline`: Pipeline instance

**Example:**
```python
workflow = ai.create_workflow("approval_flow")
```

---

### Memory Operations

#### `load_document(path: str, doc_id: Optional[str] = None) -> bool`

Load document(s) into vector memory.

**Parameters:**
- `path` (str): File or directory path
- `doc_id` (Optional[str]): Document ID prefix

**Returns:**
- `bool`: Success status

**Example:**
```python
ai.load_document("docs/manual.pdf", doc_id="manual")
ai.load_document("docs/", doc_id="knowledge_base")
```

---

## Agents

### `class Agent(name, system_prompt, llm, tools, framework)`

General-purpose agent.

**Attributes:**
- `name` (str): Agent name
- `system_prompt` (str): System prompt
- `llm`: LLM provider
- `tools` (Dict): Available tools
- `call_count` (int): Total calls
- `success_count` (int): Successful calls

**Methods:**

#### `async run(user_input: str, context: Optional[Dict] = None) -> Dict`

Run agent on input.

**Parameters:**
- `user_input` (str): User input
- `context` (Optional[Dict]): Additional context

**Returns:**
- `Dict`: Result with keys:
  - `response` (str): Agent response
  - `success` (bool): Success status
  - `tool_calls` (List): Tools called
  - `iterations` (int): Number of iterations

**Example:**
```python
result = await agent.run("What's the weather?")
print(result['response'])
```

---

### Plan-Execute Agent

#### `async run_plan(goal: str, context: Optional[Dict] = None) -> Dict`

Run plan-and-execute loop.

**Parameters:**
- `goal` (str): Goal to achieve
- `context` (Optional[Dict]): Additional context

**Returns:**
- `Dict`: Result with keys:
  - `success` (bool): Success status
  - `goal` (str): Original goal
  - `plan` (List[str]): Executed steps
  - `answer` (str): Final answer

**Example:**
```python
planner = ai.create_planning_agent(strategy="plan-and-execute")
result = await planner.run_plan("Research market and suggest pricing")
```

---

### Tree-of-Thoughts Agent

#### `async solve_tot(goal: str, max_steps: Optional[int] = None, num_thoughts: Optional[int] = None) -> Dict`

Run tree-of-thoughts reasoning.

**Parameters:**
- `goal` (str): Problem to solve
- `max_steps` (Optional[int]): Maximum depth (defaults to `max_iterations`)
- `num_thoughts` (Optional[int]): Branches per level (defaults to `branches`)

**Returns:**
- `Dict`: Result with keys:
  - `success` (bool): Success status
  - `goal` (str): Original goal
  - `explored_paths` (int): Number of paths explored
  - `best_path` (List[str]): Selected reasoning path
  - `answer` (str): Final answer

**Example:**
```python
tot = ai.create_planning_agent(strategy="tot", max_iterations=3)
result = await tot.solve_tot("Evaluate 3 mitigation strategies")
```

---

## Tools

### `class Tool(name, func, description)`

Tool wrapper.

**Attributes:**
- `name` (str): Tool name
- `func` (Callable): Tool function
- `description` (str): Tool description

**Methods:**

#### `execute(*args, **kwargs) -> Any`

Execute tool.

**Example:**
```python
result = tool.execute("input")
```

---

## Memory

### Vector Memory

#### `add_document(doc_id: str, text: str) -> None`

Add document to vector memory.

**Parameters:**
- `doc_id` (str): Document ID
- `text` (str): Document text

**Example:**
```python
ai.vector_memory.add_document("doc1", "Kite is a framework...")
```

#### `search(query: str, top_k: int = 5) -> List[Dict]`

Search for similar documents.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results

**Returns:**
- `List[Dict]`: Results with keys:
  - `doc_id` (str): Document ID
  - `text` (str): Document text
  - `score` (float): Similarity score

**Example:**
```python
results = ai.vector_memory.search("What is Kite?", top_k=3)
```

---

### Graph RAG

#### `add_entity(name: str, type: str, properties: Dict = None) -> None`

Add entity to knowledge graph.

**Parameters:**
- `name` (str): Entity name
- `type` (str): Entity type
- `properties` (Dict): Entity properties

**Example:**
```python
ai.graph_rag.add_entity("Kite", "framework", {"language": "Python"})
```

#### `add_relationship(source: str, relation: str, target: str) -> None`

Add relationship between entities.

**Parameters:**
- `source` (str): Source entity
- `relation` (str): Relationship type
- `target` (str): Target entity

**Example:**
```python
ai.graph_rag.add_relationship("Kite", "uses", "Circuit Breakers")
```

#### `query(question: str) -> str`

Query knowledge graph.

**Parameters:**
- `question` (str): Question

**Returns:**
- `str`: Answer

**Example:**
```python
answer = ai.graph_rag.query("What does Kite use?")
```

---

### Session Memory

#### `add_message(role: str, content: str) -> None`

Add message to session.

**Parameters:**
- `role` (str): Message role (`"user"` or `"assistant"`)
- `content` (str): Message content

**Example:**
```python
ai.session_memory.add_message("user", "Hello")
ai.session_memory.add_message("assistant", "Hi!")
```

#### `get_history() -> List[Dict]`

Get conversation history.

**Returns:**
- `List[Dict]`: Messages

**Example:**
```python
history = ai.session_memory.get_history()
```

---

## Safety

### Circuit Breaker

#### `get_stats() -> Dict`

Get circuit breaker statistics.

**Returns:**
- `Dict`: Statistics with keys:
  - `state` (str): Current state
  - `total_calls` (int): Total calls
  - `successful_calls` (int): Successful calls
  - `failed_calls` (int): Failed calls
  - `rejected_calls` (int): Rejected calls
  - `success_rate` (float): Success rate

**Example:**
```python
stats = ai.circuit_breaker.get_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
```

#### `reset() -> None`

Manually reset circuit breaker.

**Example:**
```python
ai.circuit_breaker.reset()
```

---

### Idempotency Manager

#### `execute(operation_id: str, func: Callable, args: tuple = (), kwargs: dict = None, ttl: int = 3600) -> Any`

Execute operation with idempotency.

**Parameters:**
- `operation_id` (str): Unique operation ID
- `func` (Callable): Function to execute
- `args` (tuple): Function arguments
- `kwargs` (dict): Function keyword arguments
- `ttl` (int): Time-to-live in seconds

**Returns:**
- `Any`: Function result (cached or fresh)

**Example:**
```python
result = ai.idempotency.execute(
    operation_id="order_12345",
    func=process_payment,
    args=(order_id, amount)
)
```

---

## Pipelines

### `class DeterministicPipeline(name: str)`

Deterministic workflow with HITL support.

**Methods:**

#### `add_step(name: str, func: Callable) -> None`

Add workflow step.

**Parameters:**
- `name` (str): Step name
- `func` (Callable): Step function

**Example:**
```python
workflow.add_step("research", research_func)
```

#### `add_checkpoint(step_name: str, approval_required: bool = True) -> None`

Add checkpoint after step.

**Parameters:**
- `step_name` (str): Step name
- `approval_required` (bool): Require approval

**Example:**
```python
workflow.add_checkpoint("research", approval_required=True)
```

#### `add_intervention_point(step_name: str, callback: Callable) -> None`

Add intervention point before step.

**Parameters:**
- `step_name` (str): Step name
- `callback` (Callable): Callback function

**Example:**
```python
workflow.add_intervention_point("execute", user_callback)
```

#### `async execute_async(data: Dict) -> PipelineState`

Execute workflow asynchronously.

**Parameters:**
- `data` (Dict): Initial data

**Returns:**
- `PipelineState`: Pipeline state

**Example:**
```python
state = await workflow.execute_async({"query": "..."})
```

#### `async resume_async(task_id: str, feedback: Optional[str] = None) -> PipelineState`

Resume paused workflow.

**Parameters:**
- `task_id` (str): Task ID
- `feedback` (Optional[str]): User feedback

**Returns:**
- `PipelineState`: Updated state

**Example:**
```python
state = await workflow.resume_async(task_id, feedback="Approved")
```

---

## Routing

### Semantic Router

#### `route(text: str) -> str`

Route text to intent.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `str`: Intent name

**Example:**
```python
router = ai.create_semantic_router([
    ("support", "help, issue"),
    ("sales", "pricing, buy")
])
intent = router.route("I need help")  # "support"
```

---

## Configuration

### Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=openai|anthropic|groq|together|ollama
LLM_MODEL=model-name
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Embedding Provider
EMBEDDING_PROVIDER=fastembed|openai|cohere
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Safety
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT=60
IDEMPOTENCY_TTL=3600

# Memory
VECTOR_BACKEND=faiss|chroma
VECTOR_DIMENSION=384
REDIS_URL=redis://localhost:6379

# Database
POSTGRES_URL=postgresql://user:pass@localhost/db
```

---

## Error Handling

### CircuitBreakerError

Raised when circuit breaker blocks an operation.

```python
from kite.safety.circuit_breaker import CircuitBreakerError

try:
    result = ai.complete(prompt)
except CircuitBreakerError as e:
    print(f"Circuit open: {e}")
```

---

## Best Practices

1. **Always use async for agents**: `await agent.run()`
2. **Monitor circuit breaker**: Check `ai.circuit_breaker.get_stats()`
3. **Use idempotency for writes**: Prevent duplicate operations
4. **Configure timeouts**: Set appropriate LLM timeouts
5. **Handle errors gracefully**: Use try/except blocks

---

For more information, see:
- [Architecture Guide](architecture.md)
- [Quick Start](quickstart.md)
- [Deployment Guide](deployment.md)

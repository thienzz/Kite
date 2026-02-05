# Memory Systems

**Advanced memory capabilities in Kite**

This guide covers Kite's memory systems: Vector Memory, Graph RAG, and Session Memory.

---

## Table of Contents

- [Vector Memory](#vector-memory)
- [Graph RAG](#graph-rag)
- [Session Memory](#session-memory)
- [Best Practices](#best-practices)

---

## Vector Memory

### Overview

Vector Memory enables semantic search using embeddings. Documents are converted to vectors and stored for similarity-based retrieval.

**Use Cases:**
- Knowledge base search
- Document Q&A
- Semantic routing
- RAG (Retrieval-Augmented Generation)

### Configuration

```python
from kite import Kite

ai = Kite()

# Configure vector memory
ai.config['vector_backend'] = 'faiss'  # or 'chroma'
ai.config['vector_dimension'] = 384    # Embedding dimension
ai.config['embedding_provider'] = 'fastembed'
ai.config['embedding_model'] = 'BAAI/bge-small-en-v1.5'
```

### Adding Documents

```python
# Add single document
ai.vector_memory.add_document(
    doc_id="doc1",
    text="Kite is a production-ready agentic AI framework..."
)

# Add multiple documents
documents = {
    "doc1": "Content 1...",
    "doc2": "Content 2...",
    "doc3": "Content 3..."
}

for doc_id, text in documents.items():
    ai.vector_memory.add_document(doc_id, text)
```

### Loading from Files

```python
# Load single file
ai.load_document("docs/manual.pdf", doc_id="manual")

# Load directory
ai.load_document("docs/", doc_id="knowledge_base")
```

**Supported Formats:**
- PDF (`.pdf`)
- Text (`.txt`, `.md`)
- Word (`.docx`)
- HTML (`.html`)

### Searching

```python
# Semantic search
results = ai.vector_memory.search(
    query="How do I deploy Kite?",
    top_k=5
)

# Results format
for result in results:
    print(f"Doc ID: {result['doc_id']}")
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
```

### RAG Pattern

```python
# 1. Search knowledge base
results = ai.vector_memory.search(query, top_k=3)

# 2. Build context
context = "\n\n".join([r['text'] for r in results])

# 3. Generate answer
prompt = f"""Answer the question using this context:

Context:
{context}

Question: {query}

Answer:"""

answer = ai.complete(prompt)
```

### Backends

#### FAISS (Default)

```python
ai.config['vector_backend'] = 'faiss'
```

**Pros:**
- Fast similarity search
- Low memory footprint
- Good for development

**Cons:**
- Not persistent (in-memory)
- Single-machine only

#### ChromaDB

```python
ai.config['vector_backend'] = 'chroma'
ai.config['chroma_persist_directory'] = './chroma_db'
```

**Pros:**
- Persistent storage
- Metadata filtering
- Good for production

**Cons:**
- Slower than FAISS
- Higher memory usage

---

## Graph RAG

### Overview

Graph RAG combines knowledge graphs with retrieval-augmented generation for relationship-aware answers.

**Use Cases:**
- Complex knowledge bases
- Multi-hop reasoning
- Relationship queries
- Entity-centric Q&A

### Building a Knowledge Graph

```python
from kite import Kite

ai = Kite()

# Add entities
ai.graph_rag.add_entity(
    name="Kite",
    type="framework",
    properties={"language": "Python", "license": "MIT"}
)

ai.graph_rag.add_entity(
    name="Circuit Breaker",
    type="safety_feature",
    properties={"states": ["CLOSED", "OPEN", "HALF_OPEN"]}
)

ai.graph_rag.add_entity(
    name="Vector Memory",
    type="memory_system",
    properties={"backends": ["FAISS", "ChromaDB"]}
)

# Add relationships
ai.graph_rag.add_relationship("Kite", "uses", "Circuit Breaker")
ai.graph_rag.add_relationship("Kite", "includes", "Vector Memory")
ai.graph_rag.add_relationship("Circuit Breaker", "protects", "LLM calls")
```

### Querying

```python
# Simple query
answer = ai.graph_rag.query("What does Kite use?")
# "Kite uses Circuit Breaker and includes Vector Memory"

# Complex query
answer = ai.graph_rag.query("How does Kite ensure reliability?")
# "Kite uses Circuit Breaker which protects LLM calls..."
```

### Graph Traversal

```python
# Get entity
entity = ai.graph_rag.get_entity("Kite")

# Get relationships
relationships = ai.graph_rag.get_relationships("Kite")

# Get neighbors
neighbors = ai.graph_rag.get_neighbors("Kite", relation="uses")
```

### Loading from Documents

```python
# Extract entities and relationships from text
text = """
Kite is a framework that uses Circuit Breakers for safety.
Circuit Breakers protect LLM calls from cascading failures.
"""

ai.graph_rag.extract_from_text(text)
```

---

## Session Memory

### Overview

Session Memory maintains conversation context across multiple turns.

**Use Cases:**
- Chatbots
- Conversational agents
- Multi-turn dialogues
- Context-aware responses

### Basic Usage

```python
from kite import Kite

ai = Kite()

# Add messages
ai.session_memory.add_message("user", "Hello")
ai.session_memory.add_message("assistant", "Hi! How can I help?")
ai.session_memory.add_message("user", "What's the weather?")

# Get history
history = ai.session_memory.get_history()

# Use in agent
async def main():
    agent = ai.create_agent("Assistant")
    result = await agent.run(
        "What did I just ask?",
        context={"history": history}
    )
    print(result['response'])

import asyncio
asyncio.run(main())
```

### Session Management

```python
# Create session
session_id = ai.session_memory.create_session()

# Add to specific session
ai.session_memory.add_message(
    "user",
    "Hello",
    session_id=session_id
)

# Get session history
history = ai.session_memory.get_history(session_id=session_id)

# Clear session
ai.session_memory.clear_session(session_id)
```

### Context Window Management

```python
# Limit context window
ai.session_memory.config.max_messages = 10

# Get recent messages
recent = ai.session_memory.get_recent(n=5)

# Summarize old messages
summary = ai.session_memory.summarize_history()
ai.session_memory.add_message("system", f"Previous conversation: {summary}")
ai.session_memory.clear_old_messages(keep_last=2)
```

### Persistence

```python
# Save to database
ai.session_memory.config.storage = 'postgres'
ai.config['postgres_url'] = 'postgresql://user:pass@localhost/db'

# Or use Redis
ai.session_memory.config.storage = 'redis'
ai.config['redis_url'] = 'redis://localhost:6379'
```

---

## Best Practices

### 1. Choose the Right Memory System

```python
# Simple Q&A → Vector Memory
results = ai.vector_memory.search(query)

# Complex relationships → Graph RAG
answer = ai.graph_rag.query(query)

# Conversation → Session Memory
history = ai.session_memory.get_history()
```

### 2. Chunk Documents Appropriately

```python
# Bad: Entire document as one chunk
ai.vector_memory.add_document("doc1", entire_book)

# Good: Split into paragraphs/sections
for i, paragraph in enumerate(paragraphs):
    ai.vector_memory.add_document(f"doc1_p{i}", paragraph)
```

### 3. Use Metadata for Filtering

```python
# Add metadata
ai.vector_memory.add_document(
    "doc1",
    text,
    metadata={"category": "technical", "date": "2024-01-01"}
)

# Filter search
results = ai.vector_memory.search(
    query,
    filter={"category": "technical"}
)
```

### 4. Implement Hybrid Search

```python
# Combine vector search + keyword search
vector_results = ai.vector_memory.search(query, top_k=10)
keyword_results = keyword_search(query, top_k=10)

# Merge and re-rank
combined = merge_results(vector_results, keyword_results)
```

### 5. Monitor Memory Usage

```python
# Check vector memory size
size = ai.vector_memory.get_size()
print(f"Documents: {size['num_documents']}")
print(f"Memory: {size['memory_mb']} MB")

# Prune if needed
if size['num_documents'] > 10000:
    ai.vector_memory.prune_old_documents(keep_last=5000)
```

### 6. Use Graph RAG for Complex Queries

```python
# Simple fact → Vector Memory
answer = ai.vector_memory.search("What is Kite?")

# Multi-hop reasoning → Graph RAG
answer = ai.graph_rag.query(
    "What safety features does Kite use and how do they work?"
)
```

### 7. Manage Session Context

```python
# Limit context window to prevent token overflow
MAX_CONTEXT_MESSAGES = 10

history = ai.session_memory.get_recent(n=MAX_CONTEXT_MESSAGES)

# Or summarize old context
if len(history) > MAX_CONTEXT_MESSAGES:
    summary = ai.session_memory.summarize_history()
    ai.session_memory.clear_session()
    ai.session_memory.add_message("system", f"Context: {summary}")
```

---

## Production Checklist

- [ ] Vector backend configured (FAISS for dev, ChromaDB for prod)
- [ ] Embedding provider configured
- [ ] Document chunking strategy defined
- [ ] Metadata schema designed
- [ ] Graph RAG entities and relationships modeled
- [ ] Session storage configured (Redis/Postgres for prod)
- [ ] Context window limits set
- [ ] Memory usage monitoring implemented
- [ ] Pruning strategy defined

---

For more information, see:
- [Architecture Guide](architecture.md)
- [API Reference](api_reference.md)
- [Quick Start](quickstart.md)

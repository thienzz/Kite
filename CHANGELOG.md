# Changelog

All notable changes to Kite Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-10

### 🚀 New Features (LLM Integrations)
- **Groq Integration**: Added native support for Groq API (Llama 3.1 8B/70B).
- **OpenAI Integration**: Added support for OpenAI models (GPT-4o, GPT-5-nano).
- **Anthropic Integration**: Added support for Claude 3.5 Sonnet and Haiku.
- **Verification Script**: Added `examples/verify_providers.py` to test connectivity with all providers.

### 🛠 Improvements
- Updated `LLMFactory` to support new providers.
- Improved error handling for API quotas and model not found errors.
- Smoother fallback mechanisms in `create_agent`.

### 📦 Dependencies
- Added `groq`, `openai`, `anthropic` to `requirements.txt` and `setup.py`.

## [0.1.0] - 2025-01-23

### Added

#### Core Framework
- Initial release of Kite Framework
- Lazy-loading architecture with ~50ms startup time
- Multi-provider LLM support (OpenAI, Anthropic, Groq, Ollama)
- Configuration management via environment variables

#### Safety Mechanisms
- **Circuit Breaker**: Production-grade circuit breaker with 3 states (CLOSED, OPEN, HALF_OPEN)
  - Automatic failure detection and recovery
  - Configurable thresholds and timeouts
  - Statistics tracking and monitoring
- **Idempotency Manager**: Duplicate operation prevention
  - Memory and Redis backend support
  - Configurable TTL for cached results
- **Kill Switch**: Emergency stop mechanism
  - Max iterations enforcement
  - Cost limit enforcement
  - Custom stop conditions

#### Agent Reasoning Patterns
- **ReAct Agent**: Reasoning + Acting loop with tool use
- **Plan-Execute Agent**: Task decomposition with replanning
- **ReWOO Agent**: Parallel execution with dependency resolution
- **Tree-of-Thoughts Agent**: Multi-path reasoning exploration

#### Memory Systems
- **Vector Memory**: Semantic search with FAISS/ChromaDB
  - Document storage and retrieval
  - Embedding-based similarity search
- **Graph RAG**: Knowledge graph for multi-hop reasoning
  - Entity and relationship management
  - Graph traversal queries
- **Session Memory**: Conversation context management

#### Pipeline & Routing
- **Deterministic Pipelines**: Workflow orchestration
  - Sequential step execution
  - Checkpoint support for Human-in-the-Loop (HITL)
  - Intervention points for state modification
- **Semantic Router**: Intent classification using embeddings
- **Aggregator Router**: Multi-agent response combination

#### Tools & Integrations
- Built-in tools: web_search, calculator, datetime
- MCP (Model Context Protocol) integration
  - PostgreSQL MCP server
  - Gmail MCP client (basic)
  - Google Drive MCP client (basic)

#### Monitoring & Observability
- Metrics collection for agents and operations
- Circuit breaker statistics
- Cost tracking

#### Documentation
- Comprehensive README with quick start
- API reference documentation
- Architecture guide
- Safety guide
- Memory systems guide
- 9 production-ready example case studies

### Known Issues
- No comprehensive unit test suite
- Tool ecosystem limited (only 3 built-in tools)
- MCP servers are basic implementations

### Notes
This is an **alpha release** (v0.1.0). The framework is functional but has known limitations:
- Not recommended for production use without thorough testing
- API may change in future releases
- Some features are basic implementations
- Comprehensive testing needed before production deployment

For production use, we recommend:
1. Add your own unit tests for critical paths
2. Implement proper async in LLM providers if using async extensively
3. Add persistence layer for vector memory
4. Monitor circuit breaker statistics closely

### Contributors
- Thien Nguyen (@thienzz)

[0.1.1]: https://github.com/thienzz/Kite/releases/tag/v0.1.1
[0.1.0]: https://github.com/thienzz/Kite/releases/tag/v0.1.0
"""
Kite Framework - Core
General-purpose framework for ANY agentic AI application.
"""

import os
import logging
import json
from typing import Dict, Optional, Any, Callable, List
from datetime import datetime
from .data_loaders import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EventBus:
    """Simple asynchronous event bus for pub/sub monitoring."""
    def __init__(self):
        self._subscribers = {}
        self._relays = []
        self.logger = logging.getLogger("EventBus")

    def add_relay(self, url: str):
        if url not in self._relays:
            self._relays.append(url)
            self.logger.info(f"Added event relay to: {url}")

    def subscribe(self, event_name: str, callback: Callable):
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)
        self.logger.debug(f"Subscribed to {event_name}")

    def emit(self, event_name: str, data: Any):
        self.logger.debug(f"Emitting {event_name}")
        callbacks = self._subscribers.get(event_name, [])
        # Also support catch-all "*" subscribers
        callbacks += self._subscribers.get("*", [])
        
        for cb in callbacks:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(event_name, data))
                else:
                    cb(event_name, data)
            except Exception as e:
                self.logger.error(f"Error in event callback for {event_name}: {e}")
        
        # Also emit to external relays
        if self._relays:
            for relay_url in self._relays:
                try:
                    import asyncio
                    asyncio.create_task(self._relay_event(relay_url, event_name, data))
                except: pass

    async def _relay_event(self, url: str, event: str, data: Any):
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json={"event": event, "data": data, "timestamp": datetime.now().isoformat()}, timeout=2.0)
                if resp.status_code != 200:
                    self.logger.debug(f"Relay {url} returned {resp.status_code}")
        except Exception as e:
            # Fallback print for debugging
            print(f"   [Relay Error] Failed to send {event} to {url}: {e}")


class KnowledgeStore:
    """Lightweight knowledge manager for query templates and context."""
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.data = {}
        self.load()

    def load(self):
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            return
        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.knowledge_dir, filename)
                try:
                    with open(path, "r") as f:
                        name = filename.replace(".json", "")
                        self.data[name] = json.load(f)
                except Exception as e:
                    print(f"Error loading knowledge {filename}: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        # Support dot notation: "linkedin_queries.b2b_software"
        parts = key_path.split(".")
        current = self.data
        for p in parts:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                return default
        return current


class Kite:
    """
    General-Purpose Agentic AI Framework (Kite)
    
    Build ANY agentic application - not limited to customer support.
    
    Foundation components:
    - ai.llm            # LLM provider
    - ai.embeddings     # Embedding provider  
    - ai.circuit_breaker # Safety
    - ai.idempotency    # Safety
    - ai.vector_memory  # Memory
    - ai.session_memory # Memory
    - ai.graph_rag      # Memory
    - ai.semantic_router # Routing
    - ai.tools          # Tool registry
    - ai.slm            # SLM specialists
    - ai.pipeline       # Workflow system
    - ai.chat(messages)
    - ai.complete(prompt)
    - ai.embed(text)
    - ai.create_agent(...)
    - ai.create_tool(...)
    - ai.create_workflow(...)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Always load environment defaults first
        self.config = self._load_config()
        # Merge with user-provided config if any
        if config:
            self.config.update(config)
        self.logger = logging.getLogger("Kite")
        
        # Lazy storage
        self._llm = None
        self._embeddings = None
        self._vector_memory = None
        self._session_memory = None
        self._graph_rag = None
        self._semantic_router = None
        self._aggregator_router = None
        self._tools = None
        self._pipeline = None
        self._advanced_rag = None
        self._cache = None
        self._db_mcp = None
        
        # New Event System for Pub/Sub monitoring
        self.event_bus = EventBus()
        self.knowledge = KnowledgeStore()
        
        self.data_loader = DocumentLoader()
        
        # Only initialize core safety eagerly
        self._init_safety()
        
        self.logger.info("[OK] Kite initialized (lazy-loading enabled)")

    def add_knowledge_source(self, source_type: str, path: str, name: str):
        """Explicitly register and index a knowledge source."""
        self.logger.info(f"Adding knowledge source: {name} (Type: {source_type})")
        
        if source_type == "local_json":
            if not os.path.exists(path):
                self.logger.error(f"Knowledge file not found: {path}")
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
                # Store in KnowledgeStore for structured access
                self.knowledge.data[name] = data
                
                # Also index into VectorMemory for semantic retrieval
                for key, val in data.items():
                    doc_id = f"k_{name}_{key}"
                    text = f"Expert info for {key}: {val}"
                    self.vector_memory.add_document(doc_id, text, metadata={"source": name, "key": key})
        
        elif source_type == "vector_db":
            self.logger.info(f"Connected to external vector source: {path}")
            # Logic for external DB connection
            pass
        
        elif source_type == "mcp_resource":
            self.logger.info(f"Registering MCP Resource: {path}")
            # Logic for MCP integration
            pass
    
    def add_event_relay(self, url: str):
        """Add an external HTTP endpoint to relay all events to."""
        if hasattr(self, 'event_bus'):
            self.event_bus.add_relay(url)
            self.logger.info(f"Added event relay to: {url}")

    def load_document(self, path: str, doc_id: Optional[str] = None):
        """Load and store document(s) using DocumentLoader."""
        if os.path.isdir(path):
            self.logger.info(f"Loading directory: {path}")
            data = self.data_loader.load_directory(path)
            if not data:
                self.logger.warning(f"  No supported files found in {path}")
                return False
                
            for filename, text in data.items():
                if text.startswith("Error"):
                    self.logger.warning(f"  Skipping {filename}: {text}")
                    continue
                if text.startswith("Unsupported"):
                    self.logger.warning(f"  Skipping {filename} (Unsupported extension)")
                    continue
                    
                did = f"{doc_id}_{filename}" if doc_id else filename
                self.logger.info(f"  Adding document: {did}")
                self.vector_memory.add_document(did, text)
            return True
        else:
            self.logger.info(f"Loading file: {path}")
            text = self.data_loader.load_any(path)
            if text.startswith("Error") or text.startswith("Unsupported"):
                self.logger.error(text)
                return False
                
            did = doc_id or os.path.basename(path)
            self.vector_memory.add_document(did, text)
            return True
    
    def _load_config(self) -> Dict:
        from dotenv import load_dotenv
        load_dotenv(override=True) # Load standard .env and override environment
        return {
            'llm_provider': os.getenv('LLM_PROVIDER', 'ollama'),
            'llm_model': os.getenv('LLM_MODEL'),
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'sentence-transformers'),
            'embedding_model': os.getenv('EMBEDDING_MODEL'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'vector_backend': os.getenv('VECTOR_BACKEND', 'memory'), # Default to memory if not set
            'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', 3)),
            'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT_SECONDS', 60)),
            'llm_timeout': float(os.getenv('LLM_TIMEOUT', 600.0)),
            'semantic_router_threshold': float(os.getenv('SEMANTIC_ROUTER_THRESHOLD', 0.3)),
            'router_type': os.getenv('ROUTER_TYPE', 'semantic'), # 'semantic' or 'llm'
            'max_iterations': int(os.getenv('MAX_ITERATIONS', 10)),
        }
    
    @property
    def llm(self):
        if self._llm is None:
            from .llm_providers import LLMFactory
            try:
                llm_provider = self.config.get('llm_provider')
                if llm_provider:
                    api_key = self.config.get(f'{llm_provider}_api_key') or os.getenv(f'{llm_provider.upper()}_API_KEY')
                    self._llm = LLMFactory.create(
                        llm_provider,
                        self.config.get('llm_model'),
                        api_key=api_key,
                        timeout=self.config.get('llm_timeout', 180.0)
                    )
                else:
                    self._llm = LLMFactory.auto_detect()
            except Exception as e:
                self.logger.warning(f"    LLM initialization failed: {e}")
                self._llm = LLMFactory.auto_detect()
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            from .embedding_providers import EmbeddingFactory
            try:
                emb_provider = self.config.get('embedding_provider')
                if emb_provider:
                    self._embeddings = EmbeddingFactory.create(
                        emb_provider,
                        self.config.get('embedding_model')
                    )
                else:
                    self._embeddings = EmbeddingFactory.auto_detect()
            except Exception as e:
                self.logger.warning(f"    Embedding initialization failed: {e}")
                self._embeddings = EmbeddingFactory.auto_detect()
        return self._embeddings

    @property
    def vector_memory(self):
        if self._vector_memory is None:
            from .memory.vector_memory import VectorMemory
            backend = self.config.get('vector_backend', 'memory')
            self._vector_memory = VectorMemory(
                backend=backend,
                embedding_provider=self.embeddings
            )
            self.logger.info(f"  [OK] Vector Memory initialized (Backend: {backend})")
        return self._vector_memory

    @property
    def session_memory(self):
        if self._session_memory is None:
            from .memory.session_memory import SessionMemory
            self._session_memory = SessionMemory()
        return self._session_memory

    @property
    def graph_rag(self):
        if self._graph_rag is None:
            from .memory.graph_rag import GraphRAG
            self._graph_rag = GraphRAG(llm=self.llm)
        return self._graph_rag

    @property
    def semantic_router(self):
        if self._semantic_router is None:
            router_type = self.config.get('router_type', 'semantic')
            
            if router_type == 'llm':
                from .routing import LLMRouter
                self._semantic_router = LLMRouter(llm=self.llm)
            else:
                from .routing import SemanticRouter
                self._semantic_router = SemanticRouter(
                    confidence_threshold=self.config.get('semantic_router_threshold', 0.3),
                    embedding_provider=self.embeddings
                )
        return self._semantic_router

    @property
    def aggregator_router(self):
        if self._aggregator_router is None:
            from .routing import AggregatorRouter
            self._aggregator_router = AggregatorRouter(llm=self.llm)
        return self._aggregator_router

    @property
    def tools(self):
        if self._tools is None:
            from .tool_registry import ToolRegistry
            self._tools = ToolRegistry(self.config, self.logger)
            # Centralized registration of official contrib tools
            self._tools.load_standard_tools(self)
        return self._tools

    @property
    def pipeline(self):
        if self._pipeline is None:
            from .pipeline_manager import PipelineManager
            from .pipeline import DeterministicPipeline
            self._pipeline = PipelineManager(DeterministicPipeline, self.logger)
        return self._pipeline

    @property
    def advanced_rag(self):
        if self._advanced_rag is None:
            from .memory.advanced_rag import AdvancedRAG
            self._advanced_rag = AdvancedRAG(self.vector_memory, self.llm)
        return self._advanced_rag

    @property
    def cache(self):
        if self._cache is None:
            from .caching.cache_manager import CacheManager
            self._cache = CacheManager(self.vector_memory)
        return self._cache

    @property
    def db_mcp(self):
        if self._db_mcp is None:
            from .tools.mcp.database_mcp import DatabaseMCP
            self._db_mcp = DatabaseMCP()
        return self._db_mcp
    
    def _init_safety(self):
        """Initialize safety patterns."""
        from .safety import CircuitBreaker, IdempotencyManager, CircuitBreakerConfig, IdempotencyConfig
        
        self.circuit_breaker = CircuitBreaker(
            name="system",
            config=CircuitBreakerConfig(
                failure_threshold=self.config.get('circuit_breaker_threshold', 5),
                timeout_seconds=self.config.get('circuit_breaker_timeout', 60)
            )
        )
        
        self.idempotency = IdempotencyManager(
            config=IdempotencyConfig(storage_backend='memory')
        )
        
        self.logger.info("  [OK] Safety")
    
    
    # ==========================================================================
    # GENERAL-PURPOSE METHODS
    # ==========================================================================
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat with LLM."""
        return self.llm.chat(messages, **kwargs)
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Complete prompt."""
        return self.llm.complete(prompt, **kwargs)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        return self.embeddings.embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embeddings.embed_batch(texts)
    
    def create_agent(self, 
                     name: str, 
                     system_prompt: str, 
                     tools: List = None,
                     llm_provider: str = None,
                     llm_model: str = None,
                     agent_type: str = "base",
                     knowledge_sources: List[str] = None):
        """
        Create custom agent with optional specific AI configuration.
        Supported types: 'base', 'react', 'plan_execute', 'rewoo', 'tot'
        """
        from .agent import Agent
        from .agents.react_agent import ReActAgent
        from .agents.plan_execute import PlanExecuteAgent
        from .agents.rewoo import ReWOOAgent
        from .agents.tot import TreeOfThoughtsAgent
        from .llm_providers import LLMFactory
        
        # Determine LLM
        agent_llm = self.llm
        if llm_provider:
            try:
                agent_llm = LLMFactory.create(llm_provider, llm_model)
            except Exception as e:
                self.logger.warning(f"Failed to create specific LLM for {name}: {e}. Using default.")
        
        tools_list = tools or []
        
        max_iter = self.config.get('max_iterations', 10)
        
        if agent_type == "react":
            return ReActAgent(name, system_prompt, agent_llm, tools_list, self, max_iterations=max_iter, knowledge_sources=knowledge_sources)
        elif agent_type == "plan_execute":
            return PlanExecuteAgent(name, system_prompt, agent_llm, tools_list, self, max_iterations=max_iter)
        elif agent_type == "rewoo":
            return ReWOOAgent(name, system_prompt, agent_llm, tools_list, self, max_iterations=max_iter)
        elif agent_type == "tot":
            return TreeOfThoughtsAgent(name, system_prompt, agent_llm, tools_list, self, max_iterations=max_iter)
            
        return Agent(name, system_prompt, agent_llm, tools_list, self, max_iterations=max_iter, knowledge_sources=knowledge_sources)
    
    def create_react_agent(self, 
                           name: str, 
                           system_prompt: str, 
                           tools: List = None,
                           kill_switch = None,
                           **kwargs):
        """
        Create a ReActAgent for autonomous tasks.
        """
        from .agents.react_agent import ReActAgent
        from .safety.kill_switch import KillSwitch
        
        # Use provided or default kill switch
        ks = kill_switch or KillSwitch()
        
        # We can reuse the same LLM/SLM logic from create_agent if needed,
        # but for now let's keep it simple and use the base agent creation.
        base_agent = self.create_agent(name, system_prompt, tools, **kwargs)
        
        return ReActAgent(
            name=base_agent.name,
            system_prompt=base_agent.system_prompt,
            llm=base_agent.llm,
            tools=list(base_agent.tools.values()),
            framework=self,
            kill_switch=ks
        )
    
    def create_plan_execute_agent(self, 
                                  name: str = "PlanExecuteAgent", 
                                  system_prompt: str = "You are a master planner. Decompose the following goal into a sequence of actionable steps.", 
                                  tools: List = None,
                                  max_iterations: int = 10,
                                  **kwargs):
        """Create a Plan-and-Execute agent."""
        from .agents.plan_execute import PlanExecuteAgent
        base = self.create_agent(name, system_prompt, tools, **kwargs)
        return PlanExecuteAgent(
            name=base.name,
            system_prompt=base.system_prompt,
            llm=base.llm,
            tools=list(base.tools.values()),
            framework=self,
            max_iterations=max_iterations
        )
        
    def create_rewoo_agent(self, 
                           name: str = "ReWOOAgent", 
                           system_prompt: str = "You are a ReWOO planner. Create a plan to achieve the goal using available tools.", 
                           tools: List = None,
                           max_iterations: int = 10,
                           **kwargs):
        """Create a ReWOO (Reasoning WithOut Observation) agent."""
        from .agents.rewoo import ReWOOAgent
        base = self.create_agent(name, system_prompt, tools, **kwargs)
        return ReWOOAgent(
            name=base.name,
            system_prompt=base.system_prompt,
            llm=base.llm,
            tools=list(base.tools.values()),
            framework=self,
            max_iterations=max_iterations
        )
        
    def create_tot_agent(self, 
                         name: str = "TreeOfThoughtsAgent", 
                         system_prompt: str = "You are a deliberate thinker. Generate distinct, creative next steps or reasoning thoughts to achieve the goal.", 
                         max_iterations: int = 3,
                         **kwargs):
        """Create a Tree-of-Thoughts agent."""
        from .agents.tot import TreeOfThoughtsAgent
        # ToT doesn't necessarily need tools in the same way, but we'll allow them
        base = self.create_agent(name, system_prompt, **kwargs)
        return TreeOfThoughtsAgent(
            name=base.name,
            system_prompt=base.system_prompt,
            llm=base.llm,
            tools=list(base.tools.values()),
            framework=self,
            max_iterations=max_iterations
        )
    
    def create_planning_agent(self, 
                              strategy: str = "plan-and-execute",
                              name: Optional[str] = None, 
                              system_prompt: Optional[str] = None, 
                              tools: List = None,
                              max_iterations: int = 10,
                              **kwargs):
        """
        Unified factory for planning agents.
        
        Strategies:
        - "plan-and-execute": Step-by-step planning.
        - "rewoo": Reasoning WithOut Observation (Parallel).
        - "tot": Tree-of-Thoughts (Multi-path reasoning).
        """
        strategy = strategy.lower()
        
        # Set default name/prompt if not provided
        if not name:
            name = f"{strategy.replace('-', ' ').title().replace(' ', '')}Agent"
        if not system_prompt:
            if strategy == "plan-and-execute":
                system_prompt = "You are a master planner. Decompose the following goal into a sequence of actionable steps."
            elif strategy == "rewoo":
                system_prompt = "You are a ReWOO planner. Create a plan to achieve the goal using available tools."
            elif strategy == "tot":
                system_prompt = "You are a deliberate thinker. Generate distinct, creative next steps or reasoning thoughts to achieve the goal."

        if strategy == "plan-and-execute":
            return self.create_plan_execute_agent(name, system_prompt, tools, max_iterations=max_iterations, **kwargs)
        elif strategy == "rewoo":
            return self.create_rewoo_agent(name, system_prompt, tools, max_iterations=max_iterations, **kwargs)
        elif strategy == "tot":
            return self.create_tot_agent(name, system_prompt, max_iterations=max_iterations, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def create_tool(self, name: str, func: Callable, description: str):
        """Create and register custom tool."""
        from .tool import Tool
        tool = Tool(name, func, description)
        self.tools.register(name, tool)
        return tool
    
    def create_workflow(self, name: str):
        """Create workflow."""
        return self.pipeline.create(name, event_bus=self.event_bus)
    
    def create_conversation(self, 
                            agents: List["Agent"], 
                            max_turns: int = 10, 
                            min_turns: int = 3,
                            termination_condition: str = "consensus"):
        """
        Create a multi-agent conversation manager.
        """
        from .conversation import ConversationManager
        return ConversationManager(
            agents=agents,
            framework=self,
            max_turns=max_turns,
            min_turns=min_turns,
            termination_condition=termination_condition
        )

    async def process(self, tasks: List[Dict]) -> List[Dict]:
        """
        Run multiple agent tasks in parallel.
        
        Args:
            tasks: List of dicts, each containing:
                - agent: The agent instance or name
                - input: The user input
                - context: Optional context
                
        Returns:
            List of results.
        """
        import asyncio
        
        async_tasks = []
        for task in tasks:
            agent = task['agent']
            async_tasks.append(agent.run(task['input'], task.get('context')))
            
        return await asyncio.gather(*async_tasks)

    def process_sync(self, tasks: List[Dict]) -> List[Dict]:
        """
        Synchronous wrapper for process.
        Runs multiple agent tasks in parallel without requiring async/await in the caller.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.process(tasks))
        else:
            return asyncio.run(self.process(tasks))

    def get_metrics(self) -> Dict:
        return {
            'circuit_breaker': getattr(self.circuit_breaker, 'get_stats', lambda: {})(),
            'idempotency': getattr(self.idempotency, 'get_metrics', lambda: {})(),
            'vector_memory': getattr(self.vector_memory, 'get_metrics', lambda: {})(),
            'session_memory': getattr(self.session_memory, 'get_metrics', lambda: {})(),
        }

    def enable_verbose_monitoring(self):
        """Enable standardized console feedback for all events."""
        def default_on_event(event, data):
            agent = data.get('agent') or data.get('pipeline') or 'System'
            
            if "thought" in event:
                print(f"   [{agent}] Thinking: {data.get('reasoning', '')[:100]}...")
            elif "action" in event:
                print(f"   [{agent}] Action: {data.get('tool')}({data.get('args', {})})")
            elif "observation" in event:
                obs = str(data.get('observation', ''))
                print(f"   [{agent}] Observation: {obs[:100]}...")
            elif "complete" in event:
                print(f"   [{agent}] Task Completed.")
            elif "error" in event:
                print(f"   [{agent}] ERROR: {data.get('error')}")
            elif "pipeline:step" in event:
                print(f"   [Pipeline] Step '{data.get('step')}' finished.")
            elif "supervisor" in event:
                print(f"   [Supervisor] {data.get('message')}")

        self.event_bus.subscribe("*", default_on_event)
        self.logger.info("Verbose monitoring enabled via Console")


class MasterAgent:
    """High-level supervisor for goal-driven autonomous operations."""
    def __init__(self, name: str, framework):
        self.name = name
        self.framework = framework
        self.state = {"goal_reached": False, "count": 0, "results": []}

    async def run_until(self, goal_description: str, check_fn: Callable, max_iterations: int = 5):
        """
        Execute a goal-driven loop.
        The caller (orchestrator script) should yield the results back to update state.
        """
        self.framework.event_bus.emit("supervisor:goal_set", {
            "master": self.name,
            "goal": goal_description,
            "message": f"Global Mission Started: {goal_description}"
        })

        for i in range(max_iterations):
            self.framework.event_bus.emit("supervisor:iteration_start", {
                "master": self.name,
                "iteration": i + 1,
                "message": f"Planning Iteration {i+1} to reach goal..."
            })

            # The implementation script will handle the actual agent calls
            # and update supervisor.state["count"] and supervisor.state["results"]
            yield i + 1

            # Check if goal is met
            if check_fn(self.state):
                self.framework.event_bus.emit("supervisor:goal_reached", {
                    "master": self.name,
                    "count": self.state["count"],
                    "message": f"MISSION SUCCESS: Goal reached in {i+1} iterations!"
                })
                return

            self.framework.event_bus.emit("supervisor:goal_check", {
                "master": self.name,
                "current": self.state["count"],
                "status": "Goal incomplete. Continuing...",
                "message": f"Currently at {self.state['count']} items. Need more."
            })

        self.framework.event_bus.emit("supervisor:max_iterations", {
            "master": self.name,
            "error": "Failed to meet goal within iteration limit."
        })

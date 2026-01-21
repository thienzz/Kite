"""
Kite Framework - Core
General-purpose framework for ANY agentic AI application.
"""

import os
import logging
from typing import Dict, Optional, Any, Callable, List
from .data_loaders import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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
        
        self.advanced_rag = None
        self.db_mcp = None
        self.data_loader = DocumentLoader()
        
        self._init_providers()
        self._init_safety()
        self._init_memory()
        self._init_routing()
        self._init_tools()
        self._init_slm()
        self._init_pipeline()
        self._init_advanced()
        
        self.logger.info("[OK] Kite (general-purpose) initialized")
    
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
        load_dotenv() # Load standard .env
        return {
            'llm_provider': os.getenv('LLM_PROVIDER', 'ollama'),
            'llm_model': os.getenv('LLM_MODEL'),
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'sentence-transformers'),
            'embedding_model': os.getenv('EMBEDDING_MODEL'),
            'slm_provider': os.getenv('SLM_PROVIDER', 'ollama'),
            'slm_sql_model': os.getenv('SLM_SQL_MODEL'),
            'slm_classifier_model': os.getenv('SLM_CLASSIFIER_MODEL'),
            'slm_code_review_model': os.getenv('SLM_CODE_REVIEW_MODEL'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'vector_backend': os.getenv('VECTOR_BACKEND', 'memory'), # Default to memory if not set
            'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', 3)),
            'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT_SECONDS', 60)),
            'llm_timeout': float(os.getenv('LLM_TIMEOUT', 180.0)),
        }
    
    def _init_providers(self):
        from .llm_providers import LLMFactory
        from .embedding_providers import EmbeddingFactory
        
        # Initialize LLM
        try:
            llm_provider = self.config.get('llm_provider')
            if llm_provider:
                api_key = self.config.get(f'{llm_provider}_api_key')
                self.llm = LLMFactory.create(
                    llm_provider,
                    self.config.get('llm_model'),
                    api_key=api_key,
                    timeout=self.config.get('llm_timeout', 180.0)
                )
            else:
                self.llm = LLMFactory.auto_detect()
        except Exception as e:
            self.logger.warning(f"    LLM Provider {self.config.get('llm_provider')}: {e}")
            self.llm = LLMFactory.auto_detect()
            
        # Initialize Embeddings
        try:
            emb_provider = self.config.get('embedding_provider')
            if emb_provider:
                self.embeddings = EmbeddingFactory.create(
                    emb_provider,
                    self.config.get('embedding_model')
                )
            else:
                self.embeddings = EmbeddingFactory.auto_detect()
        except Exception as e:
            self.logger.warning(f"    Embedding Provider {self.config.get('embedding_provider')}: {e}")
            self.embeddings = EmbeddingFactory.auto_detect()
            
        self.logger.info(f"  [OK] LLM: {self.llm.name if self.llm else 'None'}")
        self.logger.info(f"  [OK] Embeddings: {self.embeddings.name if self.embeddings else 'None'}")
    
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
    
    def _init_memory(self):
        """Initialize memory systems."""
        from .memory.vector_memory import VectorMemory
        from .memory.session_memory import SessionMemory
        from .memory.graph_rag import GraphRAG # Added this import to maintain GraphRAG functionality
        
        backend = self.config.get('vector_backend', 'chroma')
        self.vector_memory = VectorMemory(
            backend=backend,
            embedding_provider=self.embeddings
        )
        self.session_memory = SessionMemory()
        self.logger.info(f"  [OK] Memory (Backend: {backend})")
        self.graph_rag = GraphRAG(llm=self.llm)
        
        self.logger.info("  [OK] Memory")
    
    def _init_routing(self):
        """Initialize routing systems."""
        from .routing import SemanticRouter, AggregatorRouter
        
        self.semantic_router = SemanticRouter(
            confidence_threshold=0.4,
            embedding_provider=self.embeddings
        )
        self.aggregator_router = AggregatorRouter(llm=self.llm)
        
        self.logger.info("  [OK] Routing")
    
    def _init_tools(self):
        """Initialize tool registry."""
        from .tool_registry import ToolRegistry
        from .tools.contrib import web_search, calculator, get_current_datetime
        
        self.tools = ToolRegistry(self.config, self.logger)
        
        # Register standard contrib tools
        self.create_tool("web_search", web_search, "Search the web for information")
        self.create_tool("calculator", calculator, "Evaluate mathematical expressions")
        self.create_tool("get_datetime", get_current_datetime, "Get current date and time")
        
        self.logger.info("  [OK] Tools (including contrib)")
    
    def _init_slm(self):
        """Initialize SLM specialists."""
        from .slm_manager import SLMSpecialists
        self.slm = SLMSpecialists(self.config, self.logger)
        self.logger.info("  [OK] SLM")
    
    def _init_pipeline(self):
        """Initialize workflow system."""
        from .pipeline_manager import PipelineManager
        from .pipeline import DeterministicPipeline
        
        self.pipeline = PipelineManager(DeterministicPipeline, self.logger)
        self.logger.info("  [OK] Pipeline")

    def _init_advanced(self):
        """Initialize advanced features."""
        from .memory.advanced_rag import AdvancedRAG
        from .caching.cache_manager import CacheManager
        from .tools.mcp.database_mcp import DatabaseMCP
        from .utils.cluster import ClusterNode
        from .data_loaders import DocumentLoader # Added this import
        
        self.advanced_rag = AdvancedRAG(self.vector_memory, self.llm)
        self.cache = CacheManager(self.vector_memory)
        self.db_mcp = DatabaseMCP() # Changed self.mcp_db to self.db_mcp
        
        # Distributed Cluster Support
        if self.config.get('cluster_enabled', False):
            self.cluster = ClusterNode()
            self.cluster.join_cluster()
        
        self.logger.info("  [OK] Advanced Features (RAG, Cache, MCP, Cluster)")
    
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
                     slm_provider: str = None,
                     slm_model: str = None,
                     use_slm: bool = False):
        """
        Create custom agent with optional specific AI configuration.
        Overrides global .env settings for this specific agent.
        """
        from .agent import Agent
        from .llm_providers import LLMFactory
        from .slm_manager import SLMSpecialists
        
        # Determine LLM
        agent_llm = self.llm
        if llm_provider:
            try:
                agent_llm = LLMFactory.create(llm_provider, llm_model)
            except Exception as e:
                self.logger.warning(f"Failed to create specific LLM for {name}: {e}. Using default.")
        
        # Determine SLM
        agent_slm = self.slm
        if slm_provider:
            try:
                # Merge current config with overrides
                agent_config = self.config.copy() if hasattr(self.config, 'copy') else {}
                agent_config['slm_provider'] = slm_provider
                if slm_model:
                    agent_config['slm_model'] = slm_model
                agent_slm = SLMSpecialists(agent_config, self.logger)
            except Exception as e:
                self.logger.warning(f"Failed to create specific SLM for {name}: {e}. Using default.")
                
        return Agent(name, system_prompt, agent_llm, tools or [], self, agent_slm, use_slm=use_slm)
    
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
            slm=base_agent.slm,
            kill_switch=ks
        )
    
    def create_tool(self, name: str, func: Callable, description: str):
        """Create and register custom tool."""
        from .tool import Tool
        tool = Tool(name, func, description)
        if hasattr(self, 'tools'):
            self.tools.register(name, tool)
        return tool
    
    def create_workflow(self, name: str):
        """Create workflow."""
        return self.pipeline.create(name)
    
    async def process_parallel(self, tasks: List[Dict]) -> List[Dict]:
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
            if isinstance(agent, str):
                # We don't have a direct name -> agent map in Kite yet, 
                # usually agents are managed by the user or router.
                # For this implementation, we assume the agent object is passed.
                pass
            
            async_tasks.append(agent.run(task['input'], task.get('context')))
            
        return await asyncio.gather(*async_tasks)

    def get_metrics(self) -> Dict:
        return {
            'circuit_breaker': getattr(self.circuit_breaker, 'get_stats', lambda: {})(),
            'idempotency': getattr(self.idempotency, 'get_metrics', lambda: {})(),
            'vector_memory': getattr(self.vector_memory, 'get_metrics', lambda: {})(),
            'session_memory': getattr(self.session_memory, 'get_metrics', lambda: {})(),
        }

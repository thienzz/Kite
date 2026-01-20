"""
Comprehensive test suite for Kite Framework.
Coverage target: >80%
"""

import pytest
import time
from kite import Kite
from kite.safety import CircuitBreaker, IdempotencyManager
from kite.memory import VectorMemory, SessionMemory, GraphRAG
from kite.monitoring import MetricsCollector, Tracer
from kite.ab_testing import ABTestManager, Variant


class TestFrameworkInitialization:
    """Test framework initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        ai = Kite()
        assert ai is not None
        assert hasattr(ai, 'llm')
        assert hasattr(ai, 'embeddings')
    
    def test_config_init(self):
        """Test initialization with config."""
        config = {
            'llm_provider': 'ollama',
            'embedding_provider': 'sentence-transformers'
        }
        ai = Kite(config=config)
        assert ai.config['llm_provider'] == 'ollama'


class TestSafetyPatterns:
    """Test safety patterns."""
    
    def test_circuit_breaker(self):
        """Test circuit breaker."""
        from kite.safety import CircuitBreakerConfig
        cb = CircuitBreaker(
            name="test_cb",
            config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1)
        )
        
        # Should be closed initially
        from kite.safety import CircuitState
        assert cb.state == CircuitState.CLOSED
        
        # Record failures
        for _ in range(3):
            cb._on_failure()
        
        # Should be open now
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        time.sleep(1.1)
        # Call to trigger reset check
        try:
            cb.call(lambda: None)
        except:
            pass
        assert cb.state == CircuitState.HALF_OPEN or cb.state == CircuitState.CLOSED
    
    def test_idempotency(self):
        """Test idempotency."""
        from kite.safety import IdempotencyConfig
        idempotency = IdempotencyManager(
            config=IdempotencyConfig(storage_backend='memory')
        )
        
        call_count = [0]
        
        def expensive_operation():
            call_count[0] += 1
            return "result"
        
        # First call
        key1 = idempotency.generate_id("op", {"id": "1"})
        if not idempotency.is_duplicate(key1):
            res = expensive_operation()
            idempotency.store_result(key1, res)
        
        result1 = idempotency.get_result(key1)
        assert result1 == "result"
        assert call_count[0] == 1
        
        # Second call (should be cached)
        if idempotency.is_duplicate(key1):
            result2 = idempotency.get_result(key1)
        else:
            result2 = expensive_operation()
            
        assert result2 == "result"
        assert call_count[0] == 1  # Not called again


class TestMemorySystems:
    """Test memory systems."""
    
    def test_session_memory(self):
        """Test session memory."""
        session = SessionMemory(window_size=5)
        
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi")
        
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0]['content'] == "Hello"
    
    def test_graph_rag(self):
        """Test graph RAG."""
        graph = GraphRAG()
        
        from kite.memory.graph_rag import Entity, Relationship
        
        # Add entities first
        graph.graph.add_entity(Entity("user", "person", "User"))
        graph.graph.add_entity(Entity("product", "product", "Product"))
        graph.graph.add_entity(Entity("company", "company", "Company"))
        
        graph.graph.add_relationship(Relationship("user", "product", "bought"))
        graph.graph.add_relationship(Relationship("product", "company", "manufactured_by"))
        
        neighbors = graph.graph.get_neighbors("user")
        assert any(e.id == "product" for e in neighbors)


class TestMonitoring:
    """Test monitoring system."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        metrics = MetricsCollector(enable_prometheus=False)
        
        metrics.record_request("llm", "chat", 0.5, success=True)
        metrics.record_request("llm", "chat", 0.3, success=True)
        
        data = metrics.get_metrics()
        assert 'llm.chat' in data
        assert data['llm.chat']['count'] == 2
    
    def test_tracer(self):
        """Test distributed tracing."""
        tracer = Tracer()
        
        trace = tracer.start_trace("trace1", "operation")
        span = tracer.add_span("span1")
        time.sleep(0.1)
        tracer.end_span(span)
        tracer.end_trace()
        
        traces = tracer.get_traces()
        assert len(traces) == 1
        assert traces[0]['trace_id'] == "trace1"


class TestABTesting:
    """Test A/B testing framework."""
    
    def test_variant_assignment(self):
        """Test consistent variant assignment."""
        manager = ABTestManager()
        
        exp = manager.create_experiment(
            name="test1",
            description="Test experiment",
            variants=[
                Variant("control", 0.5, {"param": "A"}),
                Variant("treatment", 0.5, {"param": "B"})
            ]
        )
        
        # Same user should get same variant
        variant1 = manager.get_variant("test1", "user123")
        variant2 = manager.get_variant("test1", "user123")
        assert variant1.name == variant2.name
    
    def test_metrics_recording(self):
        """Test metrics recording."""
        manager = ABTestManager()
        
        exp = manager.create_experiment(
            name="test2",
            description="Test",
            variants=[
                Variant("v1", 1.0, {})
            ]
        )
        
        manager.record_impression("test2", "v1")
        manager.record_conversion("test2", "v1")
        
        results = manager.get_results("test2")
        assert results['variants'][0]['impressions'] == 1
        assert results['variants'][0]['conversions'] == 1


class TestAgentSystem:
    """Test agent creation and execution."""
    
    def test_create_agent(self):
        """Test agent creation."""
        ai = Kite()
        
        agent = ai.create_agent(
            name="TestAgent",
            system_prompt="You are a test agent",
            tools=[]
        )
        
        assert agent.name == "TestAgent"
        assert agent.system_prompt == "You are a test agent"
    
    def test_create_tool(self):
        """Test tool creation."""
        ai = Kite()
        
        def test_func(x):
            return x * 2
        
        tool = ai.create_tool(
            name="test_tool",
            func=test_func,
            description="Test tool"
        )
        
        assert tool.name == "test_tool"
        assert tool.execute(5) == 10


class TestWorkflows:
    """Test workflow system."""
    
    def test_create_workflow(self):
        """Test workflow creation."""
        ai = Kite()
        
        workflow = ai.create_workflow("test_workflow")
        assert workflow is not None


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_flow(self):
        """Test complete flow."""
        ai = Kite()
        
        # Create agent with tool
        def search(query):
            return f"Results for: {query}"
        
        tool = ai.create_tool("search", search, "Search tool")
        agent = ai.create_agent("SearchAgent", "You search", [tool])
        
        # Record in session memory
        ai.session_memory.add_message("s1", "user", "Search for AI")
        
        # Get metrics
        metrics = ai.get_metrics()
        assert metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=kite"])

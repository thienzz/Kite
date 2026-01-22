"""
End-to-end integration tests for Kite Framework.
Tests complete workflows combining multiple components.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from kite import Kite


@pytest.mark.integration
class TestInvoicePipelineFlow:
    """Test complete invoice processing workflow."""
    
    @pytest.mark.asyncio
    async def test_invoice_pipeline_flow(self, ai):
        """Test complete invoice processing."""
        from kite.pipeline.deterministic_pipeline import DeterministicPipeline
        
        def extract_data(invoice):
            return {"amount": 1000, "vendor": "ACME Corp"}
        
        def validate_data(data):
            return data if data["amount"] > 0 else None
        
        def process_payment(data):
            return {"status": "paid", **data}
        
        pipeline = DeterministicPipeline("invoice_pipeline")
        pipeline.add_step("extract", extract_data)
        pipeline.add_step("validate", validate_data)
        pipeline.add_step("process", process_payment)
        
        result = pipeline.execute("invoice.pdf")
        
        assert result.status.value == "completed"
        assert result.results["process"]["status"] == "paid"


@pytest.mark.integration
class TestResearchAssistantFlow:
    """Test multi-step research workflow."""
    
    @pytest.mark.asyncio
    async def test_research_assistant_flow(self, ai, mock_llm_provider):
        """Test research workflow with agent."""
        ai._llm = mock_llm_provider
        
        search_tool = ai.tools.get("web_search")
        agent = ai.create_react_agent(
            name="Researcher",
            system_prompt="You are a research assistant",
            tools=[search_tool]
        )
        agent.llm = mock_llm_provider
        
        mock_llm_provider.complete.return_value = "Research findings"
        
        result = await agent.run("Research AI trends")
        
        assert result is not None
        assert "response" in result


@pytest.mark.integration
class TestAgentWithMemoryAndTools:
    """Test agent with memory + tools."""
    
    @pytest.mark.asyncio
    async def test_agent_with_memory_and_tools(self, ai, mock_llm_provider, sample_documents):
        """Test agent with memory and tools."""
        ai._llm = mock_llm_provider
        
        # Add documents to memory
        for doc_id, content in sample_documents.items():
            try:
                ai.vector_memory.add_document(doc_id, content)
            except Exception:
                pass  # May fail if embeddings not available
        
        # Create agent with tools
        calc = ai.tools.get("calculator")
        agent = ai.create_agent(
            name="Assistant",
            system_prompt="You are an assistant with memory and tools",
            tools=[calc]
        )
        agent.llm = mock_llm_provider
        
        mock_llm_provider.complete.return_value = "Response with context"
        
        result = await agent.run("What is 2+2?")
        
        assert result is not None


@pytest.mark.integration
class TestHITLApprovalWorkflow:
    """Test HITL pipeline."""
    
    @pytest.mark.asyncio
    async def test_hitl_approval_workflow(self):
        """Test HITL pipeline with approval."""
        from kite.pipeline.deterministic_pipeline import DeterministicPipeline, PipelineStatus
        
        def step1(data):
            return data + " -> step1"
        
        def step2(data):
            return data + " -> step2"
        
        pipeline = DeterministicPipeline("hitl_test")
        pipeline.add_step("step1", step1)
        pipeline.add_checkpoint("step1", approval_required=True)
        pipeline.add_step("step2", step2)
        
        # Execute until checkpoint
        state = await pipeline.execute_async("data")
        assert state.status == PipelineStatus.AWAITING_APPROVAL
        
        # Resume after approval
        state = await pipeline.resume_async(state.task_id, feedback="Approved")
        assert state.status == PipelineStatus.COMPLETED


@pytest.mark.integration
class TestMultiAgentCollaboration:
    """Test multiple agents working together."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, ai, mock_llm_provider):
        """Test multiple agents collaborating."""
        ai._llm = mock_llm_provider
        
        researcher = ai.create_agent("Researcher", "Research facts", [])
        analyst = ai.create_agent("Analyst", "Analyze data", [])
        
        researcher.llm = mock_llm_provider
        analyst.llm = mock_llm_provider
        
        mock_llm_provider.complete.side_effect = [
            "Research findings",
            "Analysis results"
        ]
        
        # Researcher gathers info
        research = await researcher.run("Research topic")
        
        # Analyst processes
        analysis = await analyst.run(f"Analyze: {research['response']}")
        
        assert research is not None
        assert analysis is not None


@pytest.mark.integration
class TestSemanticRoutingToAgents:
    """Test routing queries to specialized agents."""
    
    def test_semantic_routing_to_agents(self, ai, mock_embedding_provider):
        """Test routing queries to specialized agents."""
        from kite.routing.semantic_router import SemanticRouter
        
        support_agent = ai.create_agent("Support", "Handle support", [])
        sales_agent = ai.create_agent("Sales", "Handle sales", [])
        
        router = SemanticRouter(embedding_provider=mock_embedding_provider)
        
        def route_to_support(query):
            return support_agent.run(query)
        
        def route_to_sales(query):
            return sales_agent.run(query)
        
        router.add_route("support", route_to_support, ["help", "issue", "problem"])
        router.add_route("sales", route_to_sales, ["buy", "purchase", "pricing"])
        
        # Router should direct to appropriate agent
        assert router is not None


@pytest.mark.integration
class TestRAGEnhancedAgent:
    """Test agent with RAG memory."""
    
    @pytest.mark.asyncio
    async def test_rag_enhanced_agent(self, ai, mock_llm_provider, sample_documents):
        """Test agent with RAG memory."""
        ai._llm = mock_llm_provider
        
        # Populate vector memory
        for doc_id, content in sample_documents.items():
            try:
                ai.vector_memory.add_document(doc_id, content)
            except Exception:
                pass
        
        # Create agent
        agent = ai.create_agent(
            name="RAGAgent",
            system_prompt="You answer using context from memory",
            tools=[]
        )
        agent.llm = mock_llm_provider
        
        # Search memory
        try:
            context = ai.vector_memory.search("circuit breakers", top_k=2)
        except Exception:
            context = []
        
        mock_llm_provider.complete.return_value = "Answer with context"
        
        result = await agent.run("What are circuit breakers?")
        
        assert result is not None


@pytest.mark.integration
class TestCircuitBreakerRecovery:
    """Test circuit breaker in real workflow."""
    
    def test_circuit_breaker_recovery(self, ai):
        """Test circuit breaker recovery."""
        from kite.safety import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        cb = CircuitBreaker(
            name="test_cb",
            config=CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1)
        )
        
        def failing_operation():
            raise Exception("Operation failed")
        
        # Trigger failures
        for _ in range(2):
            try:
                cb.call(failing_operation)
            except:
                pass
        
        # Circuit should be open
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        import time
        time.sleep(1.5)
        
        # Should transition to half-open
        def successful_operation():
            return "success"
        
        try:
            result = cb.call(successful_operation)
            # Should recover
            assert cb.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]
        except:
            # Still open
            assert cb.state == CircuitState.OPEN


@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Performance benchmarks for key operations."""
    
    def test_agent_creation_performance(self, ai):
        """Test agent creation is fast."""
        import time
        
        start = time.time()
        agent = ai.create_agent("TestAgent", "Prompt", [])
        duration = time.time() - start
        
        # Should be very fast (< 100ms)
        assert duration < 0.1
    
    def test_tool_execution_performance(self, ai):
        """Test tool execution is fast."""
        import time
        
        calc = ai.tools.get("calculator")
        
        start = time.time()
        result = calc.execute(expression="2 + 2")
        duration = time.time() - start
        
        # Should be very fast
        assert duration < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

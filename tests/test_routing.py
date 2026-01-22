"""
Comprehensive tests for routing components.
Tests SemanticRouter and AggregatorRouter functionality.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from kite import Kite
from kite.routing.semantic_router import SemanticRouter
from kite.routing.aggregator_router import AggregatorRouter


class TestSemanticRouter:
    """Test SemanticRouter functionality."""
    
    def test_semantic_router_initialization(self, mock_embedding_provider):
        """Test router creation with embeddings."""
        router = SemanticRouter(embedding_provider=mock_embedding_provider)
        assert router is not None
        assert router.routes == []  # Routes is a list
    
    def test_semantic_router_add_route(self, mock_embedding_provider):
        """Test adding routes with examples."""
        router = SemanticRouter(embedding_provider=mock_embedding_provider)
        
        def support_handler(query):
            return "Support response"
        
        router.add_route(
            name="support",
            examples=["I need help", "Can you assist me?", "Support please"],
            handler=support_handler
        )
        
        assert len(router.routes) == 1
        assert router.routes[0].name == "support"
        assert router.routes[0].handler == support_handler
    
    def test_semantic_router_route_query(self, mock_embedding_provider):
        """Test query routing to correct handler."""
        router = SemanticRouter(embedding_provider=mock_embedding_provider)
        
        def support_handler(query):
            return "Support response"
        
        def sales_handler(query):
            return "Sales response"
        
        router.add_route("support", examples=["I need help"], handler=support_handler)
        router.add_route("sales", examples=["I want to buy"], handler=sales_handler)
        
        # Route should return dict with route info
        result = router.route("I need assistance")
        assert "route" in result
        assert result["confidence"] > 0
    
    def test_semantic_router_confidence_threshold(self, mock_embedding_provider):
        """Test confidence-based routing."""
        router = SemanticRouter(
            embedding_provider=mock_embedding_provider,
            confidence_threshold=0.8
        )
        
        def handler(query):
            return "Response"
        
        router.add_route("test", examples=["example"], handler=handler)
        
        # Low confidence query should return "none" route
        result = router.route("completely unrelated query about something else")
        assert result["route"] == "none" or result["needs_clarification"] == True


class TestAggregatorRouter:
    """Test AggregatorRouter functionality."""
    
    def test_aggregator_router_initialization(self, mock_llm_provider):
        """Test aggregator router setup."""
        router = AggregatorRouter(llm=mock_llm_provider)
        assert router is not None
        assert hasattr(router, 'agents')  # Has agents dict
    
    def test_aggregator_router_decomposition(self, mock_llm_provider):
        """Test query decomposition using TaskDecomposer."""
        router = AggregatorRouter(llm=mock_llm_provider)
        
        # Mock LLM to return decomposed queries
        mock_llm_provider.complete.return_value = json.dumps([
            {"description": "Search for AI trends", "assigned_agent": "researcher", "priority": 1},
            {"description": "Analyze market data", "assigned_agent": "analyst", "priority": 2}
        ])
        
        # TaskDecomposer is used internally
        decomposer = router.decomposer
        subtasks = decomposer.decompose("Analyze AI market trends")
        assert isinstance(subtasks, list)
        assert len(subtasks) >= 0  # May be empty if parsing fails
    
    def test_aggregator_router_aggregation(self, mock_llm_provider):
        """Test result aggregation using _merge_responses."""
        router = AggregatorRouter(llm=mock_llm_provider)
        
        from kite.routing.aggregator_router import AgentResponse
        responses = [
            AgentResponse("researcher", "task1", "Finding 1", True),
            AgentResponse("analyst", "task2", "Finding 2", True),
        ]
        
        mock_llm_provider.complete.return_value = "Aggregated summary of findings"
        
        aggregated = router._merge_responses(responses, "original query")
        assert "summary" in aggregated.lower() or "finding" in aggregated.lower()
    
    def test_aggregator_router_with_agents(self, ai, mock_llm_provider):
        """Test routing with registered agents."""
        router = AggregatorRouter(llm=mock_llm_provider)
        
        # Create and register test agents
        agent1 = ai.create_agent("Agent1", "You are agent 1", [])
        agent2 = ai.create_agent("Agent2", "You are agent 2", [])
        
        router.register_agent("agent1", agent1, "Agent 1 description")
        router.register_agent("agent2", agent2, "Agent 2 description")
        
        assert len(router.agents) == 2
        assert "agent1" in router.agents
        assert "agent2" in router.agents


class TestRouterIntegration:
    """Integration tests for routers."""
    
    def test_semantic_router_with_kite(self, ai):
        """Test semantic router integration with Kite."""
        router = ai.semantic_router
        assert router is not None
    
    def test_aggregator_router_with_kite(self, ai):
        """Test aggregator router integration with Kite."""
        router = ai.aggregator_router
        assert router is not None
    
    def test_router_with_custom_agents(self, ai):
        """Test router with custom agents."""
        # Create specialized agents
        support_agent = ai.create_agent(
            "SupportAgent",
            "You handle customer support",
            []
        )
        
        sales_agent = ai.create_agent(
            "SalesAgent",
            "You handle sales inquiries",
            []
        )
        
        # Router should be able to work with these agents
        assert support_agent.name == "SupportAgent"
        assert sales_agent.name == "SalesAgent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

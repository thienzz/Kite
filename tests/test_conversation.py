"""
Comprehensive tests for conversation system.
Tests multi-agent conversations and termination conditions.
"""

import pytest
import asyncio
from kite import Kite


class TestConversationInitialization:
    """Test conversation initialization."""
    
    def test_conversation_creation(self, ai):
        """Test conversation setup."""
        agent1 = ai.create_agent("Agent1", "You are agent 1", [])
        agent2 = ai.create_agent("Agent2", "You are agent 2", [])
        
        conv = ai.create_conversation(
            agents=[agent1, agent2],
            max_turns=5
        )
        
        assert conv is not None
        assert len(conv.agents) == 2
        assert conv.max_turns == 5
    
    def test_conversation_with_single_agent(self, ai):
        """Test conversation with single agent (should work)."""
        agent = ai.create_agent("SoloAgent", "You talk to yourself", [])
        
        conv = ai.create_conversation(agents=[agent], max_turns=3)
        assert conv is not None
        assert len(conv.agents) == 1


class TestConversationExecution:
    """Test conversation execution."""
    
    @pytest.mark.asyncio
    async def test_two_agent_conversation(self, ai, mock_llm_provider):
        """Test basic two-agent dialogue."""
        # Override LLM with mock
        ai._llm = mock_llm_provider
        
        agent1 = ai.create_agent("Alice", "You are Alice", [])
        agent2 = ai.create_agent("Bob", "You are Bob", [])
        
        agent1.llm = mock_llm_provider
        agent2.llm = mock_llm_provider
        
        conv = ai.create_conversation(
            agents=[agent1, agent2],
            max_turns=4
        )
        
        # Mock responses
        responses = ["Hello Bob", "Hi Alice", "How are you?", "I'm good"]
        mock_llm_provider.complete.side_effect = responses
        
        result = await conv.run("Start a conversation")
        
        assert result is not None
        assert "turns" in result
        assert result["turns"] <= 4
    
    @pytest.mark.asyncio
    async def test_conversation_termination_max_turns(self, ai, mock_llm_provider):
        """Test max turns termination."""
        ai._llm = mock_llm_provider
        
        agent1 = ai.create_agent("Agent1", "Keep talking", [])
        agent2 = ai.create_agent("Agent2", "Keep talking", [])
        
        agent1.llm = mock_llm_provider
        agent2.llm = mock_llm_provider
        
        conv = ai.create_conversation(
            agents=[agent1, agent2],
            max_turns=3,
            termination_condition="max_turns"
        )
        
        mock_llm_provider.complete.return_value = "Continuing conversation"
        
        result = await conv.run("Talk")
        
        assert result["turns"] == 3
        assert "termination" in result
    
    @pytest.mark.asyncio
    async def test_conversation_termination_consensus(self, ai, mock_llm_provider):
        """Test consensus detection."""
        ai._llm = mock_llm_provider
        
        agent1 = ai.create_agent(
            "Researcher",
            "Provide facts. If Critic agrees, conclude with 'Final Conclusion'",
            []
        )
        agent2 = ai.create_agent(
            "Critic",
            "Challenge points. If satisfied, say 'I agree, consensus reached'",
            []
        )
        
        agent1.llm = mock_llm_provider
        agent2.llm = mock_llm_provider
        
        conv = ai.create_conversation(
            agents=[agent1, agent2],
            max_turns=6,
            termination_condition="consensus"
        )
        
        # Simulate reaching consensus
        responses = [
            "Here are the facts",
            "I have concerns",
            "Addressing your concerns",
            "I agree, consensus reached"
        ]
        mock_llm_provider.complete.side_effect = responses
        
        result = await conv.run("Discuss AI")
        
        assert result is not None
        assert result["turns"] <= 6


class TestConversationHistory:
    """Test conversation history tracking."""
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, ai, mock_llm_provider):
        """Test history tracking."""
        ai._llm = mock_llm_provider
        
        agent1 = ai.create_agent("A1", "Agent 1", [])
        agent2 = ai.create_agent("A2", "Agent 2", [])
        
        agent1.llm = mock_llm_provider
        agent2.llm = mock_llm_provider
        
        conv = ai.create_conversation(agents=[agent1, agent2], max_turns=3)
        
        mock_llm_provider.complete.side_effect = ["Msg1", "Msg2", "Msg3"]
        
        result = await conv.run("Start")
        
        assert "history" in result
        assert len(result["history"]) == result["turns"]
        
        # Each history entry should have agent and content
        for entry in result["history"]:
            assert "agent" in entry
            assert "content" in entry


class TestConversationWithTools:
    """Test agents with tools in conversation."""
    
    @pytest.mark.asyncio
    async def test_conversation_with_tools(self, ai, mock_llm_provider):
        """Test agents with tools in conversation."""
        ai._llm = mock_llm_provider
        
        calc = ai.tools.get("calculator")
        
        agent1 = ai.create_agent("MathAgent", "You do math", [calc])
        agent2 = ai.create_agent("CheckerAgent", "You verify", [])
        
        agent1.llm = mock_llm_provider
        agent2.llm = mock_llm_provider
        
        conv = ai.create_conversation(agents=[agent1, agent2], max_turns=4)
        
        mock_llm_provider.complete.return_value = "Response"
        
        result = await conv.run("Calculate 2+2 and verify")
        
        assert result is not None
        assert result["success"] == True


class TestConversationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_conversation_no_agents(self, ai):
        """Test conversation with no agents."""
        with pytest.raises(Exception):
            conv = ai.create_conversation(agents=[], max_turns=5)
    
    def test_conversation_invalid_max_turns(self, ai):
        """Test conversation with invalid max_turns."""
        agent = ai.create_agent("Agent", "Prompt", [])
        
        # Should handle gracefully or raise
        try:
            conv = ai.create_conversation(agents=[agent], max_turns=0)
            assert conv.max_turns > 0  # Should set to default
        except ValueError:
            pass  # Expected
    
    @pytest.mark.asyncio
    async def test_conversation_agent_error(self, ai, mock_llm_provider):
        """Test handling agent errors during conversation."""
        ai._llm = mock_llm_provider
        
        agent1 = ai.create_agent("A1", "Agent 1", [])
        agent2 = ai.create_agent("A2", "Agent 2", [])
        
        agent1.llm = mock_llm_provider
        agent2.llm = mock_llm_provider
        
        conv = ai.create_conversation(agents=[agent1, agent2], max_turns=3)
        
        # Mock one agent to fail
        mock_llm_provider.complete.side_effect = [
            "Response 1",
            Exception("Agent error"),
            "Recovery response"
        ]
        
        # Should handle error gracefully
        try:
            result = await conv.run("Start")
            # Either completes with error handling or raises
            assert result is not None
        except Exception as e:
            # Expected behavior
            assert "error" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Comprehensive tests for tool system.
Tests tool registry, built-in tools, and tool execution.
"""

import pytest
import logging
from unittest.mock import MagicMock, patch
from kite import Kite
from kite.tool import Tool
from kite.tool_registry import ToolRegistry


class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_tool_registry_initialization(self):
        """Test tool registry setup."""
        registry = ToolRegistry(config={}, logger=logging.getLogger())
        assert registry is not None
        assert hasattr(registry, '_tools')
    
    def test_tool_registration(self):
        """Test registering custom tools."""
        registry = ToolRegistry(config={}, logger=logging.getLogger())
        
        def custom_tool(x: int):
            return x * 2
        
        tool = Tool(
            name="double",
            func=custom_tool,
            description="Doubles a number"
        )
        
        registry.register("double", tool)
        assert registry.get("double") is not None
        assert registry.get("double").name == "double"
    
    def test_tool_get_nonexistent(self):
        """Test getting non-existent tool."""
        registry = ToolRegistry(config={}, logger=logging.getLogger())
        result = registry.get("nonexistent")
        assert result is None


class TestToolExecution:
    """Test tool execution."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        def add(a: int, b: int):
            return a + b
        
        tool = Tool(
            name="add",
            func=add,
            description="Adds two numbers"
        )
        
        assert tool.name == "add"
        assert tool.description == "Adds two numbers"
    
    def test_tool_execution(self):
        """Test tool execution with various inputs."""
        def multiply(a: int, b: int):
            return a * b
        
        tool = Tool(name="multiply", func=multiply, description="Multiply")
        result = tool.execute(a=3, b=4)
        assert result == 12
    
    def test_tool_error_handling(self):
        """Test tool error scenarios."""
        def failing_tool():
            raise ValueError("Tool error")
        
        tool = Tool(name="fail", func=failing_tool, description="Fails")
        
        with pytest.raises(ValueError):
            tool.execute()


class TestBuiltInTools:
    """Test built-in contrib tools."""
    
    def test_calculator_tool(self, ai):
        """Test calculator with expressions."""
        calc = ai.tools.get("calculator")
        assert calc is not None
        
        result = calc.execute(expression="2 + 3 * 4")
        assert result["result"] == 14
    
    def test_calculator_complex_expression(self, ai):
        """Test calculator with complex expressions."""
        calc = ai.tools.get("calculator")
        
        result = calc.execute(expression="(10 + 5) * 2 - 8")
        assert result["result"] == 22
    
    def test_calculator_error_handling(self, ai):
        """Test calculator with invalid expression."""
        calc = ai.tools.get("calculator")
        
        result = calc.execute(expression="invalid expression")
        assert "error" in result or "result" not in result
    
    @pytest.mark.requires_api
    def test_web_search_tool(self, ai):
        """Test web search (mocked)."""
        search = ai.tools.get("web_search")
        assert search is not None
        
        # Mock the search function
        with patch.object(search, 'execute', return_value={"results": ["Result 1", "Result 2"]}):
            result = search.execute(query="AI trends 2024")
            assert "results" in result
            assert len(result["results"]) > 0
    
    def test_datetime_tool(self, ai):
        """Test datetime utilities."""
        datetime_tool = ai.tools.get("get_datetime")
        assert datetime_tool is not None
        
        result = datetime_tool.execute()
        assert "datetime" in result or "time" in result or isinstance(result, str)


class TestToolWithAgent:
    """Test tool integration with agents."""
    
    def test_agent_with_single_tool(self, ai, sample_tools):
        """Test agent with one tool."""
        calc_tool = ai.tools.get("calculator")
        
        agent = ai.create_agent(
            name="MathAgent",
            system_prompt="You are a math assistant",
            tools=[calc_tool]
        )
        
        assert agent.name == "MathAgent"
        assert len(agent.tools) == 1
        assert "calculator" in agent.tools
    
    def test_agent_with_multiple_tools(self, ai):
        """Test agent with multiple tools."""
        calc = ai.tools.get("calculator")
        datetime = ai.tools.get("get_datetime")
        
        agent = ai.create_agent(
            name="MultiToolAgent",
            system_prompt="You have multiple tools",
            tools=[calc, datetime]
        )
        
        assert len(agent.tools) >= 2
    
    def test_custom_tool_creation(self, ai):
        """Test creating and using custom tool."""
        def reverse_string(text: str):
            return text[::-1]
        
        tool = ai.create_tool(
            name="reverse",
            func=reverse_string,
            description="Reverses a string"
        )
        
        assert tool.name == "reverse"
        result = tool.execute(text="hello")
        assert result == "olleh"


class TestMCPTools:
    """Test MCP tool integrations."""
    
    def test_database_mcp_initialization(self):
        """Test DatabaseMCP initialization."""
        from kite.tools.mcp.database_mcp import DatabaseMCP
        
        mcp = DatabaseMCP()
        assert mcp is not None
    
    @pytest.mark.asyncio
    async def test_database_mcp_safe_execute(self):
        """Test safe execution decorator."""
        from kite.tools.mcp.database_mcp import DatabaseMCP
        
        mcp = DatabaseMCP()
        # Should return error dict instead of raising
        result = await mcp.query_neo4j("MATCH (n) RETURN n")
        assert "error" in result


class TestToolRegistryIntegration:
    """Integration tests for tool registry."""
    
    def test_contrib_tools_registry(self, ai):
        """Test that contrib tools are registered."""
        assert ai.tools.get("web_search") is not None
        assert ai.tools.get("calculator") is not None
        assert ai.tools.get("get_datetime") is not None
    
    def test_tool_listing(self, ai):
        """Test listing all available tools."""
        tools = ai.tools
        assert tools is not None
        # Should have at least the contrib tools
        assert tools.get("calculator") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

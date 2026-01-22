"""Pytest configuration and shared fixtures."""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def ai():
    """Fixture for Kite instance."""
    from kite import Kite
    return Kite()


@pytest.fixture
def mock_llm_provider():
    """Fixture for mocked LLM provider."""
    mock = MagicMock()
    mock.complete.return_value = "Mocked LLM response"
    mock.chat.return_value = {"content": "Mocked chat response"}
    
    async def async_stream():
        yield "Mocked "
        yield "streaming "
        yield "response"
    
    mock.stream_complete = AsyncMock(return_value=async_stream())
    return mock


@pytest.fixture
def mock_embedding_provider():
    """Fixture for mocked embedding provider."""
    mock = MagicMock()
    mock.embed.return_value = [0.1] * 384
    mock.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]
    mock.dimension = 384
    return mock


@pytest.fixture
def temp_vector_db(tmp_path):
    """Fixture for temporary vector database."""
    db_path = tmp_path / "vector_db"
    db_path.mkdir()
    yield str(db_path)
    # Cleanup handled by tmp_path


@pytest.fixture
def sample_documents():
    """Fixture with sample test documents."""
    return {
        "doc1": "Kite is a production-ready agentic AI framework.",
        "doc2": "It supports multiple LLM providers including OpenAI and Anthropic.",
        "doc3": "The framework includes safety patterns like circuit breakers.",
        "doc4": "Vector memory enables semantic search capabilities.",
        "doc5": "Graph RAG provides relationship-aware retrieval.",
    }


@pytest.fixture
def sample_tools():
    """Fixture with sample tools for testing."""
    def calculator(expression: str):
        """Simple calculator tool."""
        try:
            return {"result": eval(expression)}
        except Exception as e:
            return {"error": str(e)}
    
    def search(query: str):
        """Mock search tool."""
        return {"results": [f"Result for: {query}"]}
    
    return {
        "calculator": calculator,
        "search": search,
    }


@pytest.fixture
def cleanup_resources():
    """Fixture for resource cleanup after tests."""
    resources = []
    
    def register(resource):
        resources.append(resource)
    
    yield register
    
    # Cleanup
    for resource in resources:
        if hasattr(resource, 'close'):
            resource.close()
        elif hasattr(resource, 'cleanup'):
            resource.cleanup()

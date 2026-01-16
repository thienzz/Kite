"""Pytest configuration."""
import pytest

@pytest.fixture
def ai():
    """Fixture for AgenticAI instance."""
    from agentic_framework import AgenticAI
    return AgenticAI()

"""Pytest configuration."""
import pytest

@pytest.fixture
def ai():
    """Fixture for Kite instance."""
    from kite import Kite
    return Kite()

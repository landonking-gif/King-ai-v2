# Pytest configuration and fixtures
import pytest

@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    pass

@pytest.fixture
def test_db():
    """In-memory SQLite database for testing."""
    pass

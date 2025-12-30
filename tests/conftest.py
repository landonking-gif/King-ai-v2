# Pytest configuration and fixtures
import pytest
import os

# Set up test environment variables
os.environ['DATABASE_URL'] = 'postgresql+asyncpg://test:test@localhost:5432/test'
os.environ['OLLAMA_URL'] = 'http://localhost:11434'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    pass

@pytest.fixture
def test_db():
    """In-memory SQLite database for testing."""
    pass

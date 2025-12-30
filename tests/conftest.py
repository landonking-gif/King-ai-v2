# Pytest configuration and fixtures
import pytest
import os

# Set test environment variables before any imports
os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///:memory:')
os.environ.setdefault('OLLAMA_URL', 'http://localhost:11434')
os.environ.setdefault('OLLAMA_MODEL', 'llama3.1:8b')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379')

@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    pass

@pytest.fixture
def test_db():
    """In-memory SQLite database for testing."""
    pass

"""
Memory Management Module for Kautilya.

Provides cognitive-inspired memory architecture with:
- Working Memory: Current session context
- Episodic Memory: Past interactions (Phase 2)
- Semantic Memory: Learned facts and preferences (Phase 3)
- Procedural Memory: Successful patterns (Phase 4)

Module: kautilya/memory/__init__.py
"""

from .models import (
    MemoryConfig,
    Message,
    Interaction,
    Session,
    WorkingMemoryState,
    UserProfile,
    MemoryContext,
)
from .storage import StorageBackend, RedisStorage, SQLiteStorage, get_storage
from .working import WorkingMemoryStore
from .session import SessionManager
from .manager import MemoryManager

__all__ = [
    # Models
    "MemoryConfig",
    "Message",
    "Interaction",
    "Session",
    "WorkingMemoryState",
    "UserProfile",
    "MemoryContext",
    # Storage
    "StorageBackend",
    "RedisStorage",
    "SQLiteStorage",
    "get_storage",
    # Stores
    "WorkingMemoryStore",
    # Session
    "SessionManager",
    # Manager
    "MemoryManager",
]

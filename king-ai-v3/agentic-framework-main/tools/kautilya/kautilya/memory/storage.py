"""
Storage Backends for Memory System.

Provides abstraction over different storage backends:
- Redis: Fast in-memory cache for working memory
- SQLite: Local persistent storage (default fallback)
- Memory: In-process storage for testing

Module: kautilya/memory/storage.py
"""

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        pass

    @abstractmethod
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value as JSON object."""
        pass

    @abstractmethod
    def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set JSON object value."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection."""
        pass


class RedisStorage(StorageBackend):
    """Redis storage backend for fast in-memory caching."""

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0):
        """
        Initialize Redis storage.

        Args:
            url: Redis connection URL
            db: Redis database number
        """
        self.url = url
        self.db = db
        self._client = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Redis."""
        try:
            import redis
            self._client = redis.from_url(self.url, db=self.db, decode_responses=True)
            # Test connection
            self._client.ping()
            self._available = True
            logger.info(f"Connected to Redis at {self.url}")
        except ImportError:
            logger.warning("Redis package not installed. Using fallback storage.")
            self._available = False
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Using fallback storage.")
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._available

    def get(self, key: str) -> Optional[str]:
        if not self._available:
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if not self._available:
            return False
        try:
            if ttl:
                return self._client.setex(key, ttl, value)
            return self._client.set(key, value)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            return self._client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    def exists(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            return self._client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False

    def keys(self, pattern: str) -> List[str]:
        if not self._available:
            return []
        try:
            return self._client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return []

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            return self.set(key, json.dumps(value), ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            return False

    def close(self) -> None:
        if self._client:
            self._client.close()


class SQLiteStorage(StorageBackend):
    """SQLite storage backend for persistent local storage."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.kautilya/memory.db
        """
        if db_path is None:
            config_dir = Path.home() / ".kautilya"
            config_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(config_dir / "memory.db")

        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized SQLite storage at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Key-value store for simple data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    title TEXT,
                    metadata TEXT,
                    message_count INTEGER DEFAULT 0,
                    tool_call_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Interactions table (for episodic memory - Phase 2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tools_used TEXT,
                    sources TEXT,
                    iterations INTEGER DEFAULT 0,
                    topic_tags TEXT,
                    entities TEXT,
                    importance_score REAL DEFAULT 0.5,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_store(expires_at)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        # Don't use detect_types to avoid timestamp parsing issues
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM kv_store WHERE expires_at IS NOT NULL AND expires_at < ?",
                (datetime.now(),)
            )
            conn.commit()

    def get(self, key: str) -> Optional[str]:
        self._cleanup_expired()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM kv_store WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                (key, datetime.now())
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO kv_store (key, value, expires_at, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value, expires_at))
            conn.commit()
            return True

    def delete(self, key: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM kv_store WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern (uses SQL LIKE)."""
        # Convert glob pattern to SQL LIKE
        sql_pattern = pattern.replace("*", "%").replace("?", "_")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key FROM kv_store WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?)",
                (sql_pattern, datetime.now())
            )
            return [row["key"] for row in cursor.fetchall()]

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            return self.set(key, json.dumps(value), ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            return False

    # === Session-specific methods ===

    def save_session(self, session_data: Dict[str, Any]) -> bool:
        """Save session to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, user_id, started_at, ended_at, is_active,
                    title, metadata, message_count, tool_call_count, total_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    ended_at = excluded.ended_at,
                    is_active = excluded.is_active,
                    title = excluded.title,
                    metadata = excluded.metadata,
                    message_count = excluded.message_count,
                    tool_call_count = excluded.tool_call_count,
                    total_tokens = excluded.total_tokens
            """, (
                session_data["session_id"],
                session_data.get("user_id", "default"),
                session_data.get("started_at"),
                session_data.get("ended_at"),
                1 if session_data.get("is_active", True) else 0,
                session_data.get("title"),
                json.dumps(session_data.get("metadata", {})),
                session_data.get("message_count", 0),
                session_data.get("tool_call_count", 0),
                session_data.get("total_tokens", 0),
            ))
            conn.commit()
            return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_recent_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions for user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions
                WHERE user_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_active_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get active session for user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions
                WHERE user_id = ? AND is_active = 1
                ORDER BY started_at DESC
                LIMIT 1
            """, (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    # === Message-specific methods ===

    def save_message(self, message_data: Dict[str, Any]) -> bool:
        """Save message to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (
                    message_id, session_id, role, content, timestamp,
                    tool_calls, tool_call_id, name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    content = excluded.content,
                    tool_calls = excluded.tool_calls
            """, (
                message_data["message_id"],
                message_data["session_id"],
                message_data["role"],
                message_data["content"],
                message_data.get("timestamp"),
                json.dumps(message_data.get("tool_calls")) if message_data.get("tool_calls") else None,
                message_data.get("tool_call_id"),
                message_data.get("name"),
            ))
            conn.commit()
            return True

    def get_session_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages for session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query, (session_id,))
            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                if msg.get("tool_calls"):
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                messages.append(msg)
            return messages

    # === Interaction-specific methods ===

    def save_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Save interaction to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (
                    interaction_id, session_id, user_query, agent_response,
                    timestamp, tools_used, sources, iterations,
                    topic_tags, entities, importance_score,
                    input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_data["interaction_id"],
                interaction_data["session_id"],
                interaction_data["user_query"],
                interaction_data["agent_response"],
                interaction_data.get("timestamp"),
                json.dumps(interaction_data.get("tools_used", [])),
                json.dumps(interaction_data.get("sources", [])),
                interaction_data.get("iterations", 0),
                json.dumps(interaction_data.get("topic_tags", [])),
                json.dumps(interaction_data.get("entities", [])),
                interaction_data.get("importance_score", 0.5),
                interaction_data.get("input_tokens", 0),
                interaction_data.get("output_tokens", 0),
            ))
            conn.commit()
            return True

    def get_recent_interactions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions for user across all sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.* FROM interactions i
                JOIN sessions s ON i.session_id = s.session_id
                WHERE s.user_id = ?
                ORDER BY i.timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            interactions = []
            for row in cursor.fetchall():
                interaction = dict(row)
                interaction["tools_used"] = json.loads(interaction.get("tools_used") or "[]")
                interaction["sources"] = json.loads(interaction.get("sources") or "[]")
                interaction["topic_tags"] = json.loads(interaction.get("topic_tags") or "[]")
                interaction["entities"] = json.loads(interaction.get("entities") or "[]")
                interactions.append(interaction)
            return interactions

    # === User profile methods ===

    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Save user profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_profiles (user_id, profile_data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_data = excluded.profile_data,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, json.dumps(profile_data)))
            conn.commit()
            return True

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["profile_data"])
            return None

    def close(self) -> None:
        """Close is a no-op for SQLite (connections are managed per-operation)."""
        pass


class InMemoryStorage(StorageBackend):
    """In-memory storage for testing."""

    def __init__(self):
        self._data: Dict[str, tuple] = {}  # key -> (value, expires_at)

    def _cleanup_expired(self) -> None:
        now = datetime.now()
        expired = [k for k, (_, exp) in self._data.items() if exp and exp < now]
        for k in expired:
            del self._data[k]

    def get(self, key: str) -> Optional[str]:
        self._cleanup_expired()
        if key in self._data:
            value, expires_at = self._data[key]
            if expires_at is None or expires_at > datetime.now():
                return value
        return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self._data[key] = (value, expires_at)
        return True

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, pattern: str) -> List[str]:
        import fnmatch
        self._cleanup_expired()
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            return self.set(key, json.dumps(value), ttl)
        except (TypeError, ValueError):
            return False

    def close(self) -> None:
        self._data.clear()


def get_storage(
    backend: str = "sqlite",
    redis_url: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> StorageBackend:
    """
    Get storage backend instance.

    Args:
        backend: Storage backend type ("redis", "sqlite", "memory")
        redis_url: Redis connection URL (for redis backend)
        sqlite_path: SQLite database path (for sqlite backend)

    Returns:
        StorageBackend instance
    """
    if backend == "redis":
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        storage = RedisStorage(url=redis_url)
        if storage.is_available:
            return storage
        # Fall back to SQLite if Redis unavailable
        logger.warning("Redis unavailable, falling back to SQLite")
        return SQLiteStorage(db_path=sqlite_path)

    elif backend == "sqlite":
        return SQLiteStorage(db_path=sqlite_path)

    elif backend == "memory":
        return InMemoryStorage()

    else:
        raise ValueError(f"Unknown storage backend: {backend}")

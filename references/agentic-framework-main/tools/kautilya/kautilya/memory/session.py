"""
Session Management for Kautilya Memory System.

Handles:
- Session creation and lifecycle
- Session persistence and resumption
- User identification
- Session metadata tracking

Module: kautilya/memory/session.py
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Session, Interaction, UserProfile
from .storage import StorageBackend, SQLiteStorage

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions with persistence.

    Features:
    - Session creation with unique IDs
    - Session persistence to storage
    - Session resumption
    - User identification
    - Session statistics tracking
    """

    # Storage key prefixes
    SESSION_KEY_PREFIX = "kautilya:session:"
    ACTIVE_SESSION_KEY = "kautilya:active_session:"
    USER_KEY_PREFIX = "kautilya:user:"

    def __init__(
        self,
        storage: StorageBackend,
        user_id: Optional[str] = None,
    ):
        """
        Initialize session manager.

        Args:
            storage: Storage backend instance
            user_id: Optional user identifier (auto-generated if not provided)
        """
        self.storage = storage
        self.user_id = user_id or self._get_or_create_user_id()
        self._current_session: Optional[Session] = None

    def _get_or_create_user_id(self) -> str:
        """
        Get existing user ID or create new one.

        User ID is stored in ~/.kautilya/user_id and persisted.
        """
        config_dir = Path.home() / ".kautilya"
        config_dir.mkdir(parents=True, exist_ok=True)
        user_id_file = config_dir / "user_id"

        if user_id_file.exists():
            return user_id_file.read_text().strip()

        # Generate new user ID based on machine identifier
        machine_id = self._get_machine_id()
        user_id = f"user_{hashlib.sha256(machine_id.encode()).hexdigest()[:12]}"

        user_id_file.write_text(user_id)
        logger.info(f"Created new user ID: {user_id}")

        return user_id

    def _get_machine_id(self) -> str:
        """Get a stable machine identifier."""
        # Try to get a stable machine ID
        try:
            # On macOS/Linux, use hostname + username
            import socket
            hostname = socket.gethostname()
            username = os.getenv("USER", os.getenv("USERNAME", "unknown"))
            return f"{hostname}:{username}"
        except Exception:
            # Fallback to random UUID (stored for consistency)
            return str(uuid.uuid4())

    def create_session(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new conversation session.

        Args:
            title: Optional session title
            metadata: Optional metadata dict

        Returns:
            New Session instance
        """
        # End any existing active session
        if self._current_session and self._current_session.is_active:
            self.end_session(self._current_session.session_id)

        session = Session(
            session_id=str(uuid.uuid4()),
            user_id=self.user_id,
            title=title,
            topic_tags=metadata.get("topic_tags", []) if metadata else [],
        )

        # Save to storage
        self._save_session(session)

        # Set as active session
        self._set_active_session(session.session_id)

        self._current_session = session
        logger.info(f"Created new session: {session.session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        key = f"{self.SESSION_KEY_PREFIX}{session_id}"
        data = self.storage.get_json(key)

        if data:
            return Session.from_dict(data)

        # Try SQLite-specific method if available
        if hasattr(self.storage, 'get_session'):
            data = self.storage.get_session(session_id)
            if data:
                return Session.from_dict(data)

        return None

    def get_current_session(self) -> Optional[Session]:
        """
        Get the current active session.

        Returns:
            Current Session or None
        """
        if self._current_session:
            return self._current_session

        # Try to get active session from storage
        active_key = f"{self.ACTIVE_SESSION_KEY}{self.user_id}"
        session_id = self.storage.get(active_key)

        if session_id:
            session = self.get_session(session_id)
            if session and session.is_active:
                self._current_session = session
                return session

        return None

    def get_or_create_session(self) -> Session:
        """
        Get current session or create new one.

        Returns:
            Active Session
        """
        session = self.get_current_session()
        if session:
            return session
        return self.create_session()

    def end_session(self, session_id: str) -> bool:
        """
        End a session.

        Args:
            session_id: Session identifier

        Returns:
            Success boolean
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.is_active = False
        session.ended_at = datetime.now()

        self._save_session(session)

        # Clear active session if this was it
        active_key = f"{self.ACTIVE_SESSION_KEY}{self.user_id}"
        current_active = self.storage.get(active_key)
        if current_active == session_id:
            self.storage.delete(active_key)

        if self._current_session and self._current_session.session_id == session_id:
            self._current_session = None

        logger.info(f"Ended session: {session_id}")
        return True

    def resume_session(self, session_id: str) -> Optional[Session]:
        """
        Resume a previous session.

        Args:
            session_id: Session identifier to resume

        Returns:
            Resumed Session or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Reactivate session
        session.is_active = True
        session.ended_at = None

        self._save_session(session)
        self._set_active_session(session_id)

        self._current_session = session
        logger.info(f"Resumed session: {session_id}")

        return session

    def get_recent_sessions(self, limit: int = 10) -> List[Session]:
        """
        Get recent sessions for current user.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of recent Sessions
        """
        # Try SQLite-specific method if available
        if hasattr(self.storage, 'get_recent_sessions'):
            sessions_data = self.storage.get_recent_sessions(self.user_id, limit)
            return [Session.from_dict(d) for d in sessions_data]

        # Fallback: scan keys
        pattern = f"{self.SESSION_KEY_PREFIX}*"
        keys = self.storage.keys(pattern)

        sessions = []
        for key in keys:
            data = self.storage.get_json(key)
            if data and data.get("user_id") == self.user_id:
                sessions.append(Session.from_dict(data))

        # Sort by start time, most recent first
        sessions.sort(key=lambda s: s.started_at, reverse=True)

        return sessions[:limit]

    def update_session_stats(
        self,
        session_id: str,
        message_count_delta: int = 0,
        tool_call_count_delta: int = 0,
        tokens_delta: int = 0,
    ) -> None:
        """
        Update session statistics.

        Args:
            session_id: Session identifier
            message_count_delta: Messages to add
            tool_call_count_delta: Tool calls to add
            tokens_delta: Tokens to add
        """
        session = self.get_session(session_id)
        if not session:
            return

        session.message_count += message_count_delta
        session.tool_call_count += tool_call_count_delta
        session.total_tokens += tokens_delta

        self._save_session(session)

    def update_session_title(self, session_id: str, title: str) -> None:
        """
        Update session title.

        Args:
            session_id: Session identifier
            title: New title
        """
        session = self.get_session(session_id)
        if session:
            session.title = title
            self._save_session(session)

    def add_session_topic(self, session_id: str, topic: str) -> None:
        """
        Add topic tag to session.

        Args:
            session_id: Session identifier
            topic: Topic to add
        """
        session = self.get_session(session_id)
        if session and topic not in session.topic_tags:
            session.topic_tags.append(topic)
            session.topic_tags = session.topic_tags[-10:]  # Keep last 10
            self._save_session(session)

    def generate_session_title(self, first_query: str) -> str:
        """
        Generate session title from first query.

        Args:
            first_query: First user query in session

        Returns:
            Generated title
        """
        # Simple approach: truncate first query
        if len(first_query) <= 50:
            return first_query

        # Find a good break point
        truncated = first_query[:50]
        last_space = truncated.rfind(" ")
        if last_space > 20:
            truncated = truncated[:last_space]

        return truncated + "..."

    # === User Profile Methods ===

    def get_user_profile(self) -> UserProfile:
        """
        Get user profile.

        Returns:
            UserProfile for current user
        """
        # Try SQLite-specific method
        if hasattr(self.storage, 'get_user_profile'):
            data = self.storage.get_user_profile(self.user_id)
            if data:
                return UserProfile.from_dict(data)

        # Try key-value storage
        key = f"{self.USER_KEY_PREFIX}{self.user_id}:profile"
        data = self.storage.get_json(key)

        if data:
            return UserProfile.from_dict(data)

        # Return default profile
        return UserProfile(user_id=self.user_id)

    def save_user_profile(self, profile: UserProfile) -> bool:
        """
        Save user profile.

        Args:
            profile: UserProfile to save

        Returns:
            Success boolean
        """
        profile.last_updated = datetime.now()

        # Try SQLite-specific method
        if hasattr(self.storage, 'save_user_profile'):
            return self.storage.save_user_profile(self.user_id, profile.to_dict())

        # Fallback to key-value
        key = f"{self.USER_KEY_PREFIX}{self.user_id}:profile"
        return self.storage.set_json(key, profile.to_dict())

    def update_user_profile(self, updates: Dict[str, Any]) -> UserProfile:
        """
        Update user profile with new data.

        Args:
            updates: Dict of fields to update

        Returns:
            Updated UserProfile
        """
        profile = self.get_user_profile()

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self.save_user_profile(profile)
        return profile

    # === Interaction Methods (for episodic memory bridge) ===

    def save_interaction(self, interaction: Interaction) -> bool:
        """
        Save an interaction (for episodic memory).

        Args:
            interaction: Interaction to save

        Returns:
            Success boolean
        """
        if not interaction.session_id:
            session = self.get_current_session()
            if session:
                interaction.session_id = session.session_id

        # Use SQLite-specific method if available
        if hasattr(self.storage, 'save_interaction'):
            return self.storage.save_interaction(interaction.to_dict())

        # Fallback: store in key-value
        key = f"kautilya:interaction:{interaction.interaction_id}"
        return self.storage.set_json(key, interaction.to_dict())

    def get_recent_interactions(self, limit: int = 10) -> List[Interaction]:
        """
        Get recent interactions for current user.

        Args:
            limit: Maximum number of interactions

        Returns:
            List of recent Interactions
        """
        if hasattr(self.storage, 'get_recent_interactions'):
            data_list = self.storage.get_recent_interactions(self.user_id, limit)
            return [Interaction.from_dict(d) for d in data_list]

        return []

    # === Private Methods ===

    def _save_session(self, session: Session) -> bool:
        """Save session to storage."""
        # Try SQLite-specific method
        if hasattr(self.storage, 'save_session'):
            return self.storage.save_session(session.to_dict())

        # Fallback to key-value
        key = f"{self.SESSION_KEY_PREFIX}{session.session_id}"
        return self.storage.set_json(key, session.to_dict())

    def _set_active_session(self, session_id: str) -> bool:
        """Set the active session for current user."""
        key = f"{self.ACTIVE_SESSION_KEY}{self.user_id}"
        return self.storage.set(key, session_id)


def create_session_manager(
    storage: StorageBackend,
    user_id: Optional[str] = None,
) -> SessionManager:
    """
    Factory function to create SessionManager.

    Args:
        storage: Storage backend instance
        user_id: Optional user identifier

    Returns:
        SessionManager instance
    """
    return SessionManager(storage=storage, user_id=user_id)

"""
Enhanced LLM History Validation and Recovery

Module: kautilya/llm_validation_enhanced.py

Production-grade validation and auto-recovery mechanisms to prevent
LLM API errors in production environments.

Usage:
    from kautilya.llm_validation_enhanced import EnhancedValidator

    validator = EnhancedValidator(history)
    is_valid, error = validator.validate_strict()
    if not is_valid:
        recovered = validator.sanitize_and_recover()
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """Circuit breaker for repeated validation failures."""

    failure_count: int = 0
    last_failure_time: Optional[float] = None
    circuit_open: bool = False
    max_failures: int = 3
    reset_timeout: int = 60  # seconds

    def record_failure(self) -> None:
        """Record a validation failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def record_success(self) -> None:
        """Record a successful validation."""
        self.failure_count = 0
        self.circuit_open = False

    def check_and_reset(self) -> bool:
        """
        Check if circuit breaker is open and reset if timeout passed.

        Returns:
            True if circuit is closed (can proceed), False if open
        """
        if not self.circuit_open:
            return True

        if self.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.last_failure_time
        if time_since_failure >= self.reset_timeout:
            # Reset circuit breaker
            logger.info("Circuit breaker reset after timeout")
            self.circuit_open = False
            self.failure_count = 0
            return True

        logger.warning(
            f"Circuit breaker is open. Retry in "
            f"{self.reset_timeout - time_since_failure:.0f}s"
        )
        return False


@dataclass
class ValidationResult:
    """Result of history validation."""

    is_valid: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class EnhancedValidator:
    """
    Enhanced validation and recovery for LLM conversation history.

    Validates:
    - Tool calls have matching responses
    - Tool responses appear AFTER their tool_calls (order)
    - No duplicate tool responses
    - Valid message role sequences
    """

    def __init__(self, history: Any):
        """
        Initialize validator.

        Args:
            history: ConversationHistory object with to_list() method
        """
        self.history = history
        self.circuit_breaker = CircuitBreaker()

    def validate_strict(self) -> Tuple[bool, Optional[str]]:
        """
        Strict validation of conversation history with detailed error reporting.

        Returns:
            (is_valid, error_message)
        """
        messages = self.history.to_list()
        pending_tool_calls: Dict[str, int] = {}  # tool_call_id -> message_index
        seen_tool_responses: Set[str] = set()

        for idx, msg in enumerate(messages):
            role = msg.get("role")

            # Validate tool messages
            if role == "tool":
                # Tool message MUST follow an assistant message with tool_calls
                if not pending_tool_calls:
                    return (
                        False,
                        f"Tool message at index {idx} has no preceding tool_calls",
                    )

                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    return False, f"Tool message at index {idx} missing tool_call_id"

                # Check if this tool_call_id was actually requested
                if tool_call_id not in pending_tool_calls:
                    return (
                        False,
                        f"Tool message at index {idx} references unknown "
                        f"tool_call_id: {tool_call_id}",
                    )

                # Check for duplicate responses
                if tool_call_id in seen_tool_responses:
                    return (
                        False,
                        f"Duplicate tool response for tool_call_id: {tool_call_id}",
                    )

                # Mark as responded
                seen_tool_responses.add(tool_call_id)
                del pending_tool_calls[tool_call_id]

            elif role == "assistant" and msg.get("tool_calls"):
                # Add all tool_call_ids from this assistant message
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        if tc_id in pending_tool_calls:
                            return False, f"Duplicate tool_call_id: {tc_id}"
                        pending_tool_calls[tc_id] = idx

        # Check for orphaned tool calls (calls without responses)
        if pending_tool_calls:
            orphaned_ids = list(pending_tool_calls.keys())
            return (
                False,
                f"Tool calls without responses: {orphaned_ids[:3]}"
                + ("..." if len(orphaned_ids) > 3 else ""),
            )

        return True, None

    def sanitize_history(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Attempt to sanitize history by removing invalid messages.

        Strategy:
        1. Remove orphaned tool responses (no matching tool_call)
        2. Remove incomplete tool call sequences
        3. Preserve system and valid user/assistant messages

        Returns:
            (success, sanitized_messages)
        """
        messages = self.history.to_list()
        valid_messages: List[Dict[str, Any]] = []
        pending_tool_calls: Set[str] = set()

        logger.info(f"Sanitizing history with {len(messages)} messages")

        for msg in messages:
            role = msg.get("role")

            # Always keep system and user messages
            if role in ["system", "user"]:
                valid_messages.append(msg)
                continue

            # Handle assistant messages
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Track expected tool responses
                    for tc in tool_calls:
                        if tc.get("id"):
                            pending_tool_calls.add(tc["id"])
                valid_messages.append(msg)
                continue

            # Handle tool messages - only keep if matching a pending call
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in pending_tool_calls:
                    valid_messages.append(msg)
                    pending_tool_calls.remove(tool_call_id)
                else:
                    logger.debug(f"Removing orphaned tool response: {tool_call_id}")
                # else: skip this orphaned tool response

        # If there are still pending tool calls, remove the assistant message
        # that made them (incomplete sequence)
        if pending_tool_calls:
            logger.info(
                f"Removing {len(pending_tool_calls)} incomplete tool call sequences"
            )

            # Remove from the end backwards to preserve earlier valid sequences
            i = len(valid_messages) - 1
            while i >= 0 and pending_tool_calls:
                msg = valid_messages[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tool_calls_to_remove = [
                        tc
                        for tc in msg["tool_calls"]
                        if tc.get("id") in pending_tool_calls
                    ]
                    if tool_calls_to_remove:
                        # Remove this entire message
                        logger.debug(f"Removing assistant message at index {i}")
                        valid_messages.pop(i)
                        for tc in msg["tool_calls"]:
                            pending_tool_calls.discard(tc.get("id"))
                i -= 1

        removed_count = len(messages) - len(valid_messages)
        logger.info(f"Sanitization removed {removed_count} invalid messages")

        return True, valid_messages

    def recover_and_validate(
        self, max_attempts: int = 3
    ) -> Tuple[bool, Optional[str]]:
        """
        Attempt to recover from validation failures.

        Args:
            max_attempts: Maximum recovery attempts

        Returns:
            (success, error_message)
        """
        # Check circuit breaker
        if not self.circuit_breaker.check_and_reset():
            return (
                False,
                "Circuit breaker is open. Too many recent failures. "
                f"Retry in {self.circuit_breaker.reset_timeout}s",
            )

        attempt = 1
        while attempt <= max_attempts:
            # Validate current state
            is_valid, error_msg = self.validate_strict()

            if is_valid:
                logger.info(f"Validation succeeded on attempt {attempt}")
                self.circuit_breaker.record_success()
                return True, None

            logger.warning(f"Validation failed (attempt {attempt}/{max_attempts}): {error_msg}")

            if attempt >= max_attempts:
                # Max attempts reached
                self.circuit_breaker.record_failure()
                return (
                    False,
                    f"Validation failed after {max_attempts} attempts. "
                    f"Last error: {error_msg}",
                )

            # Attempt sanitization
            logger.info(f"Attempting history sanitization (attempt {attempt})")
            success, sanitized_messages = self.sanitize_history()

            if success:
                # Replace history with sanitized version
                self.history.clear()
                from .llm_models import Message  # Import Message class

                for msg in sanitized_messages:
                    self.history.add(Message(**msg))

                logger.info("History sanitization successful, re-validating")
            else:
                logger.error("History sanitization failed")
                self.circuit_breaker.record_failure()
                return False, "Could not sanitize invalid history"

            attempt += 1

        # Should not reach here
        return False, "Unknown error in recovery loop"


def log_validation_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Log validation events for production monitoring.

    Args:
        event_type: Type of event (validation_failed, sanitized, circuit_breaker_opened, etc.)
        details: Event details dictionary
    """
    import json
    from datetime import datetime

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details,
    }
    logger.info(f"VALIDATION_EVENT: {json.dumps(event)}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Example: Simulate validation failure and recovery
    from .llm_models import ConversationHistory, Message

    history = ConversationHistory()
    history.add(Message(role="system", content="You are a helpful assistant"))
    history.add(Message(role="user", content="Hello"))

    # Simulate orphaned tool response (invalid)
    history.add(
        Message(role="tool", content='{"result": "test"}', tool_call_id="invalid_id")
    )

    validator = EnhancedValidator(history)

    # Try validation
    is_valid, error = validator.validate_strict()
    print(f"Valid: {is_valid}, Error: {error}")

    # Try recovery
    if not is_valid:
        success, error = validator.recover_and_validate()
        print(f"Recovery success: {success}, Error: {error}")

        # Check final state
        is_valid, error = validator.validate_strict()
        print(f"After recovery - Valid: {is_valid}, Error: {error}")

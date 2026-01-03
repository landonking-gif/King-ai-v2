"""
Integration module for AgentCore with Kautilya interactive mode.

Provides a bridge between the existing interactive.py/llm_client.py
and the new agentic capabilities in the agent package.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from .core import AgentCore, ProcessingResult
from .file_resolver import FileMatch, FileResolver

logger = logging.getLogger(__name__)


class AgentIntegration:
    """
    Integration layer for AgentCore with Kautilya's interactive mode.

    Provides methods that can be used alongside or instead of
    the existing LLM client streaming approach.
    """

    def __init__(
        self,
        llm_client: Any,
        tool_executor: Optional[Any] = None,
        skills_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        """
        Initialize agent integration.

        Args:
            llm_client: The KautilyaLLMClient instance
            tool_executor: Optional tool executor
            skills_dir: Directory containing skills
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.verbose = verbose

        # Create a wrapper that adapts KautilyaLLMClient to AgentCore's expected interface
        self.llm_adapter = LLMClientAdapter(llm_client)

        # Initialize AgentCore
        self.agent = AgentCore(
            llm_client=self.llm_adapter,
            tool_executor=tool_executor,
            skills_dir=skills_dir,
            verbose=verbose,
        )

        # File resolver for quick file lookups
        self.file_resolver = FileResolver()

    def resolve_file_references(self, text: str) -> List[FileMatch]:
        """
        Resolve @file references in text.

        Args:
            text: Text containing @file references

        Returns:
            List of resolved file matches
        """
        references = self.file_resolver.extract_file_references(text)
        resolved = []

        for ref in references:
            try:
                match = self.file_resolver.resolve(ref)
                resolved.append(match)
                logger.info(f"Resolved '{ref}' -> {match.path}")
            except FileNotFoundError as e:
                logger.warning(f"Could not resolve: {ref} - {e}")

        return resolved

    async def process_async(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """
        Process a request asynchronously using AgentCore.

        Args:
            request: User's request
            context: Additional context

        Returns:
            ProcessingResult with response and metadata
        """
        return await self.agent.process(request, context)

    def process(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """
        Process a request synchronously using AgentCore.

        Args:
            request: User's request
            context: Additional context

        Returns:
            ProcessingResult with response and metadata
        """
        return asyncio.run(self.process_async(request, context))

    def process_streaming(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, ProcessingResult]:
        """
        Process a request with streaming output.

        This provides a generator that yields progress updates
        while processing, and returns the final result.

        Args:
            request: User's request
            context: Additional context

        Yields:
            Progress messages during processing

        Returns:
            ProcessingResult with final response
        """
        # Resolve file references first
        resolved = self.resolve_file_references(request)
        if resolved:
            yield f"📎 Resolved {len(resolved)} file reference(s)\n"
            for match in resolved:
                yield f"  - {match.original_reference} -> {match.path}\n"

        # Show capability matching
        yield "🔍 Analyzing request...\n"

        # Run async processing
        result = self.process(request, context)

        # Yield trace info if available
        if result.react_trace and self.verbose:
            yield "\n📋 Reasoning trace:\n"
            for step in result.react_trace.steps:
                yield f"  - {step.thought[:100]}...\n"

        yield result

    def get_available_capabilities(self) -> List[Dict[str, Any]]:
        """Get list of available capabilities."""
        capabilities = self.agent.get_capabilities()
        return [
            {
                "name": cap.name,
                "type": cap.type,
                "description": cap.description,
            }
            for cap in capabilities
        ]

    def should_use_agentic_mode(self, request: str) -> bool:
        """
        Determine if a request should use agentic mode.

        Returns True for complex requests that would benefit from:
        - File resolution
        - Skill execution
        - Multi-step reasoning

        Args:
            request: User's request

        Returns:
            True if agentic mode is recommended
        """
        request_lower = request.lower()

        # Check for file references
        if "@" in request:
            return True

        # Check for skill-related keywords
        skill_keywords = [
            "extract",
            "analyze",
            "summarize",
            "research",
            "document",
            "pdf",
            "report",
        ]
        if any(kw in request_lower for kw in skill_keywords):
            return True

        # Check for multi-step indicators
        multi_step_indicators = [
            " and ",
            " then ",
            "first ",
            "after that",
            "finally",
        ]
        if any(ind in request_lower for ind in multi_step_indicators):
            return True

        # Check for explicit tool/skill requests
        if "skill" in request_lower or "tool" in request_lower:
            return True

        return False

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current agent session."""
        return self.agent.get_session_summary()


class LLMClientAdapter:
    """
    Adapter that wraps KautilyaLLMClient to provide the interface
    expected by AgentCore.
    """

    def __init__(self, llm_client: Any):
        """
        Initialize adapter.

        Args:
            llm_client: KautilyaLLMClient instance
        """
        self.llm_client = llm_client

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat message and get response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Response string
        """
        # KautilyaLLMClient uses a generator for streaming
        # We need to collect the full response
        response_text = ""

        try:
            # Try to use non-streaming mode if available
            if hasattr(self.llm_client, "chat_sync"):
                return self.llm_client.chat_sync(prompt)

            # Otherwise use the streaming generator
            gen = self.llm_client.chat(prompt, stream=True)
            for chunk in gen:
                if isinstance(chunk, str):
                    response_text += chunk

            return response_text.strip()

        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise

    def clear_history(self) -> None:
        """Clear conversation history."""
        if hasattr(self.llm_client, "clear_history"):
            self.llm_client.clear_history()


def create_agent_integration(
    llm_client: Any,
    tool_executor: Optional[Any] = None,
    config_dir: Optional[str] = None,
    verbose: bool = False,
) -> Optional[AgentIntegration]:
    """
    Factory function to create AgentIntegration.

    Args:
        llm_client: KautilyaLLMClient instance
        tool_executor: Optional tool executor
        config_dir: Configuration directory
        verbose: Enable verbose mode

    Returns:
        AgentIntegration instance or None if creation fails
    """
    try:
        # Determine skills directory
        skills_dir = None
        if config_dir:
            config_path = Path(config_dir)
            if (config_path / "skills").exists():
                skills_dir = config_path / "skills"

        return AgentIntegration(
            llm_client=llm_client,
            tool_executor=tool_executor,
            skills_dir=skills_dir,
            verbose=verbose,
        )

    except Exception as e:
        logger.error(f"Failed to create AgentIntegration: {e}")
        return None

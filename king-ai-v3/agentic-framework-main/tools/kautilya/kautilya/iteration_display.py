"""
Rich iteration display for agentic loop with progress animations.

Module: kautilya/iteration_display.py

Provides beautiful visual feedback for multi-step agent execution.
Supports two display modes:
- minimal: Compact one-liner with key metrics
- detailed: Rich panel with full progress information

Uses LLM adapters for provider-agnostic dynamic follow-up generation.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.live import Live

# Add adapters to path for unified configuration
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import adapter factory for LLM calls
try:
    from adapters.llm import create_sync_adapter
    _ADAPTERS_AVAILABLE = True
except ImportError:
    _ADAPTERS_AVAILABLE = False


class DisplayMode(str, Enum):
    """Display mode for iteration feedback."""
    MINIMAL = "minimal"
    DETAILED = "detailed"


class OutputMode(str, Enum):
    """Output verbosity mode for agent responses."""
    CONCISE = "concise"
    VERBOSE = "verbose"


class IterationPhase(str, Enum):
    """Phases of an iteration."""
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    DECIDING = "deciding"
    COMPLETE = "complete"


# Global display mode - can be changed at runtime
_current_display_mode: DisplayMode = DisplayMode.DETAILED

# Global output verbosity mode - default to verbose for transparency
# This ensures sources are always shown by default
_current_output_mode: OutputMode = OutputMode.VERBOSE


def get_display_mode() -> DisplayMode:
    """Get the current display mode."""
    global _current_display_mode
    # Check environment variable first
    env_mode = os.getenv("KAUTILYA_DISPLAY_MODE", "").lower()
    if env_mode in ("minimal", "detailed"):
        return DisplayMode(env_mode)
    return _current_display_mode


def set_display_mode(mode: DisplayMode) -> None:
    """Set the current display mode."""
    global _current_display_mode
    _current_display_mode = mode


def toggle_display_mode() -> DisplayMode:
    """Toggle between minimal and detailed modes."""
    global _current_display_mode
    if _current_display_mode == DisplayMode.MINIMAL:
        _current_display_mode = DisplayMode.DETAILED
    else:
        _current_display_mode = DisplayMode.MINIMAL
    return _current_display_mode


def get_output_mode() -> OutputMode:
    """Get the current output verbosity mode."""
    global _current_output_mode
    # Check environment variable first
    env_mode = os.getenv("KAUTILYA_OUTPUT_MODE", "").lower()
    if env_mode in ("concise", "verbose"):
        return OutputMode(env_mode)
    return _current_output_mode


def set_output_mode(mode: OutputMode) -> None:
    """Set the current output verbosity mode."""
    global _current_output_mode
    _current_output_mode = mode


def toggle_output_mode() -> OutputMode:
    """Toggle between concise and verbose output modes."""
    global _current_output_mode
    if _current_output_mode == OutputMode.CONCISE:
        _current_output_mode = OutputMode.VERBOSE
    else:
        _current_output_mode = OutputMode.CONCISE
    return _current_output_mode


def is_verbose_output() -> bool:
    """Check if output mode is verbose."""
    return get_output_mode() == OutputMode.VERBOSE


class IterationDisplay:
    """Display manager for agentic loop iterations with rich animations."""

    # Phase emojis and colors
    PHASE_CONFIG = {
        IterationPhase.THINKING: {
            "emoji": "🤔",
            "label": "Thinking",
            "color": "cyan",
            "description": "Analyzing the problem"
        },
        IterationPhase.PLANNING: {
            "emoji": "📋",
            "label": "Planning",
            "color": "blue",
            "description": "Deciding which tools to use"
        },
        IterationPhase.EXECUTING: {
            "emoji": "⚡",
            "label": "Executing",
            "color": "yellow",
            "description": "Running tools and gathering data"
        },
        IterationPhase.REVIEWING: {
            "emoji": "🔍",
            "label": "Reviewing",
            "color": "magenta",
            "description": "Analyzing results"
        },
        IterationPhase.DECIDING: {
            "emoji": "🎯",
            "label": "Deciding",
            "color": "green",
            "description": "Determining next action"
        },
        IterationPhase.COMPLETE: {
            "emoji": "✅",
            "label": "Complete",
            "color": "bright_green",
            "description": "Task finished"
        },
    }

    # Tool type emojis for minimal mode
    TOOL_EMOJIS = {
        "file_read": "📖",
        "file_write": "✏️",
        "file_edit": "✏️",
        "file_glob": "🔍",
        "file_grep": "🔎",
        "bash_exec": "💻",
        "web_search": "🌐",
        "web_fetch": "🌐",
        "mcp_call": "🔌",
        "default": "⚡",
    }

    def __init__(self, console: Console, max_iterations: int):
        """
        Initialize iteration display.

        Args:
            console: Rich console for output
            max_iterations: Maximum number of iterations
        """
        self.console = console
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.current_phase = None
        self.tools_executed: List[str] = []
        self.iteration_start_time: Optional[float] = None
        self.tokens_used: int = 0
        self.current_thinking: str = ""
        self.current_tool: Optional[str] = None
        self.current_tool_target: Optional[str] = None

    def _get_mode(self) -> DisplayMode:
        """Get current display mode."""
        return get_display_mode()

    def _get_tool_emoji(self, tool_name: str) -> str:
        """Get emoji for a tool type."""
        for key, emoji in self.TOOL_EMOJIS.items():
            if key in tool_name.lower():
                return emoji
        return self.TOOL_EMOJIS["default"]

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m{secs}s"

    def _format_tokens(self, tokens: int) -> str:
        """Format token count."""
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}k"
        return str(tokens)

    def start_iteration(self, iteration: int, thinking: Optional[str] = None):
        """Start a new iteration."""
        self.current_iteration = iteration
        self.iteration_start_time = time.time()
        self.current_thinking = thinking or ""
        self.current_tool = None
        self.current_tool_target = None

        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            self._start_iteration_detailed(iteration)
        # Minimal mode doesn't show iteration start - waits for tool call

    def _start_iteration_detailed(self, iteration: int):
        """Start iteration in detailed mode."""
        progress_bar = self._create_progress_bar(iteration, self.max_iterations)

        header = Panel(
            Text.from_markup(
                f"[bold cyan]Iteration {iteration}/{self.max_iterations}[/bold cyan]\n"
                f"{progress_bar}"
            ),
            border_style="cyan",
            padding=(0, 2),
        )
        self.console.print(header)

    def _create_progress_bar(self, current: int, total: int) -> str:
        """Create a text-based progress bar."""
        filled = int((current / total) * 20)
        bar = "━" * filled + "─" * (20 - filled)
        percentage = int((current / total) * 100)
        return f"{bar} {percentage}%"

    def show_phase(self, phase: IterationPhase, detail: Optional[str] = None):
        """
        Show current phase with animation.

        Args:
            phase: Current phase
            detail: Optional detail message
        """
        self.current_phase = phase
        if detail:
            self.current_thinking = detail

        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            self._show_phase_detailed(phase, detail)
        # Minimal mode doesn't show phases separately

    def _show_phase_detailed(self, phase: IterationPhase, detail: Optional[str] = None):
        """Show phase in detailed mode."""
        config = self.PHASE_CONFIG[phase]

        phase_text = Text()
        phase_text.append(f"{config['emoji']} ", style=config['color'])
        phase_text.append(config['label'], style=f"bold {config['color']}")

        if detail:
            phase_text.append(f": {detail}", style="dim")
        else:
            phase_text.append(f" - {config['description']}", style="dim")

        self.console.print(phase_text)

    def show_tool_execution(
        self,
        tool_name: str,
        args: Optional[dict] = None,
        thinking: Optional[str] = None
    ):
        """
        Show tool being executed.

        Args:
            tool_name: Name of the tool
            args: Optional tool arguments
            thinking: Optional reasoning/thinking snippet
        """
        self.tools_executed.append(tool_name)
        self.current_tool = tool_name

        # Extract target from args
        if args:
            target = args.get("file_path") or args.get("path") or args.get("pattern") or args.get("command")
            if target:
                # Truncate long targets
                if isinstance(target, str) and len(target) > 40:
                    target = "..." + target[-37:]
                self.current_tool_target = str(target)
            else:
                self.current_tool_target = None

        if thinking:
            self.current_thinking = thinking

        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            self._show_tool_execution_detailed(tool_name, args)
        else:
            self._show_tool_execution_minimal(tool_name, args)

    def _show_tool_execution_detailed(self, tool_name: str, args: Optional[dict] = None):
        """Show tool execution in detailed mode."""
        tool_text = Text()
        tool_text.append("  ⚡ Executing: ", style="yellow")
        tool_text.append(tool_name, style="bold yellow")

        if args:
            arg_items = list(args.items())[:2]
            if arg_items:
                args_str = ", ".join(f"{k}={v}" for k, v in arg_items if v is not None)
                if args_str:
                    tool_text.append(f" ({args_str[:60]})", style="dim")

        self.console.print(tool_text)

    def _show_tool_execution_minimal(self, tool_name: str, args: Optional[dict] = None):
        """Show tool execution in minimal mode - compact one-liner."""
        elapsed = time.time() - self.iteration_start_time if self.iteration_start_time else 0
        emoji = self._get_tool_emoji(tool_name)

        # Build compact line: [5/10] 📖 file_read → config.yaml │ 💭 "thinking..." │ ⏱️ 2.1s │ 🎫 2.4k
        parts = []

        # Progress
        parts.append(f"[dim][{self.current_iteration}/{self.max_iterations}][/dim]")

        # Tool and target
        parts.append(f"{emoji} [bold yellow]{tool_name}[/bold yellow]")
        if self.current_tool_target:
            parts.append(f"→ [cyan]{self.current_tool_target}[/cyan]")

        # Thinking snippet (truncated)
        if self.current_thinking:
            thinking_short = self.current_thinking[:35]
            if len(self.current_thinking) > 35:
                thinking_short += "..."
            parts.append(f"│ 💭 [dim]\"{thinking_short}\"[/dim]")

        # Metrics
        parts.append(f"│ ⏱️ [dim]{self._format_duration(elapsed)}[/dim]")
        if self.tokens_used > 0:
            parts.append(f"│ 🎫 [dim]{self._format_tokens(self.tokens_used)}[/dim]")

        self.console.print(Text.from_markup(" ".join(parts)))

    def show_tool_result(
        self,
        tool_name: str,
        success: bool = True,
        summary: Optional[str] = None
    ):
        """
        Show tool execution result.

        Args:
            tool_name: Name of the tool
            success: Whether execution succeeded
            summary: Optional result summary
        """
        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            self._show_tool_result_detailed(tool_name, success, summary)
        # Minimal mode doesn't show individual tool results

    def _show_tool_result_detailed(
        self,
        tool_name: str,
        success: bool = True,
        summary: Optional[str] = None
    ):
        """Show tool result in detailed mode."""
        if success:
            result_text = Text()
            result_text.append("  ✓ ", style="green")
            result_text.append(tool_name, style="green")
            if summary:
                result_text.append(f": {summary[:80]}", style="dim")
            self.console.print(result_text)
        else:
            result_text = Text()
            result_text.append("  ✗ ", style="red")
            result_text.append(tool_name, style="red")
            result_text.append(" failed", style="dim red")
            self.console.print(result_text)

    def update_tokens(self, tokens: int):
        """Update token count for current iteration."""
        self.tokens_used = tokens

    def update_thinking(self, thinking: str):
        """Update thinking/reasoning snippet."""
        self.current_thinking = thinking

    def show_iteration_summary(self, duration: Optional[float] = None, tokens: Optional[int] = None):
        """Show summary of current iteration."""
        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            self._show_iteration_summary_detailed(duration, tokens)
        # Minimal mode already shows info inline

        # Reset for next iteration
        self.tools_executed = []
        self.current_thinking = ""
        self.current_tool = None
        self.current_tool_target = None

    def _show_iteration_summary_detailed(
        self,
        duration: Optional[float] = None,
        tokens: Optional[int] = None
    ):
        """Show iteration summary in detailed mode."""
        if not self.tools_executed and not duration:
            return

        summary_parts = []

        if self.tools_executed:
            summary_parts.append(f"Tools: [cyan]{', '.join(self.tools_executed)}[/cyan]")

        if duration:
            summary_parts.append(f"⏱️ {self._format_duration(duration)}")

        if tokens:
            summary_parts.append(f"🎫 {self._format_tokens(tokens)} tokens")

        if summary_parts:
            self.console.print(Text.from_markup(f"  [dim]{' │ '.join(summary_parts)}[/dim]"))

    def show_completion(self, total_iterations: int, total_tools: int, total_duration: Optional[float] = None):
        """
        Show completion summary.

        Args:
            total_iterations: Total iterations completed
            total_tools: Total tools executed
            total_duration: Total time elapsed
        """
        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            self._show_completion_detailed(total_iterations, total_tools, total_duration)
        else:
            self._show_completion_minimal(total_iterations, total_tools, total_duration)

    def _show_completion_detailed(
        self,
        total_iterations: int,
        total_tools: int,
        total_duration: Optional[float] = None
    ):
        """Show completion in detailed mode."""
        duration_str = f"\n[dim]Duration: {self._format_duration(total_duration)}[/dim]" if total_duration else ""

        completion_panel = Panel(
            Text.from_markup(
                f"[bold green]✓ Task Complete[/bold green]\n\n"
                f"[dim]Iterations: {total_iterations}/{self.max_iterations}[/dim]\n"
                f"[dim]Tools executed: {total_tools}[/dim]"
                f"{duration_str}"
            ),
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(completion_panel)

    def _show_completion_minimal(
        self,
        total_iterations: int,
        total_tools: int,
        total_duration: Optional[float] = None
    ):
        """Show completion in minimal mode."""
        duration_str = f" │ ⏱️ {self._format_duration(total_duration)}" if total_duration else ""
        self.console.print(Text.from_markup(
            f"[bold green]✓ Complete[/bold green] │ "
            f"[dim]{total_iterations} iterations │ {total_tools} tools{duration_str}[/dim]"
        ))

    def show_max_iterations_reached(self):
        """Show that max iterations was reached."""
        mode = self._get_mode()

        if mode == DisplayMode.DETAILED:
            warning_panel = Panel(
                Text.from_markup(
                    f"[bold yellow]⚠️  Max Iterations Reached[/bold yellow]\n\n"
                    f"[dim]Stopped after {self.max_iterations} iterations[/dim]\n"
                    f"[dim]Increase with: /llm set-params --max-iterations N[/dim]"
                ),
                border_style="yellow",
                padding=(1, 2),
            )
            self.console.print(warning_panel)
        else:
            self.console.print(Text.from_markup(
                f"[bold yellow]⚠️ Max iterations ({self.max_iterations}) reached[/bold yellow]"
            ))


def get_numbered_sources() -> Dict[int, Any]:
    """
    Get sources with assigned reference numbers for inline citations.

    Returns:
        Dictionary mapping reference numbers to source entries
    """
    from kautilya.tool_executor import get_source_tracker

    tracker = get_source_tracker()
    sources = tracker.get_sources()

    numbered = {}
    for i, source in enumerate(sources, 1):
        numbered[i] = source

    return numbered


def inject_inline_citations(response_text: str, sources: List[Any]) -> str:
    """
    Post-process response to inject inline citations [1], [2], etc.

    This is more reliable than relying on the LLM to add citations
    because we programmatically match sources to relevant content.

    Args:
        response_text: The LLM's response text
        sources: List of SourceEntry objects

    Returns:
        Response text with inline citations injected
    """
    if not sources or not response_text:
        return response_text

    import re

    # Build keyword to source mapping
    # Extract key terms from each source for matching
    source_keywords: Dict[int, List[str]] = {}

    for i, source in enumerate(sources, 1):
        keywords = []

        # Extract keywords from location
        loc = source.location.lower()

        # For web sources, extract domain and path keywords
        if source.source_type.value in ("web_fetch", "web_search"):
            # Extract domain name parts
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', loc)
            if domain_match:
                domain = domain_match.group(1)
                # Add domain parts as keywords (e.g., "reuters", "bloomberg")
                for part in domain.replace('.', ' ').split():
                    if len(part) > 3 and part not in ('com', 'org', 'net', 'www', 'http', 'https'):
                        keywords.append(part)

        # For file sources, extract filename
        elif source.source_type.value in ("file_read", "file_search", "config_read"):
            # Get filename without path
            filename = loc.split('/')[-1].split('\\')[-1]
            if '.' in filename:
                name_part = filename.rsplit('.', 1)[0]
                keywords.append(name_part.lower())
            keywords.append(filename.lower())

        # For MCP sources, extract tool name
        elif source.source_type.value == "mcp_call":
            # location is usually "tool_id.tool_name"
            parts = loc.split('.')
            keywords.extend([p.lower() for p in parts if len(p) > 2])

        # Add description keywords
        if source.description:
            desc = source.description.lower()
            # Extract significant words (>4 chars, not common words)
            common_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been', 'were', 'call', 'called', 'search', 'read', 'wrote'}
            for word in re.findall(r'\b[a-z]{4,}\b', desc):
                if word not in common_words:
                    keywords.append(word)

        source_keywords[i] = list(set(keywords))[:5]  # Limit keywords per source

    # Now scan response for places to inject citations
    # We'll add citations after sentences that mention source-related keywords

    result_lines = []
    response_lines = response_text.split('\n')

    citations_added = set()

    for line in response_lines:
        line_lower = line.lower()
        line_citations = []

        # Check each source for keyword matches in this line
        for source_num, keywords in source_keywords.items():
            if source_num in citations_added:
                continue  # Already cited this source

            for keyword in keywords:
                if keyword in line_lower and len(keyword) > 3:
                    line_citations.append(source_num)
                    citations_added.add(source_num)
                    break

        # Add citations to end of line if found
        if line_citations and line.strip():
            # Sort citations numerically
            line_citations.sort()
            citation_str = ''.join(f'[{n}]' for n in line_citations[:3])  # Max 3 per line

            # Add citation after punctuation or at end
            if line.rstrip().endswith(('.', '!', '?', ':')):
                line = line.rstrip()[:-1] + ' ' + citation_str + line.rstrip()[-1]
            else:
                line = line.rstrip() + ' ' + citation_str

        result_lines.append(line)

    return '\n'.join(result_lines)


def generate_llm_followups(
    user_query: str,
    response_content: str,
) -> Optional[List[str]]:
    """
    Generate dynamic follow-up questions using an LLM call.

    Makes a lightweight LLM call to generate truly contextual and engaging
    follow-up questions based on the actual conversation content.

    Args:
        user_query: The original user query
        response_content: The generated response content

    Returns:
        List of 3 contextual follow-up questions, or None if LLM call fails
    """
    # Check if dynamic follow-ups are enabled
    dynamic_enabled = os.getenv("KAUTILYA_DYNAMIC_FOLLOWUPS", "true").lower() == "true"
    if not dynamic_enabled:
        return None

    # Don't make LLM call for very short responses
    if len(response_content) < 100:
        return None

    try:
        # Use adapter factory for provider-agnostic LLM call
        if _ADAPTERS_AVAILABLE:
            adapter = create_sync_adapter()
        else:
            # Fallback to direct OpenAI if adapters not available
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None

            adapter = None
            client = OpenAI(api_key=api_key)

        # Truncate content if too long to keep costs low
        truncated_response = response_content[:2000] if len(response_content) > 2000 else response_content

        prompt = f"""Based on this conversation, generate exactly 3 engaging follow-up questions that would help the user get more value. Questions should be specific to the content discussed, not generic.

User asked: {user_query}

Response summary: {truncated_response[:1000]}

Generate 3 follow-up questions that:
1. Are specific to the topic discussed
2. Would lead to deeper insights or actionable next steps
3. Are concise (under 15 words each)
4. Are phrased naturally as questions

Return ONLY the 3 questions, one per line, without numbering or bullets."""

        # Call LLM via adapter or direct client
        if _ADAPTERS_AVAILABLE and adapter:
            questions_text = adapter.complete_text(prompt, temperature=0.7, max_tokens=150)
        else:
            # Fallback to direct OpenAI call
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            model_lower = model.lower()
            is_reasoning_model = any(x in model_lower for x in ["o1", "o3", "gpt-5", "reasoning"])

            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            if is_reasoning_model:
                api_params["max_completion_tokens"] = 1000
            else:
                api_params["max_tokens"] = 150
                api_params["temperature"] = 0.7

            response = client.chat.completions.create(**api_params)
            questions_text = response.choices[0].message.content.strip()

        # Parse the response
        questions = []

        for line in questions_text.split('\n'):
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Remove common prefixes (numbering, bullets)
            line = re.sub(r'^[\d]+[\.\)\-]\s*', '', line)  # Remove "1." "1)" "1-"
            line = re.sub(r'^[\-\*\•]\s*', '', line)  # Remove bullets
            line = line.strip()

            if len(line) < 10:
                continue

            # Ensure it ends with a question mark (add if missing)
            if not line.endswith('?'):
                line = line.rstrip('.') + '?'

            questions.append(line)

        # Return up to 3 questions
        return questions[:3] if questions else None

    except Exception as e:
        # Silently fall back to rule-based if LLM fails
        return None


def generate_contextual_followups(
    user_query: str,
    response_content: str,
) -> List[str]:
    """
    Generate contextual follow-up questions based on user query and response.

    First attempts to use LLM for dynamic generation, then falls back to
    rule-based generation if LLM is unavailable or fails.

    Args:
        user_query: The original user query
        response_content: The generated response content

    Returns:
        List of contextual follow-up questions
    """
    # Try LLM-based generation first
    llm_questions = generate_llm_followups(user_query, response_content)
    if llm_questions and len(llm_questions) >= 2:
        return llm_questions
    questions = []
    query_lower = user_query.lower()
    response_lower = response_content.lower()

    # Detect topic categories from query
    is_market_finance = any(w in query_lower for w in [
        "market", "stock", "share", "investment", "trading", "gold", "silver",
        "precious metal", "commodity", "portfolio", "sensex", "nifty", "price"
    ])
    is_geopolitical = any(w in query_lower for w in [
        "situation", "crisis", "conflict", "political", "country", "region",
        "government", "war", "tension", "impact"
    ])
    is_technical = any(w in query_lower for w in [
        "code", "config", "setup", "install", "error", "bug", "implement",
        "api", "function", "class", "file"
    ])
    is_analysis = any(w in query_lower for w in [
        "analyse", "analyze", "compare", "evaluate", "assess", "review",
        "outlook", "forecast", "predict"
    ])
    is_howto = any(w in query_lower for w in [
        "how to", "how do", "how can", "what is", "explain", "guide"
    ])

    # Extract key entities/topics from query for personalization
    # Simple extraction - get capitalized words, excluding common sentence starters
    import re
    common_starters = {'Tell', 'What', 'How', 'Why', 'When', 'Where', 'Who', 'Can', 'Could', 'Would', 'Should', 'Is', 'Are', 'Do', 'Does', 'Please', 'Help', 'Show', 'Give', 'Find', 'Search', 'Get', 'Make', 'Create', 'The', 'This', 'That', 'These', 'Those', 'My', 'Your', 'Our', 'Their'}
    potential_topics = [t for t in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_query) if t not in common_starters]
    main_topic = potential_topics[0] if potential_topics else ""

    # Generate contextual questions based on detected categories
    if is_market_finance:
        questions.extend([
            f"What specific stocks or sectors should I watch closely?",
            f"What's the short-term vs long-term outlook for investors?",
            f"How can I hedge my portfolio against these risks?",
        ])
    elif is_geopolitical:
        questions.extend([
            f"What are the latest developments in this situation?",
            f"How might this evolve in the coming weeks?",
            f"What are the key indicators to monitor?",
        ])
    elif is_technical:
        questions.extend([
            "Can you show me a code example for this?",
            "What are the common pitfalls to avoid?",
            "How do I test this implementation?",
        ])
    elif is_analysis:
        questions.extend([
            "What are the key risks and opportunities here?",
            "Can you provide more data to support this analysis?",
            "What's your confidence level in this assessment?",
        ])
    elif is_howto:
        questions.extend([
            "Can you provide step-by-step instructions?",
            "What tools or resources do I need?",
            "Are there any prerequisites I should know about?",
        ])

    # Add topic-specific personalization if we found a main topic
    if main_topic and len(questions) < 3:
        questions.insert(0, f"What's the latest news on {main_topic}?")

    # Detect content-specific follow-ups from response
    if "sector" in response_lower or "industry" in response_lower:
        questions.append("Which specific companies are most affected?")
    if "risk" in response_lower:
        questions.append("How can I mitigate these risks?")
    if "option" in response_lower or "alternative" in response_lower:
        questions.append("Which option would you recommend and why?")
    if "trend" in response_lower:
        questions.append("How long is this trend expected to continue?")

    # Fallback generic questions if we don't have enough
    generic_questions = [
        "Can you elaborate on the key points?",
        "What actions should I take based on this?",
        "Are there any important caveats I should know?",
    ]

    # Fill up to 3 questions
    for q in generic_questions:
        if len(questions) >= 3:
            break
        if q not in questions:
            questions.append(q)

    # Return unique questions, limited to 3
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen and len(unique_questions) < 3:
            seen.add(q)
            unique_questions.append(q)

    return unique_questions


def display_followup_questions(
    console: Console,
    user_query: str = "",
    response_content: str = "",
    questions: Optional[List[str]] = None,
) -> None:
    """
    Display suggested follow-up questions that are contextual and natural.

    ALWAYS displays follow-up questions for better UX continuity.

    Args:
        console: Rich console for output
        user_query: The original user query (for context)
        response_content: The generated response (for context)
        questions: Optional explicit list of questions. If None, generates contextual ones.
    """
    if not questions:
        if user_query or response_content:
            # Generate contextual questions based on query and response
            questions = generate_contextual_followups(user_query, response_content)
        else:
            # Fallback to source-based questions if no query/response provided
            from kautilya.tool_executor import get_source_tracker

            tracker = get_source_tracker()
            sources = tracker.get_sources()

            questions = []

            has_web = any(s.source_type.value in ("web_fetch", "web_search") for s in sources)
            has_files = any(s.source_type.value in ("file_read", "file_search") for s in sources)

            if has_web:
                questions.append("Can you search for more recent information on this?")
            if has_files:
                questions.append("Show me related code or configuration")

            # Always provide at least some generic follow-ups
            if len(questions) < 3:
                questions.extend([
                    "Can you elaborate on the key points?",
                    "What are the next steps I should take?",
                    "Are there any alternatives to consider?",
                ])

    # Limit to 3 questions
    questions = questions[:3]

    # ALWAYS show follow-up questions for UX continuity
    if questions:
        console.print()
        console.print(Text.from_markup("[bold cyan]💡 Follow-up Questions[/bold cyan]"))
        for i, q in enumerate(questions, 1):
            console.print(Text.from_markup(f"   [dim][{i}][/dim] {q}"))


def display_sources_panel(console: Console) -> None:
    """
    Display the enhanced source attribution panel with:
    - Grouped sources by type with numbered references [1][2]
    - Descriptions for each source
    - Key sources footer

    This is the main display function for Option A+D design.

    Args:
        console: Rich console for output
    """
    from kautilya.tool_executor import get_source_tracker, SourceType

    tracker = get_source_tracker()
    if not tracker.has_sources():
        return

    sources = tracker.get_sources()
    assumptions = tracker.get_assumptions()

    # Group sources by type
    groups: Dict[str, List[tuple]] = {
        "web": [],      # 🌐 Web
        "files": [],    # 📄 Files
        "mcp": [],      # 🔌 MCP Tools
        "commands": [], # ⚡ Commands
    }

    # Track which sources are most referenced (for key sources)
    key_sources = []

    for i, source in enumerate(sources, 1):
        st = source.source_type.value

        # Format location with line range if available
        loc = source.location
        if source.line_range:
            loc = f"{source.location}:{source.line_range}"

        # Truncate long locations
        if len(loc) > 35:
            loc = "..." + loc[-32:]

        # Truncate description
        desc = source.description[:25] if source.description else ""
        if source.description and len(source.description) > 25:
            desc += "..."

        entry = (i, loc, desc)

        if st in ("web_fetch", "web_search"):
            groups["web"].append(entry)
            if len(key_sources) < 3:
                key_sources.append((i, loc))
        elif st in ("file_read", "file_search", "config_read"):
            groups["files"].append(entry)
            if len(key_sources) < 3:
                key_sources.append((i, loc))
        elif st == "mcp_call":
            groups["mcp"].append(entry)
            if len(key_sources) < 3:
                key_sources.append((i, loc))
        elif st == "bash_exec":
            groups["commands"].append(entry)

    # Build panel content
    lines = []

    # Add each group with entries
    group_config = [
        ("web", "🌐 Web", groups["web"]),
        ("files", "📄 Files", groups["files"]),
        ("mcp", "🔌 MCP Tools", groups["mcp"]),
        ("commands", "⚡ Commands", groups["commands"]),
    ]

    for group_key, group_label, entries in group_config:
        if entries:
            lines.append(f"  [bold]{group_label}[/bold]")
            for ref_num, location, description in entries[:8]:  # Max 8 per group
                # Format: [1] location ──── description
                # Calculate padding to align descriptions
                pad_len = max(1, 30 - len(location))
                padding = "─" * pad_len
                if description:
                    lines.append(f"     [cyan][{ref_num}][/cyan] {location} [dim]{padding}[/dim] {description}")
                else:
                    lines.append(f"     [cyan][{ref_num}][/cyan] {location}")

            if len(entries) > 8:
                lines.append(f"     [dim]... and {len(entries) - 8} more[/dim]")
            lines.append("")

    # Remove trailing empty line
    if lines and lines[-1] == "":
        lines.pop()

    # Add assumptions if any
    if assumptions:
        lines.append("")
        lines.append("  [yellow]💡 Assumptions:[/yellow]")
        for assumption in assumptions[:3]:
            lines.append(f"     • {assumption}")

    # Add key sources footer
    if key_sources:
        lines.append("")
        key_parts = [f"{loc} [cyan][{num}][/cyan]" for num, loc in key_sources[:3]]
        remaining = len(sources) - len(key_sources)
        if remaining > 0:
            key_parts.append(f"[dim]+{remaining} more[/dim]")
        lines.append(f"  [bold]📌 Key:[/bold] {' • '.join(key_parts)}")

    if lines:
        content = "\n".join(lines)
        source_panel = Panel(
            Text.from_markup(content),
            title=f"[bold]📚 Sources ({len(sources)})[/bold]",
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        )
        console.print()
        console.print(source_panel)


def display_sources_summary(console: Console) -> None:
    """
    Display source attribution summary.

    In VERBOSE mode: Shows full panel with grouped sources and key sources footer
    In CONCISE mode: Shows brief one-liner with source count

    ALWAYS shows something - either sources panel or a note that response is from base knowledge.

    Args:
        console: Rich console for output
    """
    from kautilya.tool_executor import get_source_tracker

    tracker = get_source_tracker()

    if not tracker.has_sources():
        # Show a note when no external sources were used
        console.print()
        console.print(Text.from_markup("[dim]📚 Sources: Response based on training knowledge (no external sources used)[/dim]"))
        return

    sources = tracker.get_sources()

    if is_verbose_output():
        # VERBOSE MODE: Use the enhanced panel display
        display_sources_panel(console)
    else:
        # CONCISE MODE: Brief one-liner
        # Count sources by type
        file_count = sum(1 for s in sources if s.source_type.value in ("file_read", "file_search", "config_read"))
        mcp_count = sum(1 for s in sources if s.source_type.value == "mcp_call")
        web_count = sum(1 for s in sources if s.source_type.value in ("web_fetch", "web_search"))
        cmd_count = sum(1 for s in sources if s.source_type.value == "bash_exec")

        parts = []
        if file_count > 0:
            parts.append(f"📄 {file_count} files")
        if mcp_count > 0:
            parts.append(f"🔌 {mcp_count} MCP")
        if web_count > 0:
            parts.append(f"🌐 {web_count} web")
        if cmd_count > 0:
            parts.append(f"⚡ {cmd_count} cmd")

        if parts:
            summary = " • ".join(parts)
            console.print()
            console.print(Text.from_markup(f"[dim]📚 Sources: {summary}  (/output verbose for details)[/dim]"))


def format_sources_inline() -> str:
    """
    Format sources as inline citations for embedding in responses.

    Returns:
        Formatted string of sources, or empty string if no sources.
    """
    from kautilya.tool_executor import get_source_tracker

    tracker = get_source_tracker()
    if not tracker.has_sources():
        return ""

    sources = tracker.get_sources()
    if not sources:
        return ""

    # Format as compact inline list
    lines = ["**Sources:**"]
    for i, source in enumerate(sources[:10], 1):
        lines.append(f"  {i}. {source.to_display()}")

    if len(sources) > 10:
        lines.append(f"  ... and {len(sources) - 10} more")

    return "\n".join(lines)


class IterationProgressBar:
    """Live progress bar for iteration phases."""

    def __init__(self, console: Console):
        """Initialize progress bar."""
        self.console = console
        self.progress = None
        self.task_id = None
        self.live = None

    def start(self, description: str, total: int = 100):
        """Start progress bar."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=True,
        )
        self.task_id = self.progress.add_task(description, total=total)
        self.live = Live(self.progress, console=self.console, refresh_per_second=10)
        self.live.start()

    def update(self, completed: int, description: Optional[str] = None):
        """Update progress bar."""
        if self.progress and self.task_id is not None:
            if description:
                self.progress.update(self.task_id, completed=completed, description=description)
            else:
                self.progress.update(self.task_id, completed=completed)

    def stop(self):
        """Stop progress bar."""
        if self.live:
            self.live.stop()
            self.live = None
            self.progress = None
            self.task_id = None

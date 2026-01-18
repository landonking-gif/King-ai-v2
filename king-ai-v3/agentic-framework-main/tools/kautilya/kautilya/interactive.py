"""
Interactive Mode for Kautilya with LLM Chat Support.

Module: kautilya/interactive.py

Provides both slash command interface and natural language chat
backed by OpenAI LLM for intelligent interaction.
"""

from typing import Dict, List, Optional, Tuple
import sys
import os
import time
import threading
import re
import fnmatch
from pathlib import Path
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from .config import Config
from .commands import (
    init,
    agent,
    skill,
    llm,
    mcp,
    manifest,
    runtime,
)
from .iteration_display import IterationDisplay, IterationPhase
from .iteration_status import IterationStatusExtractor, ResponseBuffer
from .animations import (
    WelcomeScreen,
    ModernSpinner,
    TypingEffect,
    ToolExecutionVisualizer,
    Celebration,
    CommandSuggestion,
    GradientText,
    IterationProgress,
)

console = Console()


class ThinkingSpinner:
    """Animated spinner that shows 'Thinking...' with elapsed time."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, console: Console, style: str = "cyan"):
        """
        Initialize thinking spinner.

        Args:
            console: Rich console for output
            style: Color style for the spinner
        """
        self.console = console
        self.style = style
        self.running = False
        self.start_time = 0.0
        self.thread: Optional[threading.Thread] = None
        self.frame_idx = 0

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time nicely."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"

    def _spin(self) -> None:
        """Animation loop running in background thread."""
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
            self.frame_idx += 1

            # Build the display text
            display = Text()
            display.append(f"{frame} ", style=self.style)
            display.append("Thinking", style=f"bold {self.style}")
            display.append("...", style=self.style)
            display.append(f" ({self._format_time(elapsed)})", style="dim")

            # Clear line and print (using carriage return for animation)
            self.console.print(display, end="\r")

            time.sleep(0.08)  # ~12 fps animation

    def start(self) -> None:
        """Start the thinking animation."""
        self.running = True
        self.start_time = time.time()
        self.frame_idx = 0
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        """
        Stop the thinking animation.

        Returns:
            Total elapsed time in seconds
        """
        self.running = False
        elapsed = time.time() - self.start_time

        if self.thread:
            self.thread.join(timeout=0.2)
            self.thread = None

        # Clear the spinner line
        self.console.print(" " * 50, end="\r")

        return elapsed


class InteractiveMode:
    """Interactive mode handler for Kautilya with LLM chat support."""

    def __init__(self, config_dir: str, config: Config):
        """
        Initialize interactive mode.

        Args:
            config_dir: Configuration directory
            config: Loaded configuration
        """
        self.config_dir = config_dir
        self.config = config
        self.session = PromptSession(history=InMemoryHistory())
        self.llm_client = None
        self.tool_executor = None
        self.llm_enabled = False

        # Agentic executor for intelligent skill-based execution
        self.agentic_executor = None
        self.agentic_enabled = False

        # Track query statistics
        self.last_query_stats = None

        # Attached file context storage
        self.attached_context: Dict[str, str] = {}  # path -> content
        self.attached_stats: Dict[str, dict] = {}  # path -> {size, lines, type}

        # Initialize memory manager
        self.memory_manager = None
        self._init_memory_manager()

        # Define available commands
        self.commands = {
            "/help": self.show_help,
            "/init": self.cmd_init,
            "/agent": self.cmd_agent,
            "/skill": self.cmd_skill,
            "/llm": self.cmd_llm,
            "/mcp": self.cmd_mcp,
            "/websearch": self.cmd_websearch,
            "/manifest": self.cmd_manifest,
            "/run": self.cmd_run,
            "/status": self.cmd_status,
            "/logs": self.cmd_logs,
            "/memory": self.cmd_memory,
            "/chat": self.cmd_toggle_chat,
            "/clear": self.cmd_clear_history,
            "/stats": self.cmd_show_stats,
            "/verbose": self.cmd_toggle_verbose,
            "/display": self.cmd_display,
            "/output": self.cmd_output,
            "/attach": self.cmd_attach,
            "/detach": self.cmd_detach,
            "/context": self.cmd_context,
            "/exit": self.cmd_exit,
            "/quit": self.cmd_exit,
        }

        # Command completer
        self.completer = WordCompleter(
            list(self.commands.keys()), ignore_case=True, sentence=True
        )

        # Try to initialize LLM client
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client and agentic executor."""
        try:
            # Import triggers .env loading in llm_client module
            from .llm_client import KautilyaLLMClient
            from .tool_executor import ToolExecutor

            # Let KautilyaLLMClient handle API key resolution
            # It will load from .env -> llm.yaml -> raise error
            self.llm_client = KautilyaLLMClient()
            self.tool_executor = ToolExecutor(config_dir=self.config_dir)
            self.llm_enabled = True

            # Initialize agentic executor for intelligent skill-based execution
            self._init_agentic_executor()

            # Check web search availability
            try:
                from ddgs import DDGS
                mode_str = "Agentic mode" if self.agentic_enabled else "LLM chat"
                console.print(
                    f"[dim]{mode_str} enabled with web search. Type naturally or use /commands.[/dim]"
                )
            except ImportError:
                mode_str = "Agentic mode" if self.agentic_enabled else "LLM chat"
                console.print(
                    f"[dim]{mode_str} enabled. Type naturally or use /commands.[/dim]"
                )
                console.print(
                    "[yellow]⚠ Web search unavailable. Install with: uv pip install ddgs[/yellow]"
                )
        except ImportError as e:
            console.print(
                f"[dim yellow]LLM chat unavailable: {e}. Using commands only.[/dim yellow]"
            )
        except ValueError as e:
            # API key not found in .env or llm.yaml
            console.print(
                f"[dim yellow]LLM chat disabled: {e}[/dim yellow]"
            )
        except Exception as e:
            console.print(
                f"[dim yellow]LLM init error: {e}. Using commands only.[/dim yellow]"
            )

    def _init_agentic_executor(self) -> None:
        """Initialize the agentic executor for intelligent skill-based execution."""
        # Check if agentic mode is enabled (default: True)
        agentic_mode = os.getenv("KAUTILYA_AGENTIC_MODE", "true").lower()
        if agentic_mode == "false" or agentic_mode == "0":
            console.print("[dim]Agentic mode disabled. Using legacy LLM client.[/dim]")
            return

        try:
            from .agentic_executor import AgenticExecutor

            verbose = os.getenv("KAUTILYA_VERBOSE_MODE", "false").lower() == "true"
            self.agentic_executor = AgenticExecutor(
                config_dir=self.config_dir,
                verbose=verbose,
            )
            self.agentic_enabled = True

            # Show discovered skills count
            skills = self.agentic_executor.get_available_skills()
            if skills:
                console.print(
                    f"[dim]Agentic mode: {len(skills)} skills available "
                    f"(document_qa, deep_research, file_ops, etc.)[/dim]"
                )
        except Exception as e:
            console.print(
                f"[dim yellow]Agentic executor init failed: {e}. Using legacy mode.[/dim yellow]"
            )
            self.agentic_enabled = False

    def _init_memory_manager(self) -> None:
        """Initialize memory manager for persistent context."""
        try:
            from .memory import MemoryManager, MemoryConfig

            config = MemoryConfig(
                storage_backend="sqlite",  # Use SQLite for local persistence
                max_working_messages=50,
                working_memory_ttl=3600,
            )
            self.memory_manager = MemoryManager(config=config)
            console.print(
                f"[dim]Memory: Session {self.memory_manager.session_id[:8]}... "
                f"(User: {self.memory_manager.user_id})[/dim]"
            )
        except Exception as e:
            console.print(
                f"[dim yellow]Memory init warning: {e}. Using session-only memory.[/dim yellow]"
            )
            self.memory_manager = None

    def run(self) -> None:
        """Run interactive mode."""
        self.show_welcome()

        while True:
            try:
                # Show prompt with mode indicator
                prompt_text = "[chat] > " if self.llm_enabled else "> "
                user_input = self.session.prompt(
                    prompt_text, completer=self.completer
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    # Parse slash command
                    parts = user_input.split(maxsplit=1)
                    command = parts[0]
                    args = parts[1] if len(parts) > 1 else ""

                    if command in self.commands:
                        try:
                            self.commands[command](args)
                        except Exception as e:
                            console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    else:
                        console.print(
                            f"[yellow]Unknown command:[/yellow] {command}. "
                            f"Type [bold]/help[/bold] for available commands."
                        )
                else:
                    # Natural language - use agentic executor or LLM
                    if self.agentic_enabled and self.agentic_executor:
                        self._handle_chat_agentic(user_input)
                    elif self.llm_enabled and self.llm_client:
                        self._handle_chat(user_input)
                    else:
                        console.print(
                            "[dim]LLM chat not available. Use slash commands "
                            "(e.g., /init, /agent new) or type /help for options.[/dim]"
                        )

            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit or /quit to leave interactive mode[/yellow]")
                continue
            except EOFError:
                break

    def _handle_chat(self, user_input: str) -> None:
        """Handle natural language chat with LLM."""
        console.print()

        # Process @file mentions in user input (auto-attach referenced files)
        original_input = user_input
        user_input, newly_attached = self._resolve_at_mentions(user_input)
        if newly_attached:
            console.print(f"[dim]📎 Auto-attached {len(newly_attached)} file(s) from @mentions[/dim]")

        # Build context from attached files
        context_prompt = self._build_context_prompt()
        if context_prompt:
            # Prepend context to user message
            user_input = f"{context_prompt}\n\n[USER QUERY]\n{user_input}"

        # Clear source tracker for new query
        from kautilya.tool_executor import clear_source_tracker
        clear_source_tracker()

        # Check if detailed iteration display is enabled
        show_iteration_details = os.getenv("KAUTILYA_SHOW_ITERATION_DETAILS", "true").lower() == "true"
        verbose_mode = os.getenv("KAUTILYA_VERBOSE_MODE", "true").lower() == "true"
        max_iterations = int(os.getenv("KAUTILYA_MAX_ITERATIONS", "5"))

        # Check if animations are enabled
        animations_enabled = os.getenv("KAUTILYA_ANIMATIONS", "true").lower() == "true"

        # Initialize iteration display (use new IterationProgress if animations enabled)
        iteration_display = None
        if show_iteration_details and animations_enabled:
            iteration_display = IterationProgress(console, max_iterations)
        elif show_iteration_details:
            iteration_display = IterationDisplay(console, max_iterations)

        # Start thinking animation (use ModernSpinner if animations enabled)
        if animations_enabled:
            spinner = ModernSpinner(console, "Analyzing your request", "pulse", style="cyan")
        else:
            spinner = ThinkingSpinner(console, style="cyan")

        spinner.start()
        spinner_stopped = False
        live_context: Optional[Live] = None
        live_content_displayed = False  # Track if content was shown via live streaming

        # Track iteration state and statistics
        current_iteration = 0
        in_tool_execution = False
        current_tool = None
        current_tool_start_time = None
        current_tool_args = None
        tools_used = []
        import time
        query_start_time = time.time()

        # Response buffer for dynamic status extraction
        response_buffer = ResponseBuffer(max_size=500)

        try:
            # Stream response
            response_text = ""
            chat_result = None

            # Get the generator
            chat_generator = self.llm_client.chat(
                user_input,
                tool_executor=self.tool_executor,
                stream=True,
            )

            # Iterate and capture return value
            # Note: Using while loop with next() to properly capture StopIteration.value
            try:
                while True:
                    try:
                        chunk = next(chat_generator)
                    except StopIteration as e:
                        chat_result = e.value
                        break

                    # Detect max iterations marker
                    if "[Max iterations reached" in chunk:
                        if iteration_display:
                            console.print()
                            console.print("[yellow]⚠ Max iterations reached, generating final response...[/yellow]")
                        continue  # Don't add to response text

                    # Detect iteration markers
                    if "[Iteration " in chunk and show_iteration_details:
                        if not spinner_stopped:
                            think_time = spinner.stop()
                            spinner_stopped = True

                            # Start the iteration display after spinner stops
                            if isinstance(iteration_display, IterationProgress):
                                iteration_display.start()

                        # Extract iteration number
                        match = re.search(r'\[Iteration (\d+)/(\d+)\]', chunk)
                        if match:
                            iteration_num = int(match.group(1))
                            if iteration_num != current_iteration:
                                current_iteration = iteration_num

                                # Extract dynamic planning info from buffered response
                                planning_info = IterationStatusExtractor.extract_planning_info(
                                    response_buffer.get(),
                                    current_iteration
                                )

                                # Update iteration display
                                if isinstance(iteration_display, IterationProgress):
                                    iteration_display.update_iteration(
                                        current_iteration,
                                        step_info=planning_info,
                                        tools=[]
                                    )
                                elif iteration_display:
                                    # Old display style
                                    if current_iteration > 1:
                                        iteration_display.show_iteration_summary()
                                    console.print()
                                    iteration_display.start_iteration(current_iteration)
                                    iteration_display.show_phase(IterationPhase.PLANNING)

                                # Clear buffer for next iteration
                                response_buffer.clear()

                        continue  # Don't add iteration markers to response text

                    # Detect tool execution
                    if "> Executing: " in chunk:
                        if not spinner_stopped:
                            think_time = spinner.stop()
                            spinner_stopped = True
                            console.print(f"[dim]Thought for {think_time:.1f}s[/dim]\n")

                        # Extract tool name
                        tool_match = re.search(r'> Executing: ([a-z_]+)', chunk)
                        if tool_match:
                            current_tool = tool_match.group(1)
                            tools_used.append(current_tool)
                            current_tool_start_time = time.time()

                            # Try to extract tool args from LLM client history (if available)
                            current_tool_args = None
                            if self.llm_client and hasattr(self.llm_client, 'history'):
                                # Get the last message which should contain tool_calls
                                messages = self.llm_client.history.to_list()
                                if messages:
                                    last_msg = messages[-1]
                                    if isinstance(last_msg, dict) and 'tool_calls' in last_msg:
                                        tool_calls = last_msg['tool_calls']
                                        if tool_calls:
                                            # Find the matching tool call
                                            for tc in tool_calls:
                                                if tc.get('function', {}).get('name') == current_tool:
                                                    import json
                                                    args_str = tc.get('function', {}).get('arguments', '{}')
                                                    try:
                                                        current_tool_args = json.loads(args_str)
                                                    except:
                                                        current_tool_args = None
                                                    break

                            # Extract dynamic tool purpose
                            tool_purpose = IterationStatusExtractor.extract_tool_purpose(
                                current_tool,
                                current_tool_args
                            )

                            # Update iteration display with tool execution info
                            if isinstance(iteration_display, IterationProgress):
                                if current_iteration == 0:
                                    current_iteration = 1
                                    if not iteration_display.live:
                                        iteration_display.start()
                                    iteration_display.update_iteration(1, "", [])

                                # Update with dynamic tool purpose
                                iteration_display.update_info(
                                    step_info=tool_purpose,
                                    tools=[current_tool]
                                )
                            elif animations_enabled:
                                # Show tool execution panel for non-iteration mode
                                ToolExecutionVisualizer.show_execution(
                                    console,
                                    current_tool,
                                    args=current_tool_args or {}
                                )
                            elif iteration_display:
                                # Old display style
                                if current_iteration == 0:
                                    current_iteration = 1
                                    iteration_display.start_iteration(1)
                                iteration_display.show_phase(IterationPhase.EXECUTING)
                                iteration_display.show_tool_execution(current_tool, current_tool_args)

                            in_tool_execution = True

                        continue  # Don't add execution markers to response text

                    # Detect tool completion (when we get content after tool execution)
                    if in_tool_execution and chunk.strip() and not chunk.startswith(">"):
                        in_tool_execution = False

                        # Extract dynamic review info from buffered response
                        review_info = IterationStatusExtractor.extract_review_info(
                            response_buffer.get(),
                            current_tool
                        )

                        # Update iteration display
                        if isinstance(iteration_display, IterationProgress):
                            iteration_display.update_info(
                                step_info=review_info,
                                tools=[]
                            )
                        elif animations_enabled and current_tool and current_tool_start_time:
                            # Show tool result for non-iteration mode
                            duration = time.time() - current_tool_start_time
                            ToolExecutionVisualizer.show_result(
                                console,
                                current_tool,
                                success=True,
                                duration=duration,
                                summary=""  # Summary not available in stream
                            )

                    # Stop spinner on first actual content
                    if chunk.strip() and not spinner_stopped and not in_tool_execution:
                        think_time = spinner.stop()
                        spinner_stopped = True
                        console.print(f"[dim]Thought for {think_time:.1f}s[/dim]\n")

                        # Start Live context for streaming display
                        # NOTE: transient=False ensures content persists after streaming
                        # We'll only use this for responses under 3000 chars to avoid rendering issues
                        live_context = Live(
                            Markdown(""),
                            console=console,
                            refresh_per_second=8,  # Reduced rate for stability
                            transient=False,
                        )
                        live_context.start()

                    response_text += chunk

                    # Update response buffer for status extraction (only actual content, not markers)
                    if chunk.strip() and not chunk.startswith(">") and "[Iteration" not in chunk:
                        response_buffer.add(chunk)

                    # Update live display with streamed content
                    # IMPORTANT: Stop using Live for long responses to avoid rendering issues
                    if live_context and response_text and not in_tool_execution:
                        if len(response_text) > 3000:
                            # Response is getting long - stop live streaming to avoid issues
                            # Final response will be printed directly at the end
                            live_context.stop()
                            live_context = None
                            live_content_displayed = False  # Force re-print at end
                            console.print("[dim]Processing longer response...[/dim]")
                        else:
                            live_context.update(Markdown(response_text))
                            live_content_displayed = True  # Mark that content was shown
            except Exception as inner_e:
                # Handle any errors during iteration
                raise

            # Stop live context and render final output
            if live_context:
                live_context.stop()
                live_context = None

            # Ensure spinner is stopped
            if not spinner_stopped:
                spinner.stop()

            # Calculate query statistics
            query_duration = time.time() - query_start_time

            # Extract token usage from chat result
            usage = None
            if chat_result and isinstance(chat_result, dict):
                usage = chat_result.get("usage")

            self.last_query_stats = {
                "iterations": current_iteration,
                "max_iterations": max_iterations,
                "tools_used": tools_used,
                "total_tools": len(tools_used),
                "duration": query_duration,
                "response_length": len(response_text),
                "usage": usage,  # Actual token usage from API
            }

            # Complete iteration display
            if isinstance(iteration_display, IterationProgress) and current_iteration > 0:
                iteration_display.complete(success=True)
            elif iteration_display and current_iteration > 0:
                # Old display style
                iteration_display.show_phase(IterationPhase.REVIEWING)
                iteration_display.show_iteration_summary()

                # Show completion
                console.print()
                if current_iteration < max_iterations:
                    iteration_display.show_completion(current_iteration, len(tools_used))
                else:
                    iteration_display.show_max_iterations_reached()

            # Get sources for citation injection
            from kautilya.tool_executor import get_source_tracker
            from kautilya.iteration_display import (
                display_followup_questions,
                display_sources_summary,
                inject_inline_citations,
            )

            tracker = get_source_tracker()
            sources = tracker.get_sources()

            # Inject inline citations programmatically (more reliable than LLM)
            if sources and response_text.strip():
                response_text = inject_inline_citations(response_text, sources)

            # Final response display
            # Only print if content was NOT already shown via Live streaming
            # Live with transient=False persists content after stop, so no need to re-print
            if response_text.strip():
                # Only print if live streaming didn't display it
                # (live_content_displayed is False when: no content, or live was stopped early for >3000 chars)
                if not live_content_displayed:
                    console.print()
                    console.print(Markdown(response_text))
                # Note: If content was shown via Live, we don't re-print even for citations
                # to avoid duplication. Citations will appear in future queries.
                console.print()
            else:
                # No response content - agent might have only executed tools
                console.print("[dim]Task completed.[/dim]")
                console.print()

            # ALWAYS display follow-up questions and source attribution
            # These should appear regardless of whether tools were used
            # Order: Follow-up questions first, then sources panel at the bottom

            # Show follow-up questions (contextual based on query and response)
            display_followup_questions(
                console,
                user_query=user_input,
                response_content=response_text,
            )

            # Show sources panel (full panel in verbose mode, brief summary in concise)
            # This now handles empty sources gracefully
            display_sources_summary(console)

            # Remember this interaction in memory
            if self.memory_manager and response_text.strip():
                try:
                    from .memory.models import SourceEntry as MemSourceEntry

                    # Convert sources to memory format
                    mem_sources = []
                    for s in sources:
                        mem_sources.append(MemSourceEntry(
                            source_type=s.source_type.value,
                            location=s.location,
                            description=s.description,
                        ))

                    self.memory_manager.remember(
                        user_query=user_input,
                        agent_response=response_text,
                        tools_used=tools_used,
                        sources=mem_sources,
                        iterations=current_iteration,
                        input_tokens=usage.get("prompt_tokens", 0) if usage else 0,
                        output_tokens=usage.get("completion_tokens", 0) if usage else 0,
                    )
                except Exception as e:
                    # Don't fail the chat if memory fails
                    pass

            # Show query statistics (controlled by KAUTILYA_SHOW_QUERY_STATS env var)
            self._show_query_stats()

        except Exception as e:
            if not spinner_stopped:
                spinner.stop()
            if live_context:
                live_context.stop()
            if isinstance(iteration_display, IterationProgress):
                iteration_display.stop()
            elif iteration_display:
                pass  # Old display doesn't need explicit stop

            # Show beautiful error panel if animations enabled
            if animations_enabled:
                Celebration.error(
                    console,
                    f"Chat error: {str(e)}",
                    details="An error occurred while processing your request."
                )
            else:
                console.print(f"[red]Chat error:[/red] {str(e)}")

    def _handle_chat_agentic(self, user_input: str) -> None:
        """
        Handle natural language chat using the AgentCore-based agentic executor.

        This method uses intelligent skill selection to choose the best skill
        for each task (e.g., document_qa for PDF extraction, deep_research for
        web research).
        """
        console.print()

        # Process @file mentions in user input (auto-attach referenced files)
        original_input = user_input
        user_input, newly_attached = self._resolve_at_mentions(user_input)
        if newly_attached:
            console.print(f"[dim]📎 Auto-attached {len(newly_attached)} file(s) from @mentions[/dim]")

        # Build context from attached files
        context_prompt = self._build_context_prompt()
        if context_prompt:
            # Prepend context to user message
            user_input = f"{context_prompt}\n\n[USER QUERY]\n{user_input}"

        # Check if animations are enabled
        animations_enabled = os.getenv("KAUTILYA_ANIMATIONS", "true").lower() == "true"
        max_iterations = int(os.getenv("KAUTILYA_MAX_ITERATIONS", "5"))

        # Start thinking animation
        if animations_enabled:
            spinner = ModernSpinner(console, "Analyzing with skills", "pulse", style="cyan")
        else:
            spinner = ThinkingSpinner(console, style="cyan")

        spinner.start()
        spinner_stopped = False

        # Track statistics
        import time
        query_start_time = time.time()
        tools_used = []
        current_iteration = 0

        try:
            # Clear source tracker for new query
            from kautilya.tool_executor import clear_source_tracker
            clear_source_tracker()

            response_text = ""
            result = None

            # Show skill selection info
            if self.agentic_executor and self.agentic_executor._agent_core:
                registry = self.agentic_executor._agent_core.capability_registry
                relevant_caps = registry.get_relevant_capabilities(user_input, max_results=3)

                if relevant_caps:
                    skills_info = []
                    for cap in relevant_caps[:3]:
                        skills_info.append(f"{cap.name}")

                    spinner.stop()
                    spinner_stopped = True
                    console.print(f"[dim]🎯 Skills selected: {', '.join(skills_info)}[/dim]")

                    # Show primary skill details
                    primary = relevant_caps[0]
                    if primary.when_to_use:
                        when_to_use_short = primary.when_to_use.split('\n')[0][:80]
                        console.print(f"[dim]   Primary: {primary.name} - {when_to_use_short}...[/dim]")
                    console.print()

                    # Restart spinner for execution - reset spinner_stopped flag
                    spinner_stopped = False
                    if animations_enabled:
                        spinner = ModernSpinner(console, f"Executing {primary.name}", "pulse", style="green")
                    else:
                        spinner = ThinkingSpinner(console, style="green")
                    spinner.start()

            # Execute using agentic executor
            result, progress_messages = self.agentic_executor.execute(
                user_input,
                context={"attached_files": self.attached_context},
                attached_files=self.attached_context,
            )

            # Display progress messages
            for msg in progress_messages:
                if msg.startswith("[Skills selected"):
                    if not spinner_stopped:
                        spinner.stop()
                        spinner_stopped = True
                    console.print(f"[dim]🎯 {msg}[/dim]")
                elif msg.startswith("> Executing:"):
                    if not spinner_stopped:
                        spinner.stop()
                        spinner_stopped = True
                    skill_name = msg.replace("> Executing:", "").strip()
                    tools_used.append(skill_name)
                    console.print(f"[cyan]⚡ Executing: {skill_name}[/cyan]")
                elif msg.startswith("> Queued:"):
                    console.print(f"[dim]{msg}[/dim]")
                elif msg.startswith("[Error]"):
                    if not spinner_stopped:
                        spinner.stop()
                        spinner_stopped = True
                    console.print(f"[red]{msg}[/red]")
                else:
                    console.print(f"[dim]{msg}[/dim]")

            # Get response from result - clean up execution markers
            response_text = result.response or ""
            current_iteration = result.iterations

            # Remove execution markers from response (they're displayed separately)
            import re
            response_text = re.sub(r'\n*> Executing: [a-z_]+\.\.\.\n*', '\n', response_text)
            response_text = re.sub(r'\n*\[Iteration \d+/\d+\]\n*', '\n', response_text)
            response_text = response_text.strip()

            # Always stop the spinner after execution completes
            try:
                think_time = spinner.stop()
                if not spinner_stopped:
                    console.print(f"[dim]Processed in {think_time:.1f}s[/dim]\n")
                spinner_stopped = True
            except Exception:
                pass  # Spinner may already be stopped

            # Calculate query statistics
            query_duration = time.time() - query_start_time

            # Get usage from result if available
            usage = result.usage if hasattr(result, 'usage') and result.usage else None

            # Get model info from environment
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            self.last_query_stats = {
                "iterations": current_iteration or 1,
                "max_iterations": max_iterations,
                "tools_used": tools_used + (result.skills_used if hasattr(result, 'skills_used') else []),
                "total_tools": len(tools_used) + len(result.skills_used if hasattr(result, 'skills_used') else []),
                "duration": query_duration,
                "response_length": len(response_text),
                "usage": usage,
                "model": model_name,
            }

            # Display final response
            if response_text.strip():
                from rich.markdown import Markdown
                console.print(Markdown(response_text))
                console.print()
            else:
                console.print("[dim]Task completed with no text output.[/dim]")
                console.print()

            # Show sources if available
            try:
                from kautilya.tool_executor import get_source_tracker
                from kautilya.iteration_display import display_sources_summary

                display_sources_summary(console)
            except Exception:
                pass

            # Display follow-up questions (contextual based on query and response)
            try:
                from kautilya.iteration_display import display_followup_questions
                display_followup_questions(
                    console,
                    original_input,
                    response_text,
                    tools_used + (result.skills_used if hasattr(result, 'skills_used') else []),
                )
            except Exception:
                pass

            # Remember interaction in memory
            if self.memory_manager and response_text.strip():
                try:
                    self.memory_manager.remember(
                        user_query=original_input,
                        agent_response=response_text,
                        tools_used=tools_used,
                        sources=[],
                        iterations=current_iteration or 1,
                        input_tokens=0,
                        output_tokens=0,
                    )
                except Exception:
                    pass

            # Show query statistics
            self._show_query_stats()

        except Exception as e:
            if not spinner_stopped:
                spinner.stop()

            if animations_enabled:
                Celebration.error(
                    console,
                    f"Agentic execution error: {str(e)}",
                    details="An error occurred while processing with agentic skills."
                )
            else:
                console.print(f"[red]Agentic error:[/red] {str(e)}")

            # Fallback to legacy LLM client
            console.print("[dim]Falling back to legacy LLM mode...[/dim]")
            self._handle_chat(user_input)

    def show_welcome(self) -> None:
        """Show welcome message with beautiful animations."""
        # Check MCP Gateway status
        try:
            from .gateway_manager import get_gateway_manager
            gateway_manager = get_gateway_manager()
            mcp_running = gateway_manager.is_running()
        except Exception:
            mcp_running = False

        # Check if animations are enabled
        animations_enabled = os.getenv("KAUTILYA_ANIMATIONS", "true").lower() == "true"

        if animations_enabled:
            # Use new animated welcome screen
            WelcomeScreen.show(
                console,
                llm_enabled=self.llm_enabled,
                mcp_running=mcp_running,
                version="1.0.0",
                animate=True,
            )
        else:
            # Fallback to simple welcome
            chat_status = "[green]enabled[/green]" if self.llm_enabled else "[yellow]disabled[/yellow]"
            mcp_status = "[green]running[/green]" if mcp_running else "[yellow]stopped[/yellow]"

            welcome_text = f"""
[bold cyan]Kautilya v1.0 - Agentic Framework CLI[/bold cyan]

LLM Chat: {chat_status}
MCP Gateway: {mcp_status}

Type [bold green]/help[/bold green] for commands, or describe your task in natural language.
            """
            console.print(
                Panel(
                    welcome_text.strip(),
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

    def show_help(self, args: str = "") -> None:
        """Show help information."""
        table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Options / Actions", style="dim")

        # Format: (command, description, options)
        commands_help = [
            # Project Setup
            ("/init", "Initialize new agent project", "--name, --provider, --mcp"),
            # Agent Management
            ("/agent new <name>", "Generate new subagent", "--role, --capabilities, --output-type"),
            ("/agent list", "List all agents", ""),
            # Skill Management
            ("/skill new <name>", "Scaffold new skill", "--format, --description, --safety-flags"),
            ("/skill list", "List all available skills", ""),
            ("/skill import <url>", "Import skill from URL/ZIP", "--format hybrid|native|anthropic"),
            ("/skill export <name>", "Export skill for sharing", "--output"),
            ("/skill validate <name>", "Validate skill format", ""),
            # LLM Configuration
            ("/llm config", "Configure LLM provider", "--provider, --model, --api-key-env"),
            ("/llm list", "List available LLM adapters", ""),
            ("/llm test", "Test LLM connection", ""),
            # MCP Server Management
            ("/mcp add <server>", "Add MCP server to manifest", "--scopes, --rate-limit"),
            ("/mcp list", "List registered MCP servers", "--all (include disabled)"),
            ("/mcp import <file>", "Import MCP server from YAML", ""),
            ("/mcp enable <tool_id>", "Enable a disabled MCP server", ""),
            ("/mcp disable <tool_id>", "Disable an enabled MCP server", ""),
            ("/mcp test <server>", "Test MCP server connection", ""),
            # Web Search
            ("/websearch config", "Configure web search providers", "--provider ddg|tavily"),
            ("/websearch list", "List web search providers", ""),
            ("/websearch test", "Test web search", "<query>"),
            # Manifest Management
            ("/manifest new", "Create workflow manifest", "--name, --steps"),
            ("/manifest validate", "Validate manifest schema", "<file>"),
            ("/manifest run <file>", "Execute workflow", "--dry-run"),
            # Runtime Control
            ("/run", "Run project in dev mode", "--detach, --port"),
            ("/status", "Show service status", ""),
            ("/logs [agent]", "Tail service logs", "--follow, --lines"),
            # Session Control
            ("/stats", "Show query statistics", ""),
            ("/verbose", "Toggle verbose mode", "on | off"),
            ("/display [mode]", "Set display mode", "minimal | detailed | toggle"),
            ("/output [mode]", "Set output verbosity", "concise | verbose | toggle"),
            ("/chat", "Toggle LLM chat mode", "on | off"),
            ("/clear", "Clear chat history", ""),
            # File Attachment
            ("/attach <path>", "Attach file or folder to context", "--flat (non-recursive)"),
            ("/detach <name>", "Remove file from context", "--all (remove all)"),
            ("/context", "Show attached files summary", "clear (remove all)"),
            # Help & Exit
            ("/help", "Show this help message", ""),
            ("/exit", "Exit interactive mode", "(or /quit)"),
        ]

        for cmd, desc, options in commands_help:
            table.add_row(cmd, desc, options)

        console.print(table)
        console.print()

        if self.llm_enabled:
            console.print(
                "[bold green]Natural Language Chat[/bold green]\n"
                "You can also type naturally:\n"
                '  [cyan]"Help me create a research agent"[/cyan]\n'
                '  [cyan]"List available LLM providers"[/cyan]\n'
                '  [cyan]"What can agents do?"[/cyan]\n'
            )
            console.print(
                "[bold green]File Attachment (@syntax)[/bold green]\n"
                "Reference files directly in queries:\n"
                '  [cyan]"Review @./src/main.py for security issues"[/cyan]\n'
                '  [cyan]"Explain @config.yaml configuration"[/cyan]\n'
                '  [cyan]"@./src analyze this codebase"[/cyan] (folders auto-scanned)\n'
            )

    def cmd_toggle_chat(self, args: str = "") -> None:
        """Toggle LLM chat mode."""
        if not self.llm_client:
            console.print(
                "[yellow]Cannot enable chat - OPENAI_API_KEY not set.[/yellow]\n"
                "Set it with: export OPENAI_API_KEY=your-key"
            )
            return

        self.llm_enabled = not self.llm_enabled
        status = "[green]enabled[/green]" if self.llm_enabled else "[red]disabled[/red]"
        console.print(f"LLM chat mode: {status}")

    def cmd_clear_history(self, args: str = "") -> None:
        """Clear chat history."""
        if self.llm_client:
            self.llm_client.clear_history()
            self.last_query_stats = None
            console.print("[green]Chat history cleared.[/green]")
        else:
            console.print("[yellow]No chat history to clear.[/yellow]")

    def _show_query_stats(self) -> None:
        """Show detailed statistics for the last query in 3-column format."""
        if not self.last_query_stats:
            return

        # Check if stats display is enabled
        show_stats = os.getenv("KAUTILYA_SHOW_QUERY_STATS", "true").lower()
        if show_stats == "false" or show_stats == "never":
            return

        stats = self.last_query_stats

        # Create detailed stats table with 3 columns
        table = Table(
            title="Last Query Statistics",
            show_header=True,
            header_style="bold magenta",
            box=ROUNDED,
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")

        # Iteration info
        iter_pct = (stats['iterations'] / stats['max_iterations']) * 100 if stats['max_iterations'] > 0 else 0
        table.add_row(
            "Iterations",
            f"{stats['iterations']}/{stats['max_iterations']}",
            f"{iter_pct:.0f}% of max"
        )

        # Tool usage
        table.add_row(
            "Tools Executed",
            str(stats['total_tools']),
            ", ".join(stats['tools_used']) if stats['tools_used'] else "None"
        )

        # Timing
        duration = stats['duration']
        if duration > 60:
            duration_detail = f"{duration/60:.1f} minutes"
        elif duration > 10:
            duration_detail = "Moderate"
        else:
            duration_detail = "Fast"
        table.add_row(
            "Query Duration",
            f"{duration:.2f}s",
            duration_detail
        )

        # Model info - get from stats or environment
        model_name = stats.get('model') or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Cost rates per million tokens by model
        MODEL_COSTS = {
            "gpt-4o": (2.50, 10.00),  # (input, output)
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-4": (30.00, 60.00),
            "gpt-3.5-turbo": (0.50, 1.50),
            "claude-sonnet-4": (3.00, 15.00),
            "claude-opus-4": (15.00, 75.00),
            "claude-haiku": (0.25, 1.25),
        }

        # Get cost rates for model (default to gpt-4o-mini rates)
        input_rate, output_rate = MODEL_COSTS.get(model_name, MODEL_COSTS.get("gpt-4o-mini", (0.15, 0.60)))

        # Token usage - show actual if available
        usage = stats.get('usage')
        if usage:
            table.add_row(
                "Input Tokens",
                f"{usage['prompt_tokens']:,}",
                "Tokens sent to model"
            )
            table.add_row(
                "Output Tokens",
                f"{usage['completion_tokens']:,}",
                "Tokens generated by model"
            )
            # Calculate cost
            input_cost = usage['prompt_tokens'] / 1_000_000 * input_rate
            output_cost = usage['completion_tokens'] / 1_000_000 * output_rate
            total_cost = input_cost + output_cost
            table.add_row(
                "Total Tokens",
                f"{usage['total_tokens']:,}",
                f"${total_cost:.4f} cost estimate"
            )
            table.add_row(
                "Model",
                model_name,
                f"${input_rate}/M in, ${output_rate}/M out"
            )
        else:
            # Fallback to character estimate
            est_tokens = stats['response_length'] // 4
            table.add_row(
                "Response Size",
                f"{stats['response_length']:,} chars",
                f"~{est_tokens:,} tokens estimate"
            )
            table.add_row(
                "Model",
                model_name,
                "Token tracking not available"
            )

        console.print()
        console.print(table)
        console.print()

    def cmd_show_stats(self, args: str = "") -> None:
        """Show statistics for the last query."""
        if not self.last_query_stats:
            console.print("[yellow]No query statistics available.[/yellow]")
            console.print("[dim]Run a query first, then use /stats to see details.[/dim]")
            return

        stats = self.last_query_stats

        # Create detailed stats table
        table = Table(title="Last Query Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")

        # Iteration info
        iter_pct = (stats['iterations'] / stats['max_iterations']) * 100
        table.add_row(
            "Iterations",
            f"{stats['iterations']}/{stats['max_iterations']}",
            f"{iter_pct:.0f}% of max"
        )

        # Tool usage
        table.add_row(
            "Tools Executed",
            str(stats['total_tools']),
            ", ".join(stats['tools_used']) if stats['tools_used'] else "None"
        )

        # Timing
        table.add_row(
            "Query Duration",
            f"{stats['duration']:.2f}s",
            f"{stats['duration']/60:.1f} minutes" if stats['duration'] > 60 else "Fast"
        )

        # Token usage - show actual if available
        usage = stats.get('usage')
        if usage:
            table.add_row(
                "Input Tokens",
                f"{usage['prompt_tokens']:,}",
                "Tokens sent to model"
            )
            table.add_row(
                "Output Tokens",
                f"{usage['completion_tokens']:,}",
                "Tokens generated by model"
            )
            table.add_row(
                "Total Tokens",
                f"{usage['total_tokens']:,}",
                f"${usage['total_tokens']/1_000_000*0.60:.4f} cost estimate (gpt-4o-mini)"
            )
        else:
            # Fallback to character estimate
            table.add_row(
                "Response Size",
                f"{stats['response_length']} chars",
                f"~{stats['response_length']//4} tokens estimate"
            )

        # Efficiency
        if stats['total_tools'] > 0:
            tools_per_iter = stats['total_tools'] / stats['iterations']
            table.add_row(
                "Efficiency",
                f"{tools_per_iter:.1f} tools/iteration",
                "Higher is more efficient"
            )

        console.print()
        console.print(table)
        console.print()

    def cmd_toggle_verbose(self, args: str = "") -> None:
        """Toggle verbose mode for iteration details."""
        current = os.getenv("KAUTILYA_VERBOSE_MODE", "true").lower() == "true"
        new_value = "false" if current else "true"

        # Update environment variable for current session
        os.environ["KAUTILYA_VERBOSE_MODE"] = new_value

        status = "[green]enabled[/green]" if new_value == "true" else "[red]disabled[/red]"
        console.print(f"Verbose mode: {status}")
        console.print(f"[dim]This setting applies to the current session only.[/dim]")
        console.print(f"[dim]To make permanent, edit .env: KAUTILYA_VERBOSE_MODE={new_value}[/dim]")

    def cmd_display(self, args: str = "") -> None:
        """Handle /display command to toggle iteration display mode."""
        from .iteration_display import (
            DisplayMode,
            get_display_mode,
            set_display_mode,
            toggle_display_mode,
        )

        args = args.strip().lower()

        if not args:
            # Show current mode
            current = get_display_mode()
            console.print(f"Display mode: [bold cyan]{current.value}[/bold cyan]")
            console.print("[dim]Usage: /display [minimal|detailed|toggle][/dim]")
            return

        if args == "toggle":
            new_mode = toggle_display_mode()
            console.print(f"✓ Display mode: [bold cyan]{new_mode.value}[/bold cyan]")
        elif args in ("minimal", "detailed"):
            set_display_mode(DisplayMode(args))
            console.print(f"✓ Display mode: [bold cyan]{args}[/bold cyan]")

            if args == "minimal":
                console.print("[dim]Compact one-liner with key metrics[/dim]")
            else:
                console.print("[dim]Rich panels with full progress information[/dim]")
        else:
            console.print("[yellow]Usage:[/yellow] /display [minimal|detailed|toggle]")
            console.print("[dim]  minimal  - Compact one-liner format[/dim]")
            console.print("[dim]  detailed - Rich panel format (default)[/dim]")
            console.print("[dim]  toggle   - Switch between modes[/dim]")

    def cmd_output(self, args: str = "") -> None:
        """Handle /output command to toggle output verbosity mode."""
        from .iteration_display import (
            OutputMode,
            get_output_mode,
            set_output_mode,
            toggle_output_mode,
        )

        args = args.strip().lower()

        if not args:
            # Show current mode
            current = get_output_mode()
            console.print(f"Output mode: [bold cyan]{current.value}[/bold cyan]")
            console.print("[dim]Usage: /output [concise|verbose|toggle][/dim]")
            return

        if args == "toggle":
            new_mode = toggle_output_mode()
            console.print(f"✓ Output mode: [bold cyan]{new_mode.value}[/bold cyan]")
        elif args in ("concise", "verbose"):
            set_output_mode(OutputMode(args))
            console.print(f"✓ Output mode: [bold cyan]{args}[/bold cyan]")

            if args == "concise":
                console.print("[dim]Direct answers only - no action summaries[/dim]")
            else:
                console.print("[dim]Includes action summaries and reasoning[/dim]")
        else:
            console.print("[yellow]Usage:[/yellow] /output [concise|verbose|toggle]")
            console.print("[dim]  concise - Direct answers only, brief source summary[/dim]")
            console.print("[dim]  verbose - Full source details (default for transparency)[/dim]")
            console.print("[dim]  toggle  - Switch between modes[/dim]")

    # =========================================================================
    # File Attachment Commands
    # =========================================================================

    # File extensions to include by default (code and text files)
    ATTACHABLE_EXTENSIONS = {
        # Code files
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".rb",
        ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
        ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
        # Web files
        ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
        # Config files
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
        ".xml", ".properties",
        # Documentation
        ".md", ".rst", ".txt", ".adoc",
        # Data files
        ".csv", ".sql",
        # Other
        ".dockerfile", ".gitignore", ".editorconfig", ".prettierrc",
    }

    # Files/folders to always skip
    SKIP_PATTERNS = {
        "__pycache__", ".git", ".svn", ".hg", "node_modules", ".venv", "venv",
        ".env", ".DS_Store", "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",
        "*.exe", "*.bin", "*.o", "*.a", "*.class", "*.jar", "*.war",
        "*.zip", "*.tar", "*.gz", "*.rar", "*.7z",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg", "*.webp",
        "*.mp3", "*.mp4", "*.wav", "*.avi", "*.mov",
        "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
        "*.egg-info", ".tox", ".coverage", "htmlcov",
    }

    # Maximum limits
    MAX_FILE_SIZE = 1024 * 1024  # 1MB per file
    MAX_TOTAL_SIZE = 10 * 1024 * 1024  # 10MB total
    MAX_FILES = 100  # Maximum number of files

    def _should_skip_path(self, path: Path) -> bool:
        """Check if a path should be skipped based on patterns."""
        name = path.name
        for pattern in self.SKIP_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                return True
        return False

    def _is_attachable_file(self, path: Path) -> bool:
        """Check if a file can be attached based on extension."""
        # Allow files without extension if they look like config files
        if path.suffix == "":
            config_names = {"Dockerfile", "Makefile", "Procfile", "Gemfile", "Rakefile"}
            return path.name in config_names
        return path.suffix.lower() in self.ATTACHABLE_EXTENSIONS

    def _read_file_safe(self, path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Safely read a file, returning (content, error)."""
        try:
            if path.stat().st_size > self.MAX_FILE_SIZE:
                return None, f"File too large (>{self.MAX_FILE_SIZE // 1024}KB)"

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return content, None
        except Exception as e:
            return None, str(e)

    def _get_file_type(self, path: Path) -> str:
        """Get a human-readable file type."""
        ext = path.suffix.lower()
        type_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "React JSX", ".tsx": "React TSX", ".java": "Java",
            ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
            ".c": "C", ".cpp": "C++", ".cs": "C#", ".swift": "Swift",
            ".kt": "Kotlin", ".scala": "Scala", ".sh": "Shell",
            ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
            ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
            ".toml": "TOML", ".xml": "XML", ".md": "Markdown",
            ".sql": "SQL", ".csv": "CSV", ".txt": "Text",
        }
        return type_map.get(ext, ext.upper()[1:] if ext else "Unknown")

    def _attach_file(self, path: Path) -> Tuple[bool, str]:
        """Attach a single file to context."""
        if not path.exists():
            return False, f"File not found: {path}"

        if not path.is_file():
            return False, f"Not a file: {path}"

        if self._should_skip_path(path):
            return False, f"Skipped (excluded pattern): {path.name}"

        if not self._is_attachable_file(path):
            return False, f"Unsupported file type: {path.suffix or 'no extension'}"

        # Check total size limit
        current_total = sum(len(c) for c in self.attached_context.values())
        if current_total >= self.MAX_TOTAL_SIZE:
            return False, "Total attachment size limit reached (10MB)"

        # Check file count limit
        if len(self.attached_context) >= self.MAX_FILES:
            return False, f"Maximum file limit reached ({self.MAX_FILES} files)"

        # Read file
        content, error = self._read_file_safe(path)
        if error:
            return False, error

        # Store with absolute path as key
        abs_path = str(path.absolute())
        self.attached_context[abs_path] = content
        self.attached_stats[abs_path] = {
            "size": len(content),
            "lines": content.count("\n") + 1,
            "type": self._get_file_type(path),
            "name": path.name,
        }

        return True, f"Attached: {path.name} ({len(content):,} bytes)"

    def _attach_folder(self, folder: Path, recursive: bool = True) -> List[Tuple[str, bool, str]]:
        """Attach all eligible files from a folder."""
        results = []

        if not folder.exists():
            return [(str(folder), False, "Folder not found")]

        if not folder.is_dir():
            return [(str(folder), False, "Not a folder")]

        # Collect files
        if recursive:
            files = [f for f in folder.rglob("*") if f.is_file()]
        else:
            files = [f for f in folder.iterdir() if f.is_file()]

        # Sort by path for consistent ordering
        files.sort()

        attached_count = 0
        for file_path in files:
            # Check if any parent folder should be skipped
            should_skip = False
            for parent in file_path.relative_to(folder).parents:
                if self._should_skip_path(Path(parent.name)):
                    should_skip = True
                    break

            if should_skip:
                continue

            success, message = self._attach_file(file_path)
            if success:
                attached_count += 1
                results.append((str(file_path), True, message))

                # Stop if we hit limits
                if len(self.attached_context) >= self.MAX_FILES:
                    results.append(("", False, f"Stopped: max file limit ({self.MAX_FILES})"))
                    break

                current_total = sum(len(c) for c in self.attached_context.values())
                if current_total >= self.MAX_TOTAL_SIZE:
                    results.append(("", False, "Stopped: max size limit (10MB)"))
                    break

        return results

    def _resolve_at_mentions(self, text: str) -> Tuple[str, List[str]]:
        """
        Resolve @file mentions in text.

        Returns:
            Tuple of (cleaned_text, list_of_attached_file_paths)
        """
        # Pattern to match @path (handles spaces with quotes)
        # Examples: @file.py, @./src/main.py, @"path with spaces/file.py"
        pattern = r'@"([^"]+)"|@(\S+)'

        attached_paths = []
        mentions_found = []

        for match in re.finditer(pattern, text):
            # Get the path (either quoted or unquoted)
            file_path = match.group(1) or match.group(2)
            mentions_found.append((match.group(0), file_path))

        # Process each mention
        for mention, file_path in mentions_found:
            path = Path(file_path).expanduser()

            # Try to resolve relative to current directory
            if not path.is_absolute():
                path = Path.cwd() / path

            if path.exists():
                if path.is_file():
                    success, _ = self._attach_file(path)
                    if success:
                        attached_paths.append(str(path.absolute()))
                elif path.is_dir():
                    results = self._attach_folder(path)
                    for file_path_str, success, _ in results:
                        if success:
                            attached_paths.append(file_path_str)

        # Remove @ mentions from text (they're now in context)
        cleaned_text = re.sub(pattern, "", text).strip()
        # Clean up multiple spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        return cleaned_text, attached_paths

    def _build_context_prompt(self) -> str:
        """Build the context prompt from attached files."""
        if not self.attached_context:
            return ""

        lines = ["[ATTACHED FILES CONTEXT]"]
        lines.append(f"The following {len(self.attached_context)} file(s) have been attached for reference:\n")

        for path, content in self.attached_context.items():
            stats = self.attached_stats.get(path, {})
            filename = stats.get("name", Path(path).name)
            file_type = stats.get("type", "Unknown")
            line_count = stats.get("lines", 0)

            lines.append(f"--- FILE: {filename} ({file_type}, {line_count} lines) ---")
            lines.append(f"Path: {path}")
            lines.append("```")
            # Truncate very large files in context
            if len(content) > 50000:
                lines.append(content[:50000])
                lines.append(f"\n... [truncated, {len(content) - 50000:,} more bytes] ...")
            else:
                lines.append(content)
            lines.append("```")
            lines.append("")

        lines.append("[END ATTACHED FILES]\n")
        return "\n".join(lines)

    def cmd_attach(self, args: str) -> None:
        """
        Handle /attach command to attach files or folders to context.

        Usage:
            /attach <file_path>       - Attach a single file
            /attach <folder_path>     - Attach all files in folder (recursive)
            /attach <folder> --flat   - Attach files in folder (non-recursive)
        """
        if not args.strip():
            console.print("[yellow]Usage:[/yellow] /attach <file_or_folder_path>")
            console.print("[dim]  /attach ./src            - Attach folder recursively[/dim]")
            console.print("[dim]  /attach ./config.yaml   - Attach single file[/dim]")
            console.print("[dim]  /attach ./src --flat    - Attach folder non-recursively[/dim]")
            return

        # Parse arguments
        parts = args.strip().split()
        path_str = parts[0]
        recursive = "--flat" not in args

        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            console.print(f"[red]Path not found:[/red] {path_str}")
            return

        console.print(f"\n[bold cyan]Attaching:[/bold cyan] {path}")

        if path.is_file():
            success, message = self._attach_file(path)
            if success:
                console.print(f"[green]✓[/green] {message}")
            else:
                console.print(f"[red]✗[/red] {message}")
        else:
            mode = "recursive" if recursive else "flat"
            console.print(f"[dim]Scanning folder ({mode})...[/dim]")

            results = self._attach_folder(path, recursive=recursive)

            success_count = sum(1 for _, success, _ in results if success)
            fail_count = len(results) - success_count

            if success_count > 0:
                console.print(f"\n[green]✓ Attached {success_count} file(s)[/green]")

            if fail_count > 0 and os.getenv("KAUTILYA_VERBOSE_MODE", "false").lower() == "true":
                console.print(f"[dim]Skipped {fail_count} file(s) (use verbose mode for details)[/dim]")

        # Show summary
        self._show_context_summary()

    def cmd_detach(self, args: str) -> None:
        """
        Handle /detach command to remove files from context.

        Usage:
            /detach <filename>  - Detach file matching name
            /detach --all       - Detach all files
        """
        if not args.strip():
            console.print("[yellow]Usage:[/yellow] /detach <filename> or /detach --all")
            return

        if args.strip() == "--all":
            count = len(self.attached_context)
            self.attached_context.clear()
            self.attached_stats.clear()
            console.print(f"[green]✓ Detached all {count} file(s)[/green]")
            return

        # Find and remove matching files
        search = args.strip().lower()
        to_remove = []

        for path in self.attached_context.keys():
            filename = Path(path).name.lower()
            if search in filename or search in path.lower():
                to_remove.append(path)

        if not to_remove:
            console.print(f"[yellow]No attached files matching:[/yellow] {args}")
            return

        for path in to_remove:
            del self.attached_context[path]
            if path in self.attached_stats:
                del self.attached_stats[path]
            console.print(f"[green]✓[/green] Detached: {Path(path).name}")

        console.print(f"\n[dim]Remaining: {len(self.attached_context)} file(s)[/dim]")

    def cmd_context(self, args: str) -> None:
        """
        Handle /context command to show attached files context.

        Usage:
            /context         - Show attached files summary
            /context clear   - Clear all attached files
        """
        if args.strip() == "clear":
            count = len(self.attached_context)
            self.attached_context.clear()
            self.attached_stats.clear()
            console.print(f"[green]✓ Cleared {count} attached file(s)[/green]")
            return

        if not self.attached_context:
            console.print("[yellow]No files attached.[/yellow]")
            console.print("[dim]Use /attach <path> to attach files or folders.[/dim]")
            console.print("[dim]Or use @filename in your query to auto-attach.[/dim]")
            return

        self._show_context_summary(detailed=True)

    def _show_context_summary(self, detailed: bool = False) -> None:
        """Show summary of attached files."""
        if not self.attached_context:
            return

        total_size = sum(len(c) for c in self.attached_context.values())
        total_lines = sum(s.get("lines", 0) for s in self.attached_stats.values())

        console.print(f"\n[bold cyan]📎 Attached Context[/bold cyan]")
        console.print(f"[dim]Files: {len(self.attached_context)} | "
                     f"Size: {total_size:,} bytes | "
                     f"Lines: {total_lines:,}[/dim]")

        if detailed:
            table = Table(show_header=True, header_style="bold", box=ROUNDED)
            table.add_column("File", style="cyan")
            table.add_column("Type", style="dim")
            table.add_column("Lines", justify="right")
            table.add_column("Size", justify="right")

            for path, stats in sorted(self.attached_stats.items(), key=lambda x: x[1].get("name", "")):
                table.add_row(
                    stats.get("name", Path(path).name),
                    stats.get("type", "?"),
                    str(stats.get("lines", 0)),
                    f"{stats.get('size', 0):,}",
                )

            console.print(table)

    def cmd_init(self, args: str) -> None:
        """Handle /init command."""
        from .commands.init import initialize_project

        initialize_project(self.config_dir)

    def cmd_agent(self, args: str) -> None:
        """Handle /agent command."""
        parts = args.split(maxsplit=1)
        if not parts or parts[0] != "new":
            console.print("[yellow]Usage:[/yellow] /agent new <name>")
            return

        agent_name = parts[1] if len(parts) > 1 else ""
        if not agent_name:
            console.print("[yellow]Usage:[/yellow] /agent new <name>")
            return

        from .commands.agent import create_agent

        create_agent(agent_name, self.config_dir, interactive=True)

    def cmd_skill(self, args: str) -> None:
        """Handle /skill command."""
        parts = args.split(maxsplit=1)
        if not parts:
            console.print(
                "[yellow]Usage:[/yellow] /skill [new|import|export|convert|list|validate|package]"
            )
            return

        subcommand = parts[0]
        rest_args = parts[1] if len(parts) > 1 else ""

        if subcommand == "new":
            if not rest_args:
                console.print("[yellow]Usage:[/yellow] /skill new <name>")
                return
            from .commands.skill import create_skill

            create_skill(rest_args, self.config_dir, interactive=True)

        elif subcommand == "import":
            if not rest_args:
                console.print("[yellow]Usage:[/yellow] /skill import <url|path>")
                return
            from .commands.skill import skill_import_cmd
            import click

            ctx = click.Context(click.Command("import"))
            skill_import_cmd.invoke(ctx, source=rest_args, import_format="hybrid", safety_flags=(), requires_approval=False)

        elif subcommand == "export":
            if not rest_args:
                console.print("[yellow]Usage:[/yellow] /skill export <name>")
                return
            from .commands.skill import skill_export_cmd
            import click

            ctx = click.Context(click.Command("export"))
            skill_export_cmd.invoke(ctx, name=rest_args, export_format="anthropic", output=None)

        elif subcommand == "convert":
            if not rest_args:
                console.print("[yellow]Usage:[/yellow] /skill convert <name>")
                return
            from .commands.skill import skill_convert_cmd
            import questionary

            to_format = questionary.select(
                "Convert to:",
                choices=["anthropic", "native"],
            ).ask()

            import click

            ctx = click.Context(click.Command("convert"))
            skill_convert_cmd.invoke(ctx, name=rest_args, to_format=to_format, include_handler=True)

        elif subcommand == "list":
            from .commands.skill import skill_list_cmd
            import click

            ctx = click.Context(click.Command("list"))
            skill_list_cmd.invoke(ctx)

        elif subcommand == "validate":
            if not rest_args:
                console.print("[yellow]Usage:[/yellow] /skill validate <name>")
                return
            from .commands.skill import skill_validate_cmd
            import click

            ctx = click.Context(click.Command("validate"))
            skill_validate_cmd.invoke(ctx, name=rest_args)

        elif subcommand == "package":
            if not rest_args:
                console.print("[yellow]Usage:[/yellow] /skill package <name>")
                return
            from .commands.skill import skill_package_cmd
            import click

            ctx = click.Context(click.Command("package"))
            skill_package_cmd.invoke(ctx, name=rest_args, output=None)

        else:
            console.print(
                f"[yellow]Unknown subcommand:[/yellow] {subcommand}\n"
                "[dim]Available: new, import, export, convert, list, validate, package[/dim]"
            )

    def cmd_llm(self, args: str) -> None:
        """Handle /llm command."""
        subcommand = args.split()[0] if args else ""

        if subcommand == "config":
            from .commands.llm import configure_llm

            configure_llm(self.config_dir, interactive=True)
        elif subcommand == "list":
            from .commands.llm import list_llm_providers

            list_llm_providers()
        elif subcommand == "test":
            from .commands.llm import test_llm_connection

            test_llm_connection(self.config_dir)
        else:
            console.print("[yellow]Usage:[/yellow] /llm [config|list|test]")

    def cmd_mcp(self, args: str) -> None:
        """Handle /mcp command."""
        parts = args.split(maxsplit=1)
        if not parts:
            console.print("[yellow]Usage:[/yellow] /mcp [add|list|test|enable|disable] [server]")
            return

        subcommand = parts[0]
        server_name = parts[1] if len(parts) > 1 else ""

        if subcommand == "add":
            from .commands.mcp import add_mcp_server

            add_mcp_server(server_name, self.config_dir, interactive=True)
        elif subcommand == "list":
            from .commands.mcp import list_mcp_servers

            list_mcp_servers()
        elif subcommand == "test":
            from .commands.mcp import test_mcp_server

            test_mcp_server(server_name, self.config_dir)
        elif subcommand == "enable":
            if not server_name:
                console.print("[yellow]Usage:[/yellow] /mcp enable <tool_id>")
                return
            from .commands.mcp import enable_disable_server

            enable_disable_server(server_name, enable=True)
        elif subcommand == "disable":
            if not server_name:
                console.print("[yellow]Usage:[/yellow] /mcp disable <tool_id>")
                return
            from .commands.mcp import enable_disable_server

            enable_disable_server(server_name, enable=False)
        else:
            console.print("[yellow]Usage:[/yellow] /mcp [add|list|test|enable|disable] [server]")

    def cmd_websearch(self, args: str) -> None:
        """Handle /websearch command for configuring web search providers."""
        parts = args.split(maxsplit=1)
        if not parts:
            console.print("[yellow]Usage:[/yellow] /websearch [config|list|test]")
            return

        subcommand = parts[0]

        if subcommand == "config":
            self._configure_websearch_interactive()
        elif subcommand == "list":
            self._list_websearch_providers()
        elif subcommand == "test":
            query = parts[1] if len(parts) > 1 else "latest AI news 2025"
            self._test_websearch(query)
        else:
            console.print("[yellow]Usage:[/yellow] /websearch [config|list|test] [query]")

    def _configure_websearch_interactive(self) -> None:
        """Interactive web search configuration."""
        if not self.tool_executor:
            console.print("[red]Tool executor not initialized[/red]")
            return

        console.print("\n[bold cyan]Web Search Configuration[/bold cyan]\n")

        # Show current configuration
        config = self.tool_executor._load_websearch_config()
        console.print(f"Current default provider: [cyan]{config['default_provider']}[/cyan]")
        console.print(f"Tavily configured: [cyan]{bool(config.get('tavily_api_key'))}[/cyan]\n")

        # Ask which provider to configure
        console.print("[bold]Available providers:[/bold]")
        console.print("  1. [green]DuckDuckGo[/green] (free, no API key)")
        console.print("  2. [yellow]Tavily[/yellow] (requires API key, more accurate)\n")

        choice = console.input("Select provider to configure [1-2] or 'q' to quit: ")

        if choice == "q":
            return

        if choice == "1":
            # Configure DuckDuckGo as default
            result = self.tool_executor._exec_configure_websearch(
                provider="duckduckgo",
                set_as_default=True,
            )
            if result["success"]:
                console.print("[green]✓[/green] DuckDuckGo set as default provider")
            else:
                console.print(f"[red]Error:[/red] {result.get('error')}")

        elif choice == "2":
            # Configure Tavily
            console.print("\n[bold]Tavily Configuration[/bold]")
            console.print("Get your API key from: https://tavily.com\n")

            api_key = console.input("Enter Tavily API key (or press Enter to skip): ")

            if api_key:
                set_default = console.input("Set Tavily as default provider? [y/N]: ")
                result = self.tool_executor._exec_configure_websearch(
                    provider="tavily",
                    tavily_api_key=api_key,
                    set_as_default=set_default.lower() == "y",
                )
                if result["success"]:
                    console.print("[green]✓[/green] Tavily configured successfully")
                    if set_default.lower() == "y":
                        console.print("[green]✓[/green] Set as default provider")
                else:
                    console.print(f"[red]Error:[/red] {result.get('error')}")
            else:
                console.print("[yellow]Skipped Tavily configuration[/yellow]")
        else:
            console.print("[red]Invalid choice[/red]")

    def _list_websearch_providers(self) -> None:
        """List available web search providers."""
        if not self.tool_executor:
            console.print("[red]Tool executor not initialized[/red]")
            return

        result = self.tool_executor._exec_list_websearch_providers()

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result.get('error')}")
            return

        console.print("\n[bold cyan]Web Search Providers[/bold cyan]\n")

        table = Table(show_header=True, box=None)
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Cost", style="yellow")
        table.add_column("Description")

        for provider in result["providers"]:
            status = "✓ Configured" if provider["configured"] else "Not configured"
            if provider["is_default"]:
                status = f"[bold green]✓ Default[/bold green]"

            table.add_row(
                provider["name"],
                status,
                provider["cost"],
                provider["description"],
            )

        console.print(table)
        console.print(f"\n[dim]Use /websearch config to configure providers[/dim]")

    def _test_websearch(self, query: str) -> None:
        """Test web search with a query."""
        if not self.tool_executor:
            console.print("[red]Tool executor not initialized[/red]")
            return

        console.print(f"\n[bold cyan]Testing web search:[/bold cyan] {query}\n")

        result = self.tool_executor._exec_web_search(query=query, max_results=3)

        if not result["success"]:
            console.print(f"[red]Error:[/red] {result.get('error')}")
            return

        console.print(f"[green]✓[/green] Provider: {result['provider']}")
        console.print(f"[green]✓[/green] Found {result['result_count']} results\n")

        for i, item in enumerate(result["results"][:3], 1):
            console.print(f"[bold]{i}. {item['title']}[/bold]")
            console.print(f"   [dim]{item['url']}[/dim]")
            console.print(f"   {item['snippet'][:150]}...\n")

        if result.get("answer"):
            console.print(f"[bold cyan]Direct answer:[/bold cyan] {result['answer']}\n")

    def cmd_manifest(self, args: str) -> None:
        """Handle /manifest command."""
        subcommand = args.split()[0] if args else ""

        if subcommand == "new":
            from .commands.manifest import create_manifest

            create_manifest(self.config_dir, interactive=True)
        elif subcommand == "validate":
            manifest_file = args.split()[1] if len(args.split()) > 1 else ""
            from .commands.manifest import validate_manifest

            validate_manifest(manifest_file, self.config_dir)
        elif subcommand == "run":
            manifest_file = args.split()[1] if len(args.split()) > 1 else ""
            from .commands.manifest import run_manifest

            run_manifest(manifest_file, self.config_dir)
        else:
            console.print("[yellow]Usage:[/yellow] /manifest [new|validate|run] [file]")

    def cmd_run(self, args: str) -> None:
        """Handle /run command."""
        from .commands.runtime import run_project

        run_project(self.config_dir)

    def cmd_status(self, args: str) -> None:
        """Handle /status command."""
        from .commands.runtime import show_status

        show_status(self.config_dir)

    def cmd_logs(self, args: str) -> None:
        """Handle /logs command."""
        agent_name = args.strip() if args else None
        from .commands.runtime import tail_logs

        tail_logs(agent_name, self.config_dir)

    def cmd_memory(self, args: str) -> None:
        """Handle /memory command for memory management."""
        if not self.memory_manager:
            console.print("[yellow]Memory manager not initialized.[/yellow]")
            return

        parts = args.strip().split(maxsplit=1)
        subcommand = parts[0] if parts else ""
        rest_args = parts[1] if len(parts) > 1 else ""

        if subcommand == "status" or not subcommand:
            self._memory_status()
        elif subcommand == "sessions":
            self._memory_sessions()
        elif subcommand == "search":
            self._memory_search(rest_args)
        elif subcommand == "profile":
            self._memory_profile()
        elif subcommand == "clear":
            self._memory_clear()
        elif subcommand == "resume":
            self._memory_resume(rest_args)
        else:
            console.print("[yellow]Usage:[/yellow] /memory [status|sessions|search|profile|clear|resume]")
            console.print("[dim]  status   - Show memory statistics[/dim]")
            console.print("[dim]  sessions - List recent sessions[/dim]")
            console.print("[dim]  search   - Search past conversations[/dim]")
            console.print("[dim]  profile  - Show user profile[/dim]")
            console.print("[dim]  clear    - Clear working memory[/dim]")
            console.print("[dim]  resume   - Resume a previous session[/dim]")

    def _memory_status(self) -> None:
        """Show memory statistics."""
        stats = self.memory_manager.get_memory_stats()

        table = Table(title="Memory Status", show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("User ID", stats["user_id"])
        table.add_row("Session ID", stats["current_session_id"][:8] + "...")
        table.add_row("Storage Backend", stats["storage_backend"])
        table.add_row("", "")
        table.add_row("[bold]Working Memory[/bold]", "")
        table.add_row("  Messages", str(stats["working_memory"]["message_count"]))
        table.add_row("  Active Topics", str(stats["working_memory"]["active_topics"]))
        table.add_row("  Active Entities", str(stats["working_memory"]["active_entities"]))
        table.add_row("  Recent Tools", str(stats["working_memory"]["recent_tools"]))
        table.add_row("", "")
        table.add_row("[bold]Sessions[/bold]", "")
        table.add_row("  Total Sessions", str(stats["sessions"]["total_sessions"]))
        table.add_row("  Session Started", stats["sessions"]["session_started"][:19])

        console.print()
        console.print(table)
        console.print()

    def _memory_sessions(self) -> None:
        """List recent sessions."""
        sessions = self.memory_manager.get_recent_sessions(limit=10)

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="Recent Sessions", show_header=True)
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Title", style="white", width=40)
        table.add_column("Started", style="dim", width=16)
        table.add_column("Messages", style="green", width=8)
        table.add_column("Active", style="yellow", width=6)

        for session in sessions:
            table.add_row(
                session.session_id[:8] + "...",
                (session.title or "Untitled")[:38],
                session.started_at.strftime("%Y-%m-%d %H:%M"),
                str(session.message_count),
                "Yes" if session.is_active else "No",
            )

        console.print()
        console.print(table)
        console.print()
        console.print("[dim]Use /memory resume <session_id> to resume a session[/dim]")

    def _memory_search(self, query: str) -> None:
        """Search past conversations."""
        if not query:
            console.print("[yellow]Usage:[/yellow] /memory search <query>")
            return

        results = self.memory_manager.search_history(query, limit=5)

        if not results:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            return

        console.print(f"\n[bold cyan]Search results for '{query}':[/bold cyan]\n")

        for i, interaction in enumerate(results, 1):
            console.print(f"[bold]{i}. {interaction.timestamp.strftime('%Y-%m-%d %H:%M')}[/bold]")
            console.print(f"   [cyan]Q:[/cyan] {interaction.user_query[:80]}...")
            console.print(f"   [green]A:[/green] {interaction.agent_response[:100]}...")
            console.print()

    def _memory_profile(self) -> None:
        """Show user profile."""
        profile = self.memory_manager.get_user_profile()

        console.print("\n[bold cyan]User Profile[/bold cyan]\n")
        console.print(f"User ID: [bold]{profile.user_id}[/bold]")
        console.print(f"Detail Level: {profile.preferred_detail_level}")
        console.print(f"Format: {profile.preferred_format}")
        console.print(f"Tone: {profile.tone_preference}")

        if profile.expertise_areas:
            console.print("\n[bold]Expertise Areas:[/bold]")
            for area, confidence in profile.expertise_areas.items():
                console.print(f"  {area}: {confidence:.0%}")

        if profile.topics_of_interest:
            console.print(f"\n[bold]Topics of Interest:[/bold] {', '.join(profile.topics_of_interest)}")

        if profile.inferred_role:
            console.print(f"\n[bold]Inferred Role:[/bold] {profile.inferred_role}")

        console.print()

    def _memory_clear(self) -> None:
        """Clear working memory."""
        confirm = console.input("[yellow]Clear working memory? This cannot be undone. [y/N]:[/yellow] ")
        if confirm.lower() == 'y':
            self.memory_manager.clear_working_memory()
            console.print("[green]Working memory cleared.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")

    def _memory_resume(self, session_id: str) -> None:
        """Resume a previous session."""
        if not session_id:
            console.print("[yellow]Usage:[/yellow] /memory resume <session_id>")
            return

        # Try to find session by partial ID
        sessions = self.memory_manager.get_recent_sessions(limit=50)
        matching = [s for s in sessions if s.session_id.startswith(session_id)]

        if not matching:
            console.print(f"[red]Session not found: {session_id}[/red]")
            return

        if len(matching) > 1:
            console.print("[yellow]Multiple sessions match. Please be more specific:[/yellow]")
            for s in matching[:5]:
                console.print(f"  {s.session_id[:12]}... - {s.title or 'Untitled'}")
            return

        session = self.memory_manager.resume_session(matching[0].session_id)
        if session:
            console.print(f"[green]Resumed session: {session.title or session.session_id[:8]}[/green]")
            console.print(f"[dim]Messages: {session.message_count} | Tools: {session.tool_call_count}[/dim]")
        else:
            console.print(f"[red]Failed to resume session[/red]")

    def cmd_exit(self, args: str = "") -> None:
        """Exit interactive mode."""
        console.print("[cyan]Goodbye![/cyan]")
        sys.exit(0)

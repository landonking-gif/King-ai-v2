"""
Animation Integration Example

Shows how to integrate the new animations into existing interactive.py

This is a reference implementation - copy patterns to your actual interactive.py
"""

from typing import Optional
from rich.console import Console
from rich.live import Live

# Import new animations
from kautilya.animations import (
    WelcomeScreen,
    ModernSpinner,
    TypingEffect,
    ToolExecutionVisualizer,
    SmoothProgress,
    Celebration,
    CommandSuggestion,
    GradientText,
)

console = Console()


# ============================================================================
# EXAMPLE 1: Enhanced Welcome Screen
# ============================================================================

def show_welcome_enhanced():
    """Enhanced welcome screen with animations."""

    # Instead of the old panel-based welcome:
    # console.print(Panel("Welcome to Kautilya..."))

    # Use animated welcome screen:
    WelcomeScreen.show(
        console,
        llm_enabled=True,  # From self._init_llm_client()
        mcp_running=True,  # From gateway_manager.is_running()
        version="1.0.0",
        animate=True,
    )

    # Optionally show initial command suggestions
    if not has_config():
        suggestions = [
            ("/init", "Initialize a new agent project"),
            ("/llm config", "Configure LLM provider"),
            ("/help", "Show all available commands"),
        ]
        CommandSuggestion.show(console, suggestions)


# ============================================================================
# EXAMPLE 2: Enhanced Chat Handler
# ============================================================================

def _handle_chat_enhanced(self, user_input: str) -> None:
    """Enhanced chat handler with beautiful animations."""

    console.print()

    # 1. Start with modern spinner (instead of ThinkingSpinner)
    spinner = ModernSpinner(
        console,
        "Analyzing your request",
        spinner_type="pulse",  # Smooth pulsing animation
        style="cyan",
    )
    spinner.start()

    # Track state
    current_tool = None
    tool_start_time = None
    response_text = ""

    try:
        # Stream response from LLM
        chat_generator = self.llm_client.chat(
            user_input,
            tool_executor=self.tool_executor,
            stream=True,
        )

        for chunk in chat_generator:
            # Detect tool execution
            if chunk.startswith("\n\n> Executing:"):
                # Stop thinking spinner
                spinner.stop()

                # Extract tool name
                tool_name = chunk.split(":")[1].strip().rstrip(".")

                # Show tool execution with beautiful panel
                ToolExecutionVisualizer.show_execution(
                    console,
                    tool_name,
                    args=None,  # Could extract from chunk if available
                )

                current_tool = tool_name
                tool_start_time = time.time()

                # Start new spinner for tool execution
                spinner = ModernSpinner(
                    console,
                    f"Running {tool_name}",
                    spinner_type="dots",
                    style="yellow",
                )
                spinner.start()

            # Detect iteration markers
            elif "[Iteration" in chunk:
                # Show iteration with gradient
                iteration_text = GradientText.apply(chunk.strip(), gradient="cyberpunk")
                spinner.stop()
                console.print(iteration_text)
                spinner = ModernSpinner(console, "Thinking", "pulse", style="cyan")
                spinner.start()

            else:
                # Regular response content
                response_text += chunk

        # Stop final spinner
        spinner.stop()

        # Show tool result if we executed one
        if current_tool and tool_start_time:
            duration = time.time() - tool_start_time
            ToolExecutionVisualizer.show_result(
                console,
                current_tool,
                success=True,
                duration=duration,
            )

        # Type out the response with animation
        if response_text.strip():
            console.print()  # Spacing
            TypingEffect.print_with_typing(
                console,
                response_text.strip(),
                style="cyan",
                speed=0.025,  # Slightly faster than default
            )

    except Exception as e:
        spinner.stop()

        # Show error with beautiful panel
        Celebration.error(
            console,
            f"Error processing request",
            details=str(e),
        )


# ============================================================================
# EXAMPLE 3: Enhanced Command Execution
# ============================================================================

def cmd_agent_new_enhanced(self, args: str) -> None:
    """Enhanced agent creation with progress animations."""

    # Parse arguments
    # name = extract_name(args)
    # role = extract_role(args)
    name = "research-agent"
    role = "research"

    console.print()

    # 1. Show what we're doing
    header = GradientText.apply(
        f"Creating Agent: {name}",
        gradient="ocean",
    )
    console.print(header)
    console.print()

    # 2. Validate with spinner
    validator = ModernSpinner(
        console,
        "Validating agent configuration",
        spinner_type="dots",
        style="cyan",
    )
    validator.start()

    # Simulate validation
    import time
    time.sleep(0.8)

    is_valid = validate_agent_config(name, role)

    if not is_valid:
        validator.stop()
        Celebration.error(
            console,
            "Invalid agent configuration",
            details=f"Agent '{name}' already exists",
        )
        return

    validator.stop("Configuration valid")

    # 3. Create agent with progress bar
    progress = SmoothProgress(
        console,
        "Setting up agent structure",
        total=6,
    )
    progress.start()

    steps = [
        ("Creating directory structure", lambda: create_dirs(name)),
        ("Generating config.yaml", lambda: create_config(name, role)),
        ("Generating capabilities.json", lambda: create_capabilities(name)),
        ("Creating prompt templates", lambda: create_prompts(name)),
        ("Setting up skill directory", lambda: create_skills_dir(name)),
        ("Generating test scaffolding", lambda: create_tests(name)),
    ]

    for i, (description, func) in enumerate(steps):
        progress.update(advance=1, description=description)
        func()
        time.sleep(0.3)  # Simulate work

    progress.complete("Agent structure ready")
    time.sleep(0.5)
    progress.stop()

    # 4. Success celebration
    console.print()
    Celebration.success(
        console,
        f"Agent '{name}' created successfully!\n\n"
        f"Location: agents/{name}/\n"
        f"Role: {role}",
        confetti=True,  # Celebration!
    )

    # 5. Show next steps
    console.print()
    suggestions = [
        (f"/agent test {name}", "Test the new agent"),
        (f"/skill add {name} web_search", "Add web search capability"),
        (f"/manifest new", "Create a workflow for this agent"),
    ]
    CommandSuggestion.show(console, suggestions)


# ============================================================================
# EXAMPLE 4: Enhanced MCP Server Management
# ============================================================================

def cmd_mcp_enable_enhanced(self, tool_id: str) -> None:
    """Enhanced MCP enable with connection animation."""

    # 1. Check gateway connection with pulsing indicator
    from kautilya.animations import PulsingIndicator

    indicator = PulsingIndicator(
        console,
        "Connecting to MCP Gateway",
        style="cyan",
    )
    indicator.start()

    import time
    time.sleep(0.8)  # Simulate connection check

    gateway_ok = check_gateway()

    if not gateway_ok:
        indicator.stop()
        Celebration.error(
            console,
            "MCP Gateway is not running",
            details="Start with: kautilya (gateway auto-starts)\nOr manually: cd mcp-gateway && ./start.sh",
        )
        return

    indicator.stop("[green]✓ Gateway online[/green]")

    # 2. Enable server with tool visualizer
    ToolExecutionVisualizer.show_execution(
        console,
        "mcp_enable",
        args={"tool_id": tool_id},
    )

    start = time.time()
    result = enable_server(tool_id)
    duration = time.time() - start

    ToolExecutionVisualizer.show_result(
        console,
        "mcp_enable",
        success=result["success"],
        duration=duration,
        summary=f"Enabled {tool_id}" if result["success"] else "Failed to enable",
    )

    # 3. Show result
    if result["success"]:
        console.print()
        console.print(f"[green]✓[/green] Server '{tool_id}' is now enabled")

        # Show available tools
        if result.get("tools"):
            console.print(f"[dim]Available tools: {', '.join(result['tools'])}[/dim]")
    else:
        Celebration.error(
            console,
            f"Failed to enable '{tool_id}'",
            details=result.get("error", "Unknown error"),
        )


# ============================================================================
# EXAMPLE 5: Enhanced Status Display
# ============================================================================

def cmd_status_enhanced(self) -> None:
    """Enhanced status display with live updates."""

    from rich.table import Table
    from rich.panel import Panel

    console.print()

    # 1. Gather status with spinner
    spinner = ModernSpinner(
        console,
        "Gathering system status",
        spinner_type="dots",
        style="cyan",
    )
    spinner.start()

    import time
    time.sleep(0.5)

    status = gather_status()
    spinner.stop("Status retrieved")

    # 2. Show status with gradient header
    header = GradientText.apply("System Status", gradient="ocean")
    console.print(header)
    console.print()

    # 3. Create beautiful status table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    # Add rows with colored status
    components = [
        ("LLM Client", status["llm_enabled"], status["llm_model"]),
        ("MCP Gateway", status["mcp_running"], status["mcp_url"]),
        ("Vector DB", status["vector_db"], status["vector_url"]),
        ("Redis", status["redis"], status["redis_url"]),
    ]

    for component, enabled, details in components:
        if enabled:
            status_str = "[green]✓ Online[/green]"
        else:
            status_str = "[dim]○ Offline[/dim]"

        table.add_row(component, status_str, details)

    # Show in panel
    panel = Panel(
        table,
        title="[bold]Kautilya Status[/bold]",
        border_style="cyan",
    )
    console.print(panel)


# ============================================================================
# Helper Functions (stubs for example)
# ============================================================================

def has_config():
    return False

def validate_agent_config(name, role):
    return True

def create_dirs(name):
    pass

def create_config(name, role):
    pass

def create_capabilities(name):
    pass

def create_prompts(name):
    pass

def create_skills_dir(name):
    pass

def create_tests(name):
    pass

def check_gateway():
    return True

def enable_server(tool_id):
    return {"success": True, "tools": ["tool1", "tool2"]}

def gather_status():
    return {
        "llm_enabled": True,
        "llm_model": "gpt-4",
        "mcp_running": True,
        "mcp_url": "http://localhost:8080",
        "vector_db": True,
        "vector_url": "http://localhost:19530",
        "redis": True,
        "redis_url": "redis://localhost:6379",
    }


# ============================================================================
# MAIN INTEGRATION PATTERN
# ============================================================================

def integrate_into_interactive_py():
    """
    Integration checklist for interactive.py:

    1. Add import at top:
        from .animations import (
            WelcomeScreen,
            ModernSpinner,
            TypingEffect,
            ToolExecutionVisualizer,
            SmoothProgress,
            Celebration,
            CommandSuggestion,
            GradientText,
        )

    2. Replace show_welcome():
        - Use WelcomeScreen.show() instead of Panel

    3. Enhance _handle_chat():
        - Replace ThinkingSpinner with ModernSpinner
        - Add ToolExecutionVisualizer for tool calls
        - Use TypingEffect for AI responses
        - Add Celebration for errors

    4. Enhance command handlers:
        - Add progress bars for multi-step operations
        - Use ToolExecutionVisualizer for tool operations
        - Add Celebration for success/failure
        - Add CommandSuggestion for next steps

    5. Add configuration:
        # In __init__ or config
        self.enable_animations = os.getenv("KAUTILYA_ANIMATIONS", "true").lower() == "true"
        self.typing_speed = float(os.getenv("KAUTILYA_TYPING_SPEED", "0.03"))

    6. Graceful fallback:
        if self.enable_animations:
            TypingEffect.print_with_typing(console, text)
        else:
            console.print(text)
    """
    pass


if __name__ == "__main__":
    print("This is a reference implementation.")
    print("See the examples above for integration patterns.")
    print("\nTo test animations, run:")
    print("  python -m kautilya.animations_demo")

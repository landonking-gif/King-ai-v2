#!/usr/bin/env python3
"""
Animation Demo for Kautilya

Shows all available animations and effects in action.
Run this to see what's available and test performance.

Usage:
    python -m kautilya.animations_demo
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .animations import (
    GradientText,
    ParticleEffect,
    TypingEffect,
    SmoothProgress,
    ToolExecutionVisualizer,
    PulsingIndicator,
    WelcomeScreen,
    ModernSpinner,
    Celebration,
    CommandSuggestion,
)

console = Console()


def demo_section(title: str):
    """Print a demo section header."""
    console.print()
    header = Panel(
        Text(title, style="bold cyan", justify="center"),
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(header)
    time.sleep(0.5)


def demo_gradients():
    """Demo gradient text effects."""
    demo_section("1. Gradient Text Effects")

    gradients = ["ocean", "sunset", "forest", "purple", "cyberpunk", "fire", "ice"]

    for gradient in gradients:
        text = f"  {gradient.upper():12} → Kautilya - Agentic Framework CLI"
        gradient_text = GradientText.apply(text, gradient=gradient)
        console.print(gradient_text)
        time.sleep(0.3)


def demo_particles():
    """Demo particle effects."""
    demo_section("2. Particle Burst Effect")

    console.print("[dim]Watch for the particle burst...[/dim]\n")
    time.sleep(1)

    ParticleEffect.create_burst(console, "✨ Tool Executed! ✨", duration=1.5)

    console.print("[green]✓ Particle effect complete[/green]")


def demo_typing():
    """Demo typing effect."""
    demo_section("3. Typing Effect")

    console.print("[dim]Simulating AI response...[/dim]\n")
    time.sleep(0.5)

    TypingEffect.print_with_typing(
        console,
        "I've analyzed your codebase and found 3 optimization opportunities...",
        style="cyan",
        speed=0.04,
    )


def demo_progress():
    """Demo smooth progress bar."""
    demo_section("4. Smooth Progress Bar")

    progress = SmoothProgress(
        console,
        "Processing documents",
        total=100,
    )

    progress.start()

    # Simulate work
    for i in range(10):
        time.sleep(0.3)
        progress.update(advance=10, description=f"Processing document {i+1}/10")

    progress.complete("All documents processed")
    time.sleep(0.5)
    progress.stop()


def demo_tool_execution():
    """Demo tool execution visualizer."""
    demo_section("5. Tool Execution Visualizer")

    # Show execution
    ToolExecutionVisualizer.show_execution(
        console,
        "web_search",
        args={"query": "latest AI frameworks", "max_results": 10},
    )

    # Simulate work
    time.sleep(1.5)

    # Show result
    ToolExecutionVisualizer.show_result(
        console,
        "web_search",
        success=True,
        duration=1.47,
        summary="Found 10 results from 3 sources",
    )


def demo_pulsing_indicator():
    """Demo pulsing indicator."""
    demo_section("6. Pulsing Indicator")

    indicator = PulsingIndicator(console, "Connecting to MCP Gateway", style="cyan")

    indicator.start()
    time.sleep(2.0)
    indicator.stop(final_message="[green]✓ Connected to MCP Gateway[/green]")


def demo_spinners():
    """Demo modern spinners."""
    demo_section("7. Modern Spinner Collection")

    spinner_types = ["dots", "line", "arrows", "bounce", "pulse", "grow"]

    for spinner_type in spinner_types:
        spinner = ModernSpinner(
            console,
            f"Loading ({spinner_type} style)",
            spinner_type=spinner_type,
            style="cyan",
        )

        spinner.start()
        time.sleep(1.5)
        spinner.stop(f"Loaded with {spinner_type} spinner")
        time.sleep(0.3)


def demo_celebrations():
    """Demo success/error celebrations."""
    demo_section("8. Success & Error Celebrations")

    console.print("\n[dim]Success example:[/dim]")
    time.sleep(0.5)
    Celebration.success(
        console,
        "Created 5 new agents and configured LLM adapters",
        confetti=True,
    )

    time.sleep(1.5)

    console.print("\n[dim]Error example:[/dim]")
    time.sleep(0.5)
    Celebration.error(
        console,
        "Failed to connect to OpenAI API",
        details="API key not found in environment. Set OPENAI_API_KEY=sk-...",
    )


def demo_command_suggestions():
    """Demo command suggestions."""
    demo_section("9. Command Suggestions")

    commands = [
        ("/agent new", "Create a new agent"),
        ("/skill import", "Import a skill from marketplace"),
        ("/llm config", "Configure LLM provider"),
        ("/mcp list", "List available MCP servers"),
        ("/chat", "Toggle chat mode"),
    ]

    CommandSuggestion.show(console, commands)


def demo_welcome():
    """Demo welcome screen."""
    demo_section("10. Welcome Screen")

    console.print("\n[dim]Full welcome screen with status:[/dim]\n")
    time.sleep(0.5)

    WelcomeScreen.show(
        console,
        llm_enabled=True,
        mcp_running=True,
        version="1.0.0",
        animate=False,  # Skip matrix for demo
    )


def demo_complete_workflow():
    """Demo a complete workflow with multiple animations."""
    demo_section("11. Complete Workflow Example")

    console.print("\n[bold cyan]Simulating: User asks to create a new agent[/bold cyan]\n")
    time.sleep(1)

    # 1. Thinking
    spinner = ModernSpinner(
        console,
        "Understanding your request",
        spinner_type="pulse",
        style="cyan",
    )
    spinner.start()
    time.sleep(1.5)
    spinner.stop("Request understood")

    time.sleep(0.3)

    # 2. Planning
    TypingEffect.print_with_typing(
        console,
        "I'll help you create a research agent with web search capabilities.",
        style="cyan",
        speed=0.03,
    )

    time.sleep(0.5)

    # 3. Tool execution
    ToolExecutionVisualizer.show_execution(
        console,
        "agent_scaffolder",
        args={"name": "research-agent", "role": "research"},
    )

    time.sleep(1.0)

    ToolExecutionVisualizer.show_result(
        console,
        "agent_scaffolder",
        success=True,
        duration=0.85,
        summary="Created agent directory structure",
    )

    time.sleep(0.5)

    # 4. Progress for file operations
    progress = SmoothProgress(console, "Setting up agent files", total=5)
    progress.start()

    files = ["config.yaml", "capabilities.json", "prompts/system.txt", "skills/", "tests/"]

    for i, file in enumerate(files):
        time.sleep(0.4)
        progress.update(advance=1, description=f"Creating {file}")

    progress.complete("Agent files ready")
    time.sleep(0.5)
    progress.stop()

    # 5. Success celebration
    time.sleep(0.5)
    Celebration.success(
        console,
        "Research agent created successfully!\n\nNext steps:\n• Configure capabilities\n• Add custom skills\n• Test with /run command",
        confetti=True,
    )

    time.sleep(1)

    # 6. Command suggestions
    commands = [
        ("/agent test research-agent", "Test the new agent"),
        ("/skill add research-agent summarize", "Add summarize skill"),
        ("/run research-agent", "Run the agent"),
    ]

    CommandSuggestion.show(console, commands)


def main():
    """Run all demos."""
    console.clear()

    title = Panel(
        Text.from_markup(
            "[bold cyan]Kautilya Animation Demo[/bold cyan]\n\n"
            "[dim]Showcasing all available animations and effects[/dim]"
        ),
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(title)
    console.print()

    # Run demos
    demos = [
        ("Gradient Text", demo_gradients),
        ("Particle Effects", demo_particles),
        ("Typing Effect", demo_typing),
        ("Progress Bars", demo_progress),
        ("Tool Execution", demo_tool_execution),
        ("Pulsing Indicator", demo_pulsing_indicator),
        ("Spinners", demo_spinners),
        ("Celebrations", demo_celebrations),
        ("Command Suggestions", demo_command_suggestions),
        ("Welcome Screen", demo_welcome),
        ("Complete Workflow", demo_complete_workflow),
    ]

    try:
        for name, demo_func in demos:
            demo_func()
            time.sleep(1.0)

        # Final message
        console.print()
        final_panel = Panel(
            GradientText.apply(
                "Demo Complete! All animations ready to use in Kautilya.",
                gradient="cyberpunk",
            ),
            border_style="green",
            padding=(1, 2),
        )
        console.print(final_panel)
        console.print()

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted by user[/yellow]")


if __name__ == "__main__":
    main()

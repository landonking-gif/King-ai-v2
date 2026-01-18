"""
Modern CLI Animations for Kautilya

Module: kautilya/animations.py

Provides engaging, minimalistic animations for the agentic framework CLI.
Enhances UX with smooth transitions, particle effects, and beautiful visualizations.
"""

import time
import random
import threading
from typing import Optional, List, Tuple, Callable
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.box import ROUNDED, MINIMAL, HEAVY
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)


class AnimationStyle(str, Enum):
    """Animation style presets."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    FANCY = "fancy"


# ============================================================================
# Gradient Text Effects
# ============================================================================

class GradientText:
    """Create beautiful gradient text effects."""

    GRADIENTS = {
        "ocean": ["#0077be", "#00a8e8", "#00c9ff"],
        "sunset": ["#ff6b6b", "#ff8e53", "#ffd93d"],
        "forest": ["#2d6a4f", "#40916c", "#52b788"],
        "purple": ["#7209b7", "#b5179e", "#f72585"],
        "cyberpunk": ["#00ff41", "#00b8ff", "#ff00ff"],
        "fire": ["#ff0000", "#ff6600", "#ffaa00"],
        "ice": ["#4cc9f0", "#4895ef", "#4361ee"],
        "saffron": ["#ff9933", "#ffb347", "#ffd700"],  # Saffron to gold - ancient Indian wisdom
    }

    @staticmethod
    def apply(text: str, gradient: str = "ocean") -> Text:
        """Apply gradient to text."""
        colors = GradientText.GRADIENTS.get(gradient, GradientText.GRADIENTS["ocean"])
        result = Text()

        chars_per_color = max(1, len(text) // len(colors))

        for i, char in enumerate(text):
            color_idx = min(i // chars_per_color, len(colors) - 1)
            result.append(char, style=colors[color_idx])

        return result


# ============================================================================
# Particle Effects
# ============================================================================

class ParticleEffect:
    """Create particle effects for tool execution."""

    PARTICLES = ["✨", "⭐", "🌟", "💫", "✦", "✧", "⚡", "🔥"]
    FRAMES = 20

    @staticmethod
    def create_burst(console: Console, center_text: str, duration: float = 1.0):
        """Create a particle burst effect."""
        particles = []

        # Generate random particles
        for _ in range(8):
            particles.append({
                "char": random.choice(ParticleEffect.PARTICLES),
                "x": random.randint(-15, 15),
                "y": random.randint(-3, 3),
            })

        frame_duration = duration / ParticleEffect.FRAMES

        for frame in range(ParticleEffect.FRAMES):
            # Calculate particle positions (expanding outward)
            progress = frame / ParticleEffect.FRAMES

            display = Text()
            display.append("\n" * 2)

            # Center text
            display.append(center_text, style="bold cyan")
            display.append("\n")

            # Draw particles
            for p in particles:
                final_x = int(p["x"] * progress)
                final_y = int(p["y"] * progress)

                if abs(final_y) == 1:
                    spacing = " " * (len(center_text) // 2 + final_x)
                    display.append(spacing + p["char"] + "\n", style="yellow")

            console.print(display, end="\r")
            time.sleep(frame_duration)

        # Clear particles
        console.print("\n" * 4, end="\r")


# ============================================================================
# Typing Effect
# ============================================================================

class TypingEffect:
    """Create realistic typing effect for text."""

    @staticmethod
    def print_with_typing(
        console: Console,
        text: str,
        style: str = "white",
        speed: float = 0.03,
    ):
        """Print text with typing animation."""
        display_text = Text()

        for char in text:
            display_text.append(char, style=style)
            console.print(display_text, end="\r")
            time.sleep(speed + random.uniform(-0.01, 0.01))

        console.print()  # Final newline


# ============================================================================
# Smooth Progress Bar
# ============================================================================

class SmoothProgress:
    """Smooth progress bar with gradient effects."""

    def __init__(
        self,
        console: Console,
        description: str,
        total: int = 100,
        style: str = "ocean",
    ):
        """Initialize smooth progress."""
        self.console = console
        self.description = description
        self.total = total
        self.style = style
        self.current = 0
        self.live = None
        self.progress = None
        self.task_id = None

    def start(self):
        """Start the progress bar."""
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(
                complete_style="cyan",
                finished_style="green",
                bar_width=40,
            ),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

        self.task_id = self.progress.add_task(
            self.description,
            total=self.total,
        )

        self.live = Live(
            self.progress,
            console=self.console,
            refresh_per_second=20,
            transient=False,
        )
        self.live.start()

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress."""
        if self.progress and self.task_id is not None:
            if description:
                self.progress.update(
                    self.task_id,
                    advance=advance,
                    description=description,
                )
            else:
                self.progress.update(self.task_id, advance=advance)

    def complete(self, message: str = "✓ Complete"):
        """Mark as complete with celebration."""
        if self.progress and self.task_id is not None:
            # Jump to 100%
            remaining = self.total - self.current
            self.progress.update(self.task_id, advance=remaining)

            # Update description
            self.progress.update(
                self.task_id,
                description=f"[bold green]{message}[/bold green]",
            )

            # Let it show for a moment
            time.sleep(0.5)

    def stop(self):
        """Stop the progress bar."""
        if self.live:
            self.live.stop()
            self.live = None


# ============================================================================
# Tool Execution Visualizer
# ============================================================================

class ToolExecutionVisualizer:
    """Beautiful visualizations for tool execution."""

    TOOL_ICONS = {
        "search": "🔍",
        "web": "🌐",
        "file": "📄",
        "database": "🗄️",
        "api": "🔌",
        "code": "💻",
        "analysis": "📊",
        "default": "⚡",
    }

    @staticmethod
    def get_icon(tool_name: str) -> str:
        """Get icon for tool."""
        tool_lower = tool_name.lower()
        for key, icon in ToolExecutionVisualizer.TOOL_ICONS.items():
            if key in tool_lower:
                return icon
        return ToolExecutionVisualizer.TOOL_ICONS["default"]

    @staticmethod
    def show_execution(
        console: Console,
        tool_name: str,
        args: Optional[dict] = None,
    ):
        """Show tool execution start."""
        icon = ToolExecutionVisualizer.get_icon(tool_name)

        # Create execution panel
        content = Text()
        content.append(f"{icon} ", style="bold yellow")
        content.append(tool_name, style="bold cyan")

        if args:
            content.append("\n\n", style="dim")
            # Show pretty-printed args
            for key, value in list(args.items())[:3]:  # Max 3 args
                content.append(f"  {key}: ", style="dim")
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:47] + "..."
                content.append(f"{str_value}\n", style="cyan")

        panel = Panel(
            content,
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 2),
            title="[bold yellow]Executing Tool[/bold yellow]",
            title_align="left",
        )

        console.print(panel)

    @staticmethod
    def show_result(
        console: Console,
        tool_name: str,
        success: bool = True,
        duration: float = 0.0,
        summary: Optional[str] = None,
    ):
        """Show tool execution result."""
        icon = ToolExecutionVisualizer.get_icon(tool_name)

        if success:
            # Success animation
            result_text = Text()
            result_text.append("✓ ", style="bold green")
            result_text.append(tool_name, style="green")
            result_text.append(f" ({duration:.2f}s)", style="dim")

            if summary:
                result_text.append(f"\n  {summary}", style="dim")

            console.print(result_text)
        else:
            # Error display
            result_text = Text()
            result_text.append("✗ ", style="bold red")
            result_text.append(tool_name, style="red")
            result_text.append(" failed", style="dim red")

            console.print(result_text)


# ============================================================================
# Pulsing Status Indicator
# ============================================================================

class PulsingIndicator:
    """Animated pulsing indicator for long operations."""

    PULSE_FRAMES = ["◐", "◓", "◑", "◒"]

    def __init__(self, console: Console, message: str, style: str = "cyan"):
        """Initialize pulsing indicator."""
        self.console = console
        self.message = message
        self.style = style
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_idx = 0

    def _animate(self):
        """Animation loop."""
        while self.running:
            frame = self.PULSE_FRAMES[self.frame_idx % len(self.PULSE_FRAMES)]
            self.frame_idx += 1

            display = Text()
            display.append(f"{frame} ", style=self.style)
            display.append(self.message, style=f"bold {self.style}")

            self.console.print(display, end="\r")
            time.sleep(0.15)

    def start(self):
        """Start pulsing."""
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self, final_message: Optional[str] = None):
        """Stop pulsing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

        # Clear line
        self.console.print(" " * 80, end="\r")

        if final_message:
            self.console.print(final_message)


# ============================================================================
# Matrix Rain Effect (for startup)
# ============================================================================

class MatrixRain:
    """Minimalistic matrix rain effect for startup."""

    @staticmethod
    def show(console: Console, duration: float = 2.0, height: int = 10):
        """Show matrix rain effect."""
        chars = "01"
        width = 60
        columns = [random.randint(0, height) for _ in range(width)]

        frames = int(duration * 10)  # 10 fps

        for _ in range(frames):
            lines = []

            for row in range(height):
                line = Text()
                for col in range(width):
                    if columns[col] == row:
                        # Bright character at head
                        line.append(random.choice(chars), style="bold green")
                    elif columns[col] > row and columns[col] - row < 5:
                        # Fading trail
                        line.append(random.choice(chars), style="green")
                    else:
                        line.append(" ")
                lines.append(line)

            # Move columns down
            for i in range(len(columns)):
                columns[i] += 1
                if columns[i] > height + 5:
                    columns[i] = 0

            # Print frame
            for line in lines:
                console.print(line)

            time.sleep(0.1)

            # Clear for next frame
            console.print(f"\033[{height}A", end="")  # Move cursor up


# ============================================================================
# Welcome Screen Animation
# ============================================================================

class WelcomeScreen:
    """Beautiful welcome screen with animations."""

    LOGO = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   ██╗  ██╗ █████╗ ██╗   ██╗████████╗██╗██╗  ██╗   ██╗ █████╗             ║
║   ██║ ██╔╝██╔══██╗██║   ██║╚══██╔══╝██║██║  ╚██╗ ██╔╝██╔══██╗            ║
║   █████╔╝ ███████║██║   ██║   ██║   ██║██║   ╚████╔╝ ███████║            ║
║   ██╔═██╗ ██╔══██║██║   ██║   ██║   ██║██║    ╚██╔╝  ██╔══██║            ║
║   ██║  ██╗██║  ██║╚██████╔╝   ██║   ██║███████╗██║   ██║  ██║            ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝            ║
║                                                                            ║
║              🕉️  Ancient Wisdom, Modern AI Strategy  📜                    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """

    @staticmethod
    def show(
        console: Console,
        llm_enabled: bool = False,
        mcp_running: bool = False,
        version: str = "1.0.0",
        animate: bool = True,
    ):
        """Show welcome screen with status."""

        if animate:
            # Quick matrix effect
            # MatrixRain.show(console, duration=1.0, height=6)
            pass  # Skip matrix for speed

        # Show logo with gradient
        logo_lines = WelcomeScreen.LOGO.split("\n")
        for line in logo_lines:
            gradient_line = GradientText.apply(line, gradient="saffron")
            console.print(gradient_line)
            if animate:
                time.sleep(0.05)

        # Status table
        status_table = Table(
            show_header=False,
            box=MINIMAL,
            padding=(0, 2),
            collapse_padding=True,
        )

        status_table.add_column(style="dim", justify="right")
        status_table.add_column(style="cyan")

        # Add status rows
        status_table.add_row(
            "Version:",
            f"[bold cyan]{version}[/bold cyan]",
        )

        llm_status = "[bold green]✓ Online[/bold green]" if llm_enabled else "[dim]○ Offline[/dim]"
        status_table.add_row("LLM Chat:", llm_status)

        mcp_status = "[bold green]✓ Running[/bold green]" if mcp_running else "[dim]○ Stopped[/dim]"
        status_table.add_row("MCP Gateway:", mcp_status)

        console.print()
        console.print(Align.center(status_table))
        console.print()

        # Quick tips
        tips = Panel(
            Text.from_markup(
                "[bold cyan]Quick Tips[/bold cyan]\n\n"
                "• Type naturally to chat with AI\n"
                "• Use [bold]/commands[/bold] for direct actions\n"
                "• Press [bold]Tab[/bold] for auto-completion\n"
                "• Type [bold]/help[/bold] to see all commands"
            ),
            border_style="dim",
            box=ROUNDED,
            padding=(0, 2),
        )

        console.print(tips)
        console.print()


# ============================================================================
# Loading Spinner Collection
# ============================================================================

class ModernSpinner:
    """Collection of modern spinner animations."""

    SPINNERS = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["─", "\\", "|", "/"],
        "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "bounce": ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"],
        "pulse": ["○", "◔", "◐", "◕", "●", "◕", "◐", "◔"],
        "grow": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"],
    }

    def __init__(
        self,
        console: Console,
        message: str,
        spinner_type: str = "dots",
        style: str = "cyan",
    ):
        """Initialize modern spinner."""
        self.console = console
        self.message = message
        self.frames = self.SPINNERS.get(spinner_type, self.SPINNERS["dots"])
        self.style = style
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_idx = 0
        self.start_time = 0.0

    def _spin(self):
        """Spin animation."""
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.frames[self.frame_idx % len(self.frames)]
            self.frame_idx += 1

            display = Text()
            display.append(f"{frame} ", style=self.style)
            display.append(self.message, style=f"bold {self.style}")

            # Add elapsed time for long operations
            if elapsed > 3.0:
                mins = int(elapsed // 60)
                secs = elapsed % 60
                if mins > 0:
                    time_str = f"{mins}m {secs:.0f}s"
                else:
                    time_str = f"{secs:.1f}s"
                display.append(f" ({time_str})", style="dim")

            self.console.print(display, end="\r")
            time.sleep(0.08)

    def start(self):
        """Start spinner."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self, final_message: Optional[str] = None) -> float:
        """
        Stop spinner.

        Args:
            final_message: Optional message to display after stopping

        Returns:
            Elapsed time in seconds
        """
        self.running = False
        elapsed = time.time() - self.start_time

        if self.thread:
            self.thread.join(timeout=0.5)

        # Clear line
        self.console.print(" " * 100, end="\r")

        if final_message:
            result = Text()
            result.append("✓ ", style="green")
            result.append(final_message, style="green")
            result.append(f" ({elapsed:.2f}s)", style="dim")
            self.console.print(result)

        return elapsed


# ============================================================================
# Success/Error Celebrations
# ============================================================================

class Celebration:
    """Success and error visual feedback."""

    @staticmethod
    def success(console: Console, message: str, confetti: bool = False):
        """Show success with optional confetti."""
        if confetti:
            # Quick confetti burst
            confetti_chars = ["🎉", "✨", "🎊", "⭐"]
            line = Text()
            for _ in range(20):
                line.append(random.choice(confetti_chars) + " ", style="yellow")
            console.print(line)

        # Success panel
        panel = Panel(
            Text.from_markup(
                f"[bold green]✓ Success![/bold green]\n\n"
                f"{message}"
            ),
            border_style="green",
            box=HEAVY,
            padding=(1, 3),
        )
        console.print(panel)

    @staticmethod
    def error(console: Console, message: str, details: Optional[str] = None):
        """Show error with details."""
        content = Text.from_markup(
            f"[bold red]✗ Error[/bold red]\n\n"
            f"{message}"
        )

        if details:
            content.append("\n\n", style="dim")
            content.append("Details:\n", style="dim")
            content.append(details, style="red")

        panel = Panel(
            content,
            border_style="red",
            box=HEAVY,
            padding=(1, 3),
        )
        console.print(panel)


# ============================================================================
# Command Suggestion Animation
# ============================================================================

class CommandSuggestion:
    """Animated command suggestions."""

    @staticmethod
    def show(console: Console, commands: List[Tuple[str, str]]):
        """Show command suggestions with fade-in."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=ROUNDED,
            padding=(0, 1),
        )

        table.add_column("Command", style="cyan")
        table.add_column("Description", style="dim")

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        panel = Panel(
            table,
            title="[bold]💡 Suggested Commands[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )

        console.print(panel)

# ============================================================================
# In-Place Iteration Display
# ============================================================================

class IterationProgress:
    """Live iteration display that updates in place with progress bar."""

    # Tool type emojis for minimal mode
    TOOL_EMOJIS = {
        "file_read": "📖",
        "read_file": "📖",
        "file_write": "✏️",
        "write_file": "✏️",
        "file_edit": "✏️",
        "edit_file": "✏️",
        "file_glob": "🔍",
        "glob_files": "🔍",
        "file_grep": "🔎",
        "grep_code": "🔎",
        "bash_exec": "💻",
        "bash_execute": "💻",
        "web_search": "🌐",
        "web_fetch": "🌐",
        "mcp_call": "🔌",
        "llm_config": "⚙️",
        "llm_test": "🧪",
        "default": "⚡",
    }

    def __init__(self, console: Console, max_iterations: int = 5):
        """Initialize iteration progress display."""
        self.console = console
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.current_step_info = ""
        self.current_tools = []
        self.live = None
        self.progress = None
        self.task_id = None
        self.iteration_start_time = None
        self.tokens_used = 0

    def _get_display_mode(self) -> str:
        """Get current display mode."""
        from .iteration_display import get_display_mode
        return get_display_mode().value

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

    def start(self):
        """Start the live display."""
        import time as time_module
        self.iteration_start_time = time_module.time()

        # Skip live display in minimal mode
        if self._get_display_mode() == "minimal":
            return

        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green"),
            TextColumn("[cyan]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False,
        )

        self.task_id = self.progress.add_task(
            f"Iteration 0/{self.max_iterations}",
            total=self.max_iterations,
        )

        # Create live display with progress and info panel
        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self.live.start()

    def _build_display(self) -> Panel:
        """Build the display panel with progress and info."""
        from rich.text import Text
        from rich.padding import Padding

        # Build content
        content = []

        # Add progress bar
        if self.progress:
            content.append(self.progress)

        # Add step information if available
        if self.current_step_info:
            info_text = Text()
            info_text.append("\n📍 ", style="cyan")
            info_text.append(self.current_step_info, style="white")
            content.append(info_text)

        # Add tools being used
        if self.current_tools:
            tools_text = Text()
            tools_text.append("\n🔧 Tools: ", style="dim cyan")
            tools_text.append(", ".join(self.current_tools), style="yellow")
            content.append(tools_text)

        # Combine into panel
        from rich.console import Group

        panel = Panel(
            Group(*content) if content else Text("Initializing...", style="dim"),
            title=f"[bold cyan]🕉️  Kautilya Strategy Engine[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

        return panel

    def update_iteration(self, iteration: int, step_info: str = "", tools: list = None):
        """Update current iteration and information."""
        import time as time_module

        self.current_iteration = iteration
        self.current_step_info = step_info
        self.current_tools = tools or []

        # In minimal mode, print compact line
        if self._get_display_mode() == "minimal":
            self._print_minimal_status(step_info, tools)
            return

        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id,
                completed=iteration,
                description=f"Iteration {iteration}/{self.max_iterations}",
            )

        if self.live:
            self.live.update(self._build_display())

    def update_info(self, step_info: str, tools: list = None):
        """Update just the step information without changing iteration."""
        self.current_step_info = step_info
        if tools is not None:
            self.current_tools = tools

        # In minimal mode, print compact line for tool updates
        if self._get_display_mode() == "minimal":
            self._print_minimal_status(step_info, tools)
            return

        if self.live:
            self.live.update(self._build_display())

    def _print_minimal_status(self, step_info: str = "", tools: list = None):
        """Print a minimal one-liner status."""
        import time as time_module
        from rich.text import Text

        elapsed = 0
        if self.iteration_start_time:
            elapsed = time_module.time() - self.iteration_start_time

        parts = []

        # Progress indicator
        parts.append(f"[dim][{self.current_iteration}/{self.max_iterations}][/dim]")

        # Tool emoji and name
        if tools and len(tools) > 0:
            tool_name = tools[-1] if tools else ""
            emoji = self._get_tool_emoji(tool_name)
            parts.append(f"{emoji} [bold yellow]{tool_name}[/bold yellow]")

        # Step info (thinking/reasoning)
        if step_info:
            # Truncate and clean up step info
            info = step_info.replace("Starting iteration", "").replace("analyzing the request", "").strip()
            info = info.replace("Planning:", "").replace("Strategy:", "").strip()
            if info.startswith(f"{self.current_iteration} -"):
                info = info[len(f"{self.current_iteration} -"):].strip()
            if info and len(info) > 3:
                if len(info) > 40:
                    info = info[:37] + "..."
                parts.append(f"│ 💭 [dim]\"{info}\"[/dim]")

        # Time elapsed
        parts.append(f"│ ⏱️ [dim]{self._format_duration(elapsed)}[/dim]")

        # Token count if available
        if self.tokens_used > 0:
            token_str = f"{self.tokens_used / 1000:.1f}k" if self.tokens_used >= 1000 else str(self.tokens_used)
            parts.append(f"│ 🎫 [dim]{token_str}[/dim]")

        self.console.print(Text.from_markup(" ".join(parts)))

    def complete(self, success: bool = True):
        """Complete the iteration display."""
        import time as time_module
        from rich.text import Text

        # In minimal mode, print completion line
        if self._get_display_mode() == "minimal":
            elapsed = 0
            if self.iteration_start_time:
                elapsed = time_module.time() - self.iteration_start_time

            if success:
                self.console.print(Text.from_markup(
                    f"[bold green]✓ Complete[/bold green] │ "
                    f"[dim]{self.current_iteration} iterations │ "
                    f"{len(self.current_tools)} tools │ "
                    f"⏱️ {self._format_duration(elapsed)}[/dim]"
                ))
            else:
                self.console.print(Text.from_markup(
                    f"[bold red]✗ Failed[/bold red] │ "
                    f"[dim]after {self.current_iteration} iterations[/dim]"
                ))
            return

        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id,
                completed=self.max_iterations,
                description=f"{'✓ Complete' if success else '✗ Failed'}",
            )

        if self.live:
            self.live.update(self._build_display())
            time.sleep(0.5)  # Brief pause to show completion
            self.live.stop()

    def stop(self):
        """Stop the live display."""
        # In minimal mode, there's no live display to stop
        if self._get_display_mode() == "minimal":
            return

        if self.live:
            self.live.stop()

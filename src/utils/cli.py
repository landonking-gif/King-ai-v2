"""
CLI Enhancements.
Interactive command-line interface with rich output.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("cli")


class OutputStyle(str, Enum):
    """Output styling options."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HEADER = "header"
    DIM = "dim"
    BOLD = "bold"


# ANSI color codes
COLORS = {
    OutputStyle.SUCCESS: "\033[92m",  # Green
    OutputStyle.ERROR: "\033[91m",  # Red
    OutputStyle.WARNING: "\033[93m",  # Yellow
    OutputStyle.INFO: "\033[94m",  # Blue
    OutputStyle.HEADER: "\033[95m",  # Magenta
    OutputStyle.DIM: "\033[90m",  # Gray
    OutputStyle.BOLD: "\033[1m",
}
RESET = "\033[0m"


def colorize(text: str, style: OutputStyle) -> str:
    """Apply color to text."""
    return f"{COLORS.get(style, '')}{text}{RESET}"


@dataclass
class CommandArg:
    """Command argument definition."""
    name: str
    description: str
    required: bool = False
    default: Any = None
    arg_type: type = str
    choices: List[str] = field(default_factory=list)


@dataclass
class Command:
    """CLI command definition."""
    name: str
    description: str
    handler: Callable
    args: List[CommandArg] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    category: str = "general"
    hidden: bool = False


class TableFormatter:
    """Format data as tables."""
    
    @staticmethod
    def format(
        headers: List[str],
        rows: List[List[Any]],
        max_width: int = 80,
    ) -> str:
        """Format data as an ASCII table."""
        if not headers or not rows:
            return ""
        
        # Calculate column widths
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        
        # Limit column widths
        total_width = sum(widths) + len(widths) * 3 + 1
        if total_width > max_width:
            factor = max_width / total_width
            widths = [max(5, int(w * factor)) for w in widths]
        
        # Build table
        lines = []
        
        # Header separator
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        lines.append(separator)
        
        # Header row
        header_cells = []
        for h, w in zip(headers, widths):
            header_cells.append(f" {str(h)[:w]:<{w}} ")
        lines.append("|" + "|".join(header_cells) + "|")
        lines.append(separator)
        
        # Data rows
        for row in rows:
            cells = []
            for i, (cell, w) in enumerate(zip(row, widths)):
                cell_str = str(cell)[:w]
                cells.append(f" {cell_str:<{w}} ")
            lines.append("|" + "|".join(cells) + "|")
        
        lines.append(separator)
        
        return "\n".join(lines)


class ProgressBar:
    """Display progress in terminal."""
    
    def __init__(
        self,
        total: int,
        width: int = 40,
        description: str = "",
    ):
        self.total = total
        self.width = width
        self.description = description
        self.current = 0
    
    def update(self, amount: int = 1) -> None:
        """Update progress."""
        self.current = min(self.current + amount, self.total)
        self._render()
    
    def set(self, value: int) -> None:
        """Set progress to specific value."""
        self.current = min(value, self.total)
        self._render()
    
    def _render(self) -> None:
        """Render progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = int((self.current / self.total) * 100)
        
        filled = int(self.width * self.current / max(self.total, 1))
        bar = "█" * filled + "░" * (self.width - filled)
        
        line = f"\r{self.description} [{bar}] {percent}% ({self.current}/{self.total})"
        sys.stdout.write(line)
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")
    
    def complete(self) -> None:
        """Complete the progress bar."""
        self.set(self.total)


class Spinner:
    """Display a spinner for ongoing operations."""
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Loading"):
        self.message = message
        self.running = False
        self.frame_idx = 0
    
    async def start(self) -> None:
        """Start the spinner."""
        self.running = True
        while self.running:
            frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
            sys.stdout.write(f"\r{frame} {self.message}")
            sys.stdout.flush()
            self.frame_idx += 1
            await asyncio.sleep(0.1)
    
    def stop(self, success: bool = True) -> None:
        """Stop the spinner."""
        self.running = False
        icon = colorize("✓", OutputStyle.SUCCESS) if success else colorize("✗", OutputStyle.ERROR)
        sys.stdout.write(f"\r{icon} {self.message}\n")
        sys.stdout.flush()


class InteractiveCLI:
    """
    Interactive CLI with rich output.
    
    Features:
    - Command registration
    - Argument parsing
    - Colored output
    - Tables
    - Progress bars
    - Interactive prompts
    """
    
    def __init__(
        self,
        name: str = "King AI",
        version: str = "2.0.0",
        description: str = "AI-powered business automation",
    ):
        self.name = name
        self.version = version
        self.description = description
        self.commands: Dict[str, Command] = {}
        self.aliases: Dict[str, str] = {}
        
        self._register_builtin_commands()
    
    def _register_builtin_commands(self) -> None:
        """Register built-in commands."""
        self.register_command(
            name="help",
            description="Show help for commands",
            handler=self._cmd_help,
            args=[
                CommandArg("command", "Command to get help for", required=False),
            ],
        )
        
        self.register_command(
            name="version",
            description="Show version information",
            handler=self._cmd_version,
            aliases=["v", "--version"],
        )
        
        self.register_command(
            name="exit",
            description="Exit the CLI",
            handler=self._cmd_exit,
            aliases=["quit", "q"],
        )
    
    def register_command(
        self,
        name: str,
        description: str,
        handler: Callable,
        args: List[CommandArg] = None,
        aliases: List[str] = None,
        category: str = "general",
        hidden: bool = False,
    ) -> None:
        """
        Register a command.
        
        Args:
            name: Command name
            description: Command description
            handler: Command handler function
            args: Command arguments
            aliases: Command aliases
            category: Command category
            hidden: Whether to hide from help
        """
        command = Command(
            name=name,
            description=description,
            handler=handler,
            args=args or [],
            aliases=aliases or [],
            category=category,
            hidden=hidden,
        )
        
        self.commands[name] = command
        
        for alias in command.aliases:
            self.aliases[alias] = name
    
    async def execute(self, command_line: str) -> Any:
        """
        Execute a command line.
        
        Args:
            command_line: The command to execute
            
        Returns:
            Command result
        """
        parts = command_line.strip().split()
        if not parts:
            return None
        
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        # Resolve alias
        if cmd_name in self.aliases:
            cmd_name = self.aliases[cmd_name]
        
        command = self.commands.get(cmd_name)
        if not command:
            self.error(f"Unknown command: {cmd_name}")
            self.info("Type 'help' for available commands")
            return None
        
        # Parse arguments
        try:
            parsed_args = self._parse_args(command, args)
        except ValueError as e:
            self.error(str(e))
            return None
        
        # Execute handler
        try:
            if asyncio.iscoroutinefunction(command.handler):
                return await command.handler(**parsed_args)
            else:
                return command.handler(**parsed_args)
        except Exception as e:
            self.error(f"Command failed: {e}")
            logger.exception(f"Command {cmd_name} failed")
            return None
    
    def _parse_args(
        self,
        command: Command,
        args: List[str],
    ) -> Dict[str, Any]:
        """Parse command arguments."""
        parsed = {}
        arg_idx = 0
        
        for cmd_arg in command.args:
            if arg_idx < len(args):
                value = args[arg_idx]
                
                # Validate choices
                if cmd_arg.choices and value not in cmd_arg.choices:
                    raise ValueError(
                        f"Invalid value for {cmd_arg.name}: {value}. "
                        f"Choices: {', '.join(cmd_arg.choices)}"
                    )
                
                # Type conversion
                try:
                    parsed[cmd_arg.name] = cmd_arg.arg_type(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid type for {cmd_arg.name}: expected {cmd_arg.arg_type.__name__}"
                    )
                
                arg_idx += 1
            elif cmd_arg.required:
                raise ValueError(f"Missing required argument: {cmd_arg.name}")
            else:
                parsed[cmd_arg.name] = cmd_arg.default
        
        return parsed
    
    async def run_interactive(self) -> None:
        """Run interactive CLI loop."""
        self.print_header()
        
        while True:
            try:
                prompt = colorize("king-ai> ", OutputStyle.INFO)
                command_line = input(prompt)
                
                if command_line.strip():
                    result = await self.execute(command_line)
                    
                    if result == "EXIT":
                        break
                    
            except KeyboardInterrupt:
                print()
                self.warning("Use 'exit' or 'quit' to leave")
            except EOFError:
                break
        
        self.success("Goodbye!")
    
    def print_header(self) -> None:
        """Print CLI header."""
        header = f"""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   {colorize(self.name, OutputStyle.BOLD):<54}║
║   {colorize(self.description, OutputStyle.DIM):<54}║
║   {colorize(f"Version {self.version}", OutputStyle.DIM):<54}║
║                                                       ║
║   Type 'help' for available commands                  ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
"""
        print(header)
    
    # Output methods
    def print(self, message: str) -> None:
        """Print a message."""
        print(message)
    
    def success(self, message: str) -> None:
        """Print a success message."""
        print(colorize(f"✓ {message}", OutputStyle.SUCCESS))
    
    def error(self, message: str) -> None:
        """Print an error message."""
        print(colorize(f"✗ {message}", OutputStyle.ERROR))
    
    def warning(self, message: str) -> None:
        """Print a warning message."""
        print(colorize(f"⚠ {message}", OutputStyle.WARNING))
    
    def info(self, message: str) -> None:
        """Print an info message."""
        print(colorize(f"ℹ {message}", OutputStyle.INFO))
    
    def header(self, message: str) -> None:
        """Print a header."""
        print()
        print(colorize(f"═══ {message} ═══", OutputStyle.HEADER))
        print()
    
    def table(
        self,
        headers: List[str],
        rows: List[List[Any]],
    ) -> None:
        """Print a table."""
        print(TableFormatter.format(headers, rows))
    
    def list_items(self, items: List[str], numbered: bool = False) -> None:
        """Print a list of items."""
        for i, item in enumerate(items, 1):
            if numbered:
                print(f"  {i}. {item}")
            else:
                print(f"  • {item}")
    
    def progress(
        self,
        total: int,
        description: str = "",
    ) -> ProgressBar:
        """Create a progress bar."""
        return ProgressBar(total, description=description)
    
    def spinner(self, message: str = "Loading") -> Spinner:
        """Create a spinner."""
        return Spinner(message)
    
    # Interactive prompts
    def prompt(
        self,
        message: str,
        default: str = None,
    ) -> str:
        """Prompt for text input."""
        if default:
            prompt_text = f"{message} [{default}]: "
        else:
            prompt_text = f"{message}: "
        
        response = input(prompt_text).strip()
        return response or default or ""
    
    def confirm(
        self,
        message: str,
        default: bool = False,
    ) -> bool:
        """Prompt for confirmation."""
        options = "[Y/n]" if default else "[y/N]"
        response = input(f"{message} {options}: ").strip().lower()
        
        if not response:
            return default
        
        return response in ("y", "yes", "1", "true")
    
    def select(
        self,
        message: str,
        choices: List[str],
    ) -> Optional[str]:
        """Prompt for selection from choices."""
        print(message)
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        
        try:
            selection = int(input("Enter number: ").strip())
            if 1 <= selection <= len(choices):
                return choices[selection - 1]
        except ValueError:
            pass
        
        self.error("Invalid selection")
        return None
    
    # Built-in command handlers
    async def _cmd_help(self, command: str = None) -> None:
        """Show help."""
        if command:
            cmd = self.commands.get(command)
            if not cmd:
                self.error(f"Unknown command: {command}")
                return
            
            self.header(f"Help: {cmd.name}")
            print(f"  {cmd.description}")
            
            if cmd.aliases:
                print(f"\n  Aliases: {', '.join(cmd.aliases)}")
            
            if cmd.args:
                print("\n  Arguments:")
                for arg in cmd.args:
                    required = "(required)" if arg.required else "(optional)"
                    default = f"[default: {arg.default}]" if arg.default else ""
                    print(f"    {arg.name}: {arg.description} {required} {default}")
        else:
            self.header("Available Commands")
            
            # Group by category
            categories: Dict[str, List[Command]] = {}
            for cmd in self.commands.values():
                if cmd.hidden:
                    continue
                if cmd.category not in categories:
                    categories[cmd.category] = []
                categories[cmd.category].append(cmd)
            
            for category, commands in sorted(categories.items()):
                print(colorize(f"\n  {category.upper()}", OutputStyle.BOLD))
                for cmd in sorted(commands, key=lambda c: c.name):
                    aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                    print(f"    {cmd.name:<15} {cmd.description}{aliases}")
    
    async def _cmd_version(self) -> None:
        """Show version."""
        print(f"{self.name} version {self.version}")
    
    async def _cmd_exit(self) -> str:
        """Exit CLI."""
        return "EXIT"


# Factory function
def create_cli() -> InteractiveCLI:
    """Create a CLI instance with King AI commands."""
    cli = InteractiveCLI()
    
    # Register business commands
    cli.register_command(
        name="status",
        description="Show system status",
        handler=lambda: cli.info("System is running normally"),
        category="system",
    )
    
    cli.register_command(
        name="agents",
        description="List available agents",
        handler=lambda: cli.table(
            ["Agent", "Status", "Type"],
            [
                ["Research Agent", "Active", "Research"],
                ["Finance Agent", "Active", "Finance"],
                ["Legal Agent", "Active", "Legal"],
                ["Content Agent", "Active", "Content"],
            ]
        ),
        category="agents",
    )
    
    return cli


# Global CLI instance
_cli: Optional[InteractiveCLI] = None


def get_cli() -> InteractiveCLI:
    """Get the global CLI instance."""
    global _cli
    if _cli is None:
        _cli = create_cli()
    return _cli

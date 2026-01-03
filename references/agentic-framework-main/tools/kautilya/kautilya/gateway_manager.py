"""
MCP Gateway Lifecycle Manager for Kautilya.

Module: kautilya/gateway_manager.py

Automatically manages the MCP Gateway process:
- Checks if gateway is running
- Starts it automatically if needed
- Manages background process
"""

import os
import sys
import time
import subprocess
import signal
import atexit
from pathlib import Path
from typing import Optional
import httpx
from rich.console import Console

console = Console()

# Global process handle
_gateway_process: Optional[subprocess.Popen] = None


class GatewayManager:
    """Manages MCP Gateway lifecycle automatically."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:8080",
        auto_start: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize gateway manager.

        Args:
            gateway_url: Gateway base URL
            auto_start: Automatically start gateway if not running
            verbose: Show detailed startup messages
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.auto_start = auto_start
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None

        # Find project root (where mcp-gateway directory is)
        self.project_root = self._find_project_root()
        self.gateway_dir = self.project_root / "mcp-gateway"

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        # Start from current file location
        current = Path(__file__).resolve().parent

        # Look for mcp-gateway directory by going up
        for _ in range(5):  # Check up to 5 levels up
            if (current / "mcp-gateway").exists():
                return current
            current = current.parent

        # Fallback: assume typical structure
        # tools/kautilya -> go up 2 levels
        return Path(__file__).resolve().parent.parent.parent

    def is_running(self) -> bool:
        """Check if MCP Gateway is running and healthy."""
        try:
            response = httpx.get(f"{self.gateway_url}/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _load_env_file(self) -> dict:
        """Load environment variables from .env file."""
        env_vars = {}
        env_file = self.project_root / ".env"

        if env_file.exists():
            try:
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith("#"):
                            # Handle KEY=value format
                            if "=" in line:
                                key, value = line.split("=", 1)
                                key = key.strip()
                                value = value.strip()
                                # Remove quotes if present
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                elif value.startswith("'") and value.endswith("'"):
                                    value = value[1:-1]
                                env_vars[key] = value
            except Exception as e:
                if self.verbose:
                    console.print(f"[yellow]Warning: Could not load .env: {e}[/yellow]")

        return env_vars

    def start(self) -> bool:
        """
        Start the MCP Gateway in the background.

        Returns:
            True if started successfully or already running
        """
        global _gateway_process

        # Check if already running
        if self.is_running():
            if self.verbose:
                console.print("[dim]MCP Gateway already running[/dim]")
            return True

        if not self.auto_start:
            console.print(
                "[yellow]MCP Gateway not running and auto_start=False[/yellow]"
            )
            return False

        # Ensure gateway directory exists
        if not self.gateway_dir.exists():
            console.print(
                f"[red]MCP Gateway directory not found:[/red] {self.gateway_dir}"
            )
            return False

        if self.verbose:
            console.print("[dim]Starting MCP Gateway in background...[/dim]")

        try:
            # Prepare environment - load from .env file
            env = os.environ.copy()

            # Load .env file
            env_from_file = self._load_env_file()
            env.update(env_from_file)

            # Ensure JWT_SECRET_KEY is set
            if "JWT_SECRET_KEY" not in env:
                env["JWT_SECRET_KEY"] = "dev-secret-key-for-local-development-change-in-production"

            # Find Python executable (prefer venv)
            python_executable = self._find_python_executable()

            # Start gateway process
            self.process = subprocess.Popen(
                [
                    python_executable,
                    "-m",
                    "uvicorn",
                    "service.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8080",
                ],
                cwd=str(self.gateway_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent
            )

            _gateway_process = self.process

            # Register cleanup on exit
            atexit.register(self._cleanup_on_exit)

            # Wait for gateway to be ready (max 10 seconds)
            max_wait = 10
            for i in range(max_wait):
                time.sleep(1)
                if self.is_running():
                    if self.verbose:
                        console.print(
                            "[green]✓[/green] MCP Gateway started successfully"
                        )
                    return True

                # Check if process died
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                    console.print(
                        f"[red]MCP Gateway failed to start:[/red]\n{stderr[:500]}"
                    )
                    return False

            console.print("[yellow]MCP Gateway starting (may take a moment)...[/yellow]")
            return True

        except Exception as e:
            console.print(f"[red]Failed to start MCP Gateway:[/red] {str(e)}")
            return False

    def _find_python_executable(self) -> str:
        """Find the best Python executable to use."""
        # Check for venv in project root
        venv_paths = [
            self.project_root / ".venv" / "bin" / "python",
            self.project_root / "venv" / "bin" / "python",
            Path.home() / ".local" / "share" / "uv" / "python",  # uv managed Python
        ]

        for venv_python in venv_paths:
            if venv_python.exists():
                return str(venv_python)

        # Fallback to system Python
        return sys.executable

    def _cleanup_on_exit(self):
        """Cleanup gateway process on exit (if started by us)."""
        global _gateway_process

        if _gateway_process and _gateway_process.poll() is None:
            try:
                # Send SIGTERM for graceful shutdown
                os.killpg(os.getpgid(_gateway_process.pid), signal.SIGTERM)
                _gateway_process.wait(timeout=5)
            except Exception:
                pass

    def ensure_running(self) -> bool:
        """
        Ensure gateway is running, starting it if necessary.

        Returns:
            True if gateway is running or started successfully
        """
        if self.is_running():
            return True

        if self.auto_start:
            return self.start()

        return False

    def stop(self):
        """Stop the gateway process if we started it."""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
                if self.verbose:
                    console.print("[dim]MCP Gateway stopped[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to stop gateway:[/yellow] {str(e)}")


# Singleton instance
_gateway_manager: Optional[GatewayManager] = None


def get_gateway_manager(
    gateway_url: str = "http://localhost:8080",
    auto_start: bool = True,
    verbose: bool = False,
) -> GatewayManager:
    """
    Get or create the global gateway manager instance.

    Args:
        gateway_url: Gateway base URL
        auto_start: Automatically start gateway if not running
        verbose: Show detailed startup messages

    Returns:
        GatewayManager instance
    """
    global _gateway_manager

    if _gateway_manager is None:
        _gateway_manager = GatewayManager(
            gateway_url=gateway_url, auto_start=auto_start, verbose=verbose
        )

    return _gateway_manager


def ensure_gateway_running(verbose: bool = False) -> bool:
    """
    Ensure MCP Gateway is running, starting it automatically if needed.

    Args:
        verbose: Show detailed startup messages

    Returns:
        True if gateway is running
    """
    manager = get_gateway_manager(verbose=verbose)
    return manager.ensure_running()

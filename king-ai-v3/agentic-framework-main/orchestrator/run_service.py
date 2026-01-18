#!/usr/bin/env python
"""
Quick start script for the Orchestrator service.

This script performs pre-flight checks and starts the service.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> tuple[int, str, str]:
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def main() -> None:
    """Main entry point."""
    print("=" * 50)
    print("Orchestrator Service Quick Start")
    print("=" * 50)
    print()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check if .env file exists
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    if not env_file.exists():
        print("⚠️  .env file not found.")
        if env_example.exists():
            print(f"Copying from {env_example} to {env_file}...")
            with open(env_example) as f:
                content = f.read()
            with open(env_file, "w") as f:
                f.write(content)
            print("✓ Created .env file")
        print()
        print("Required configuration:")
        print("  - ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("  - Database URLs (POSTGRES_URL, REDIS_URL)")
        print("  - Service URLs for dependent services")
        print()
        input("Press Enter to continue after configuring .env...")

    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import fastapi
        import pydantic
        import anyio
        print("✓ Core dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall dependencies with:")
        print("  uv pip install -r requirements.txt")
        sys.exit(1)

    # Optional: Run code quality checks
    print("\nRunning pre-flight checks...")
    print("-" * 50)

    # Check Python version
    version_info = sys.version_info
    if version_info < (3, 11):
        print(f"⚠️  Python {version_info.major}.{version_info.minor} detected.")
        print("   Python 3.11+ is recommended.")
    else:
        print(f"✓ Python {version_info.major}.{version_info.minor}")

    # Check for Black
    returncode, _, _ = run_command("black --version")
    if returncode == 0:
        print("\n1. Formatting check with Black...")
        returncode, stdout, stderr = run_command(
            f"black --check --line-length 100 {script_dir}/service/*.py"
        )
        if returncode == 0:
            print("   ✓ Code formatting OK")
        else:
            print("   ⚠️  Formatting issues found")
            print("   Run: black --line-length 100 orchestrator/")

    # Check for mypy
    returncode, _, _ = run_command("mypy --version")
    if returncode == 0:
        print("\n2. Type checking with mypy...")
        returncode, stdout, stderr = run_command(
            f"mypy --strict {script_dir}/service/*.py"
        )
        if returncode == 0:
            print("   ✓ Type checking passed")
        else:
            print("   ⚠️  Type checking issues found")
            if stderr:
                print(f"   {stderr[:200]}...")

    print("\n" + "=" * 50)
    print("Starting Orchestrator Service")
    print("=" * 50)
    print()
    print("Service will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the service")
    print()

    # Import and run the service
    try:
        # Add project root to Python path
        sys.path.insert(0, str(project_root))

        # Import the main function
        from orchestrator.service.main import main as service_main

        # Run the service
        service_main()

    except KeyboardInterrupt:
        print("\n\nService stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

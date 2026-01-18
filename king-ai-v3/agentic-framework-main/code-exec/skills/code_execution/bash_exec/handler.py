"""
Bash Execution Skill Handler.

Module: code-exec/skills/prepackaged/code_execution/bash_exec/handler.py

Execute bash commands with timeout protection. Mirrors Claude Code's Bash tool.
"""

import subprocess
import time
import os
from typing import Any, Dict, Optional


MAX_OUTPUT_LENGTH = 30000  # Truncate output if longer


def execute_bash(
    command: str,
    timeout_ms: int = 120000,
    working_directory: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Execute a bash command with timeout protection.

    Args:
        command: Bash command to execute
        timeout_ms: Timeout in milliseconds
        working_directory: Working directory for execution
        environment: Additional environment variables

    Returns:
        Dictionary with execution results
    """
    timeout_seconds = timeout_ms / 1000.0

    # Prepare environment
    env = os.environ.copy()
    if environment:
        env.update(environment)

    # Validate working directory
    cwd = working_directory
    if cwd and not os.path.isdir(cwd):
        return {
            "stdout": "",
            "stderr": f"Working directory does not exist: {cwd}",
            "exit_code": 1,
            "execution_time_ms": 0,
            "timed_out": False,
            "truncated": False,
        }

    start_time = time.time()
    timed_out = False
    truncated = False

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=cwd,
            env=env,
        )

        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or "" if hasattr(e, "stdout") else ""
        stderr = e.stderr or "" if hasattr(e, "stderr") else ""
        stderr += f"\n[Command timed out after {timeout_seconds}s]"
        exit_code = 124  # Standard timeout exit code
        timed_out = True

    except Exception as e:
        stdout = ""
        stderr = str(e)
        exit_code = 1

    execution_time_ms = (time.time() - start_time) * 1000

    # Truncate output if too long
    if len(stdout) > MAX_OUTPUT_LENGTH:
        stdout = stdout[:MAX_OUTPUT_LENGTH] + "\n... [output truncated]"
        truncated = True

    if len(stderr) > MAX_OUTPUT_LENGTH:
        stderr = stderr[:MAX_OUTPUT_LENGTH] + "\n... [output truncated]"
        truncated = True

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "execution_time_ms": round(execution_time_ms, 2),
        "timed_out": timed_out,
        "truncated": truncated,
    }

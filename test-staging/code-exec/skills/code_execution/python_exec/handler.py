"""
Python Execution Skill Handler.

Module: code-exec/skills/prepackaged/code_execution/python_exec/handler.py

Execute Python code with timeout protection.
"""

import subprocess
import sys
import time
import tempfile
import os
from typing import Any, Dict, Optional


MAX_OUTPUT_LENGTH = 30000


def execute_python(
    code: str,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Execute Python code in a subprocess.

    Args:
        code: Python code to execute
        timeout_seconds: Timeout in seconds

    Returns:
        Dictionary with execution results
    """
    start_time = time.time()

    # Create a temporary file for the code
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            temp_file = f.name

        # Execute using subprocess for isolation
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        output = result.stdout
        error = result.stderr
        success = result.returncode == 0

        # Truncate if needed
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... [output truncated]"

        if len(error) > MAX_OUTPUT_LENGTH:
            error = error[:MAX_OUTPUT_LENGTH] + "\n... [output truncated]"

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "output": output,
            "error": error if error else None,
            "return_value": None,
            "execution_time_ms": round(execution_time_ms, 2),
            "success": success,
        }

    except subprocess.TimeoutExpired:
        execution_time_ms = (time.time() - start_time) * 1000
        return {
            "output": "",
            "error": f"Execution timed out after {timeout_seconds} seconds",
            "return_value": None,
            "execution_time_ms": round(execution_time_ms, 2),
            "success": False,
        }

    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        return {
            "output": "",
            "error": str(e),
            "return_value": None,
            "execution_time_ms": round(execution_time_ms, 2),
            "success": False,
        }

    finally:
        # Clean up temp file
        try:
            if "temp_file" in locals():
                os.unlink(temp_file)
        except OSError:
            pass

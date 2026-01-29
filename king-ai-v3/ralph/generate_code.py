#!/usr/bin/env python3
"""
Ralph AI Code Generator
Uses GitHub Copilot CLI to generate code based on prompts.
"""

import sys
import asyncio
import os
import json
import subprocess

async def generate_code(prompt: str) -> str:
    """Generate code using GitHub Copilot CLI"""
    # Add system prompt for code generation
    system = """You are an expert software engineer. Generate high-quality, production-ready code based on the user's requirements. Follow best practices, include proper error handling, and ensure the code is well-documented.

CRITICAL: Output ONLY the code that should be implemented. Do not include explanations, comments about the code, or any text outside of the actual code. The output will be directly executed or applied to files.

If you need to create or modify multiple files, output the changes in the following format:
```
<file_path>
```python
# code for file_path
```

<file_path2>
```python
# code for file_path2
```

Guidelines:
- Keep implementations focused and minimal
- Follow existing codebase patterns
- Include necessary imports
- Handle errors appropriately
- Use async/await where appropriate
- Follow the project's conventions (FastAPI, React, etc.)
"""

    full_prompt = f"{system}\n\n{prompt}"

    try:
        # Run GitHub Copilot CLI
        env = os.environ.copy()
        # Ensure GitHub token is available
        if not ('GITHUB_TOKEN' in env or 'GH_TOKEN' in env or 'COPILOT_GITHUB_TOKEN' in env):
            # Try to get token from gh auth
            try:
                result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, check=True)
                env['GITHUB_TOKEN'] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass  # No token available
        
        process = await asyncio.create_subprocess_exec(
            'copilot', '--allow-all-tools',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        stdout, stderr = await process.communicate(input=full_prompt.encode())

        if process.returncode == 0:
            return stdout.decode().strip()
        else:
            print(f"GitHub Copilot CLI failed: {stderr.decode()}", file=sys.stderr)
            return ""

    except FileNotFoundError:
        print("GitHub Copilot CLI not found. Please install from https://github.com/github/copilot-cli", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Error generating code: {e}", file=sys.stderr)
        return ""

# Commented out Ollama implementation for future use
"""
import sys
import asyncio
import os
import json
sys.path.append('src')

from utils.ollama_client import OllamaClient

async def generate_code_ollama(prompt: str) -> str:
    # Use environment variables or defaults
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')

    client = OllamaClient(base_url, model)

    # Add system prompt for code generation
    system = \"\"\"You are an expert software engineer. Generate high-quality, production-ready code based on the user's requirements. Follow best practices, include proper error handling, and ensure the code is well-documented.

CRITICAL: Output ONLY the code that should be implemented. Do not include explanations, comments about the code, or any text outside of the actual code. The output will be directly executed or applied to files.

If you need to create or modify multiple files, output the changes in the following format:
```
<file_path>
```python
# code for file_path
```

<file_path2>
```python
# code for file_path2
```

Guidelines:
- Keep implementations focused and minimal
- Follow existing codebase patterns
- Include necessary imports
- Handle errors appropriately
- Use async/await where appropriate
- Follow the project's conventions (FastAPI, React, etc.)
\"\"\"

    try:
        response = await client.complete(prompt, system=system, temperature=0.1)
        return response.strip()
    except Exception as e:
        print(f"Error generating code: {e}", file=sys.stderr)
        return ""
"""

if __name__ == "__main__":
    prompt = sys.stdin.read().strip()
    if prompt:
        result = asyncio.run(generate_code(prompt))
        print(result)
    else:
        print("No prompt provided", file=sys.stderr)
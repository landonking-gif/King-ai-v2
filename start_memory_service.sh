#!/bin/bash
# Local Memory Service Start Script

echo "Starting Memory Service locally..."

# Change to memory-service directory
cd /mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main/memory-service

# Set PYTHONPATH
export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"

# Try to use the main venv Python
MAIN_VENV_PYTHON="/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/venv/Scripts/python.exe"

if [ -f "$MAIN_VENV_PYTHON" ]; then
    echo "Using main venv Python..."
    PYTHONPATH="C:\\Users\\dmilner.AGV-040318-PC\\Downloads\\landon\\king-ai-v2\\king-ai-v3\\agentic-framework-main:$PYTHONPATH" "$MAIN_VENV_PYTHON" -m service.main
else
    echo "Main venv not found, using python3..."
    PYTHONPATH="/mnt/c/Users/dmilner.AGV-040318-PC/Downloads/landon/king-ai-v2/king-ai-v3/agentic-framework-main:$PYTHONPATH" python3 -m service.main
fi
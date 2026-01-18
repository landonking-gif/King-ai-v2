#!/bin/bash
# Convenience script to run kautilya with venv activated

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at .venv"
    echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Run kautilya
python -m kautilya.cli "$@"

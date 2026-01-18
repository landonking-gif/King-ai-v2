#!/bin/bash
# Test Subagent Manager service

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run tests
echo "Running Subagent Manager tests..."
pytest subagent-manager/tests/ -v --cov=subagent_manager --cov-report=term-missing

# Run type checking
echo "Running mypy type checks..."
mypy subagent-manager/service/ --ignore-missing-imports

# Run code formatting check
echo "Checking code formatting..."
black --check --line-length 100 subagent-manager/

echo "All checks passed!"

#!/bin/bash
# Quick start script for the Orchestrator service

set -e

echo "=================================="
echo "Orchestrator Service Quick Start"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating it now..."
    uv venv --python 3.11
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ Dependencies not installed. Installing now..."
    uv pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file. Please edit it with your configuration."
    echo ""
    echo "Required configuration:"
    echo "  - ANTHROPIC_API_KEY or OPENAI_API_KEY"
    echo "  - Database URLs (POSTGRES_URL, REDIS_URL)"
    echo "  - Service URLs for dependent services"
    echo ""
    read -p "Press Enter to continue after configuring .env..."
fi

# Run code quality checks
echo ""
echo "Running code quality checks..."
echo "--------------------------------"

echo "1. Formatting with Black..."
black --check --line-length 100 orchestrator/service/*.py || {
    echo "⚠️  Code formatting issues found. Run: black --line-length 100 orchestrator/"
}

echo ""
echo "2. Type checking with mypy..."
mypy --strict orchestrator/service/*.py || {
    echo "⚠️  Type checking issues found. Please review."
}

echo ""
echo "All checks completed!"
echo ""

# Start the service
echo "=================================="
echo "Starting Orchestrator Service"
echo "=================================="
echo ""
echo "Service will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Run the service
python -m orchestrator.service.main

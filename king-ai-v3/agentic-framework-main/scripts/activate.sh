#!/bin/bash
# Activation script for Enterprise Agentic Framework
# Usage: source activate.sh

echo "🚀 Activating Enterprise Agentic Framework environment..."

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Verify activation
echo "✅ Virtual environment activated!"
echo "📦 Python: $(python --version)"
echo "📍 Location: $(which python)"
echo ""
echo "Available commands:"
echo "  kautilya          - Interactive CLI utility"
echo "  pytest            - Run tests"
echo "  black .           - Format code"
echo "  mypy .            - Type check"
echo ""
echo "Services:"
echo "  python -m orchestrator.service.main        - Start orchestrator (port 8000)"
echo "  python -m memory-service.service.main      - Start memory service (port 8002)"
echo "  python -m mcp-gateway.service.main         - Start MCP gateway (port 8080)"
echo ""
echo "Or use: docker-compose up -d"
echo ""
echo "📖 Docs: See INSTALLATION_COMPLETE.md for full details"

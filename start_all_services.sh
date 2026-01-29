#!/bin/bash
# King AI v3 + MoltBot Unified Startup Script
# This script starts all King AI services including:
# - Ollama (LLM runtime)
# - King AI Orchestrator (port 8000)
# - Memory Service (port 8002)
# - MoltBot Gateway (port 18789)

set -e

echo "====================================="
echo "King AI v3 + MoltBot Startup Script"
echo "====================================="
echo ""

KINGAI_DIR="/home/ubuntu/king-ai-v3/agentic-framework-main"
MOLTBOT_DIR="/home/ubuntu/king-ai-v3/moltbot"

# Function to check if a port is in use
check_port() {
    nc -z localhost $1 2>/dev/null
    return $?
}

# Function to wait for service
wait_for_service() {
    local port=$1
    local name=$2
    local max_wait=$3
    local count=0
    
    echo -n "Waiting for $name (port $port)..."
    while ! check_port $port; do
        sleep 1
        count=$((count + 1))
        if [ $count -ge $max_wait ]; then
            echo " TIMEOUT!"
            return 1
        fi
        echo -n "."
    done
    echo " OK!"
    return 0
}

# 1. Start Ollama if not running
echo "[1/4] Checking Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi
if check_port 11434; then
    echo "Ollama is running on port 11434"
else
    echo "WARNING: Ollama may not be running properly"
fi

# 2. Start King AI Orchestrator
echo ""
echo "[2/4] Starting King AI Orchestrator..."
if check_port 8000; then
    echo "Orchestrator already running on port 8000"
else
    cd "$KINGAI_DIR"
    source venv/bin/activate
    nohup python -m uvicorn orchestrator.service.main:app --host 0.0.0.0 --port 8000 > /tmp/orchestrator.log 2>&1 &
    wait_for_service 8000 "Orchestrator" 30
fi

# 3. Start Memory Service
echo ""
echo "[3/4] Starting Memory Service..."
if check_port 8002; then
    echo "Memory Service already running on port 8002"
else
    cd "$KINGAI_DIR"
    source venv/bin/activate
    nohup python -m uvicorn memory_service.main:app --host 0.0.0.0 --port 8002 > /tmp/memory-service.log 2>&1 &
    wait_for_service 8002 "Memory Service" 30
fi

# 4. Start MoltBot Gateway
echo ""
echo "[4/4] Starting MoltBot Gateway..."
if check_port 18789; then
    echo "MoltBot already running on port 18789"
else
    cd "$MOLTBOT_DIR"
    nohup pnpm moltbot gateway --port 18789 > /tmp/moltbot.log 2>&1 &
    wait_for_service 18789 "MoltBot" 30
fi

echo ""
echo "====================================="
echo "All Services Started!"
echo "====================================="
echo ""
echo "Service Status:"
echo "  - Ollama (LLM):       http://localhost:11434"
echo "  - King AI Orchestrator: http://localhost:8000"
echo "  - Memory Service:     http://localhost:8002"
echo "  - MoltBot Gateway:    http://localhost:18789"
echo ""
echo "External Access (via AWS Public IP):"
echo "  - Dashboard:          http://100.24.50.240:8000"
echo "  - MoltBot Control:    http://100.24.50.240:18789"
echo ""
echo "LLM Model: DeepSeek R1 7B (deepseek-r1:7b)"
echo ""
echo "Log Files:"
echo "  - Ollama:       /tmp/ollama.log"
echo "  - Orchestrator: /tmp/orchestrator.log"
echo "  - Memory:       /tmp/memory-service.log"
echo "  - MoltBot:      /tmp/moltbot.log"
echo ""
echo "Multi-Channel Support (configure in ~/.moltbot/moltbot.json):"
echo "  - Telegram: Set TELEGRAM_BOT_TOKEN and restart MoltBot"
echo "  - Discord:  Set DISCORD_BOT_TOKEN and restart MoltBot"
echo "  - Slack:    Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN, restart MoltBot"
echo "  - WhatsApp: Run 'cd /home/ubuntu/king-ai-v3/moltbot && pnpm moltbot whatsapp pair'"
echo "  - Signal:   Install signal-cli and configure"
echo ""
echo "Quick Tests:"
echo "  # Test chat"
echo "  curl -X POST http://localhost:8000/api/chat -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"Hello\", \"user_id\": \"test\"}'"
echo ""
echo "  # Test OpenAI endpoint (used by MoltBot)"
echo "  curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"deepseek-r1\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}'"
echo ""
echo "Documentation:"
echo "  - MoltBot Integration: See MOLTBOT_INTEGRATION.md"
echo "  - Deployment Status: See DEPLOYMENT_STATUS.md"
echo "  - Full Guide: See USER_GUIDE.md"
echo ""

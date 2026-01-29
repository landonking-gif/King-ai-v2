#!/bin/bash
# Full King AI Orchestrator Setup
# This deploys the complete system with database, Redis, and all services

set -e

echo "=== King AI Full Orchestrator Setup ==="

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create king-ai directory
cd /home/ubuntu
mkdir -p king-ai
cd king-ai

# Create docker-compose.yml for infrastructure
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=king
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=kingai
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U king -d kingai"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
EOF

# Start infrastructure
echo "Starting PostgreSQL and Redis..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 10

# Check services
docker-compose ps

# Create .env file for King AI
cat > .env << 'EOF'
DATABASE_URL=postgresql+asyncpg://king:password@localhost:5432/kingai
REDIS_URL=redis://localhost:6379
VLLM_URL=http://localhost:8005
VLLM_MODEL=casperhansen/deepseek-r1-distill-qwen-7b-awq
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
RISK_PROFILE=moderate
EOF

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --quiet sqlalchemy asyncpg redis fastapi uvicorn pydantic pydantic-settings httpx aiohttp python-multipart

# Check database connection
echo "Testing database connection..."
python3 -c "
import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect('postgresql://king:password@localhost:5432/kingai')
        print('Database connected successfully!')
        await conn.close()
    except Exception as e:
        print(f'Database connection failed: {e}')
        
asyncio.run(test())
"

# Check Redis connection
echo "Testing Redis connection..."
python3 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print('Redis connected successfully!')
except Exception as e:
    print(f'Redis connection failed: {e}')
"

echo ""
echo "=== Infrastructure Ready ==="
echo "PostgreSQL: localhost:5432"
echo "Redis: localhost:6379"
echo "vLLM: localhost:8005"
echo ""
echo "Next: Upload and start the King AI orchestrator"

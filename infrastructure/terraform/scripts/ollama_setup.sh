#!/bin/bash
set -e

# Ollama and vLLM Setup Script for King AI v2
# This script runs on GPU instance boot to configure the inference stack

LOG_FILE="/var/log/ollama_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting Ollama Setup $(date) ==="

# Update system
apt-get update -y
apt-get upgrade -y

# Install dependencies
apt-get install -y curl wget git python3-pip python3-venv nvidia-cuda-toolkit jq

# Verify NVIDIA drivers
echo "Checking NVIDIA drivers..."
nvidia-smi || {
    echo "ERROR: NVIDIA drivers not detected"
    exit 1
}

# Get instance metadata for configuration
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

echo "Instance: $INSTANCE_ID in $REGION"

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to listen on all interfaces (for internal VPC access)
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_NUM_PARALLEL=4"
EOF

# Start Ollama service
systemctl daemon-reload
systemctl enable ollama
systemctl start ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Pull the configured model (passed as user data or default)
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:70b}"
echo "Pulling model: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

# Install vLLM for high-throughput batching
echo "Installing vLLM..."
pip3 install --upgrade pip
pip3 install vllm torch transformers accelerate

# Get HuggingFace token from Secrets Manager
HF_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id king-ai/huggingface-token \
    --query SecretString \
    --output text \
    --region $REGION 2>/dev/null || echo "")

# Create vLLM service
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Meta-Llama-3.1-70B-Instruct}"
cat > /etc/systemd/system/vllm.service << EOF
[Unit]
Description=vLLM Inference Server
After=network.target ollama.service

[Service]
Type=simple
User=root
Environment="HF_TOKEN=${HF_TOKEN}"
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
ExecStart=/usr/local/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
Restart=always
RestartSec=10
StandardOutput=append:/var/log/vllm.log
StandardError=append:/var/log/vllm.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vllm

# Only start vLLM if HF token is available
if [ -n "$HF_TOKEN" ]; then
    systemctl start vllm
    echo "vLLM service started"
else
    echo "WARNING: HuggingFace token not found, vLLM not started"
fi

# Create health check script
cat > /usr/local/bin/health_check.sh << 'HEALTHEOF'
#!/bin/bash
echo "=== King AI Inference Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check Ollama
echo -n "Ollama: "
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "OK"
    echo "  Models: $(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | tr '\n' ', ')"
else
    echo "FAILED"
fi

# Check vLLM
echo -n "vLLM: "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "OK"
else
    if systemctl is-active --quiet vllm; then
        echo "STARTING"
    else
        echo "NOT RUNNING"
    fi
fi

# GPU Status
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
HEALTHEOF
chmod +x /usr/local/bin/health_check.sh

# Create inference queue consumer script
cat > /usr/local/bin/queue_consumer.py << 'QUEUEEOF'
#!/usr/bin/env python3
"""SQS Queue Consumer for Ollama Inference Requests"""

import asyncio
import json
import os
import httpx
import boto3
from datetime import datetime

QUEUE_URL = os.environ.get("INFERENCE_QUEUE_URL", "")
OLLAMA_URL = "http://localhost:11434"
REGION = os.environ.get("AWS_REGION", "us-east-1")

sqs = boto3.client("sqs", region_name=REGION)

async def process_message(message: dict):
    """Process a single inference request from the queue."""
    try:
        body = json.loads(message["Body"])
        
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": body.get("model", "llama3.1:70b"),
                    "prompt": body.get("prompt", ""),
                    "stream": False,
                    "options": body.get("options", {})
                }
            )
            result = response.json()
        
        # Send response to callback URL if provided
        if callback_url := body.get("callback_url"):
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json={
                    "request_id": body.get("request_id"),
                    "response": result.get("response"),
                    "completed_at": datetime.utcnow().isoformat()
                })
        
        return True
    except Exception as e:
        print(f"Error processing message: {e}")
        return False

async def main():
    """Main consumer loop."""
    print(f"Starting queue consumer for {QUEUE_URL}")
    
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=20
            )
            
            for message in response.get("Messages", []):
                success = await process_message(message)
                if success:
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=message["ReceiptHandle"]
                    )
        except Exception as e:
            print(f"Queue error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
QUEUEEOF
chmod +x /usr/local/bin/queue_consumer.py

# Install CloudWatch agent for monitoring
echo "Installing CloudWatch agent..."
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Configure CloudWatch agent for GPU metrics
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CWEOF'
{
    "metrics": {
        "namespace": "KingAI/Inference",
        "metrics_collected": {
            "nvidia_gpu": {
                "measurement": [
                    "utilization_gpu",
                    "utilization_memory",
                    "memory_total",
                    "memory_used",
                    "temperature_gpu"
                ],
                "metrics_collection_interval": 60
            }
        },
        "append_dimensions": {
            "InstanceId": "${aws:InstanceId}"
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/ollama_setup.log",
                        "log_group_name": "/king-ai/inference",
                        "log_stream_name": "{instance_id}/setup"
                    },
                    {
                        "file_path": "/var/log/vllm.log",
                        "log_group_name": "/king-ai/inference",
                        "log_stream_name": "{instance_id}/vllm"
                    }
                ]
            }
        }
    }
}
CWEOF

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

echo "=== Ollama Setup Complete $(date) ==="
echo "Run '/usr/local/bin/health_check.sh' to verify status"

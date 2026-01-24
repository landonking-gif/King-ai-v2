# Force rebuild cache invalidation - 2026-01-23 23:17
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY subagent-manager /app/subagent-manager
COPY adapters /app/adapters
COPY docs /app/docs

# Set Python path
ENV PYTHONPATH=/app
ENV FORCE_REBUILD=2026-01-23-23-15

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Run the application (updated for cache invalidation - 2026-01-23 23:15)
CMD ["python", "-m", "service.main"]

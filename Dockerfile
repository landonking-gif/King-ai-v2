# Build stage
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements
COPY pyproject.toml .
# Install dependencies into a separate directory
RUN pip install --no-cache-dir --prefix=/install .

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies (only libpq for postgres)
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY cli.py .
COPY alembic.ini .
COPY .env.example .env

# Expose API port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

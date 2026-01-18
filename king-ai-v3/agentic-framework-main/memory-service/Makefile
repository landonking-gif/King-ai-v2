# Memory Service Makefile
# Provides common development commands

.PHONY: help install dev test lint format typecheck clean docker-up docker-down run

help:
	@echo "Memory Service - Available Commands"
	@echo "===================================="
	@echo "make install      - Install dependencies"
	@echo "make dev          - Run service in development mode"
	@echo "make test         - Run tests"
	@echo "make test-cov     - Run tests with coverage"
	@echo "make lint         - Run linters"
	@echo "make format       - Format code"
	@echo "make typecheck    - Run type checking"
	@echo "make docker-up    - Start infrastructure (Redis, Postgres, MinIO)"
	@echo "make docker-down  - Stop infrastructure"
	@echo "make clean        - Clean temporary files"
	@echo "make run          - Run production server"

install:
	uv pip install -r requirements.txt

dev:
	python -m uvicorn service.main:app --reload --host 0.0.0.0 --port 8001

run:
	python -m uvicorn service.main:app --host 0.0.0.0 --port 8001 --workers 4

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=service --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

test-unit:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

lint:
	ruff check service/ tests/

format:
	black service/ tests/ examples/ --line-length 100

typecheck:
	mypy service/ --strict

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@docker-compose ps

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "Cleaned temporary files"

check: format lint typecheck test-unit
	@echo "All checks passed!"

init-db:
	@echo "Initializing database..."
	python -c "import asyncio; from service.storage.postgres import PostgresAdapter; from service.config import settings; asyncio.run(PostgresAdapter(settings).init_db())"
	@echo "Database initialized"

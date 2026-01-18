#!/bin/bash
# Source this script to load environment variables from .env
# Usage: source load_env.sh

if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
    echo "✓ Environment variables loaded from .env"
else
    echo "⚠ .env file not found"
fi

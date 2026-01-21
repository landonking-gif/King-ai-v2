#!/bin/bash

# Master Control Panel Run Script

echo "Starting King AI v3 Master Control Panel..."

# Check if virtual environment exists
if [ -d "../../venv" ]; then
    source ../../venv/bin/activate
fi

# Install dependencies if needed
pip install -r requirements.txt

# Run the FastAPI server
python main.py
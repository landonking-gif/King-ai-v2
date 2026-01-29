#!/usr/bin/env python3
"""Test the full orchestrator API"""
import requests

BASE_URL = "http://localhost:8000"

# Test health
r = requests.get(f"{BASE_URL}/api/health")
print(f"Health: {r.status_code} - {r.json()}")

# Test chat
r = requests.post(f"{BASE_URL}/api/chat/", json={"message": "Hello, what can you do?"})
print(f"Chat: {r.status_code}")
print(f"Response: {r.text[:1000]}")

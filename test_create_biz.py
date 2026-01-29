#!/usr/bin/env python3
"""Test the full orchestrator API with business creation"""
import requests

BASE_URL = "http://localhost:8000"

# Test creating a business
r = requests.post(f"{BASE_URL}/api/chat/", json={"message": "Create a dropshipping business called TestShop selling electronics"})
print(f"Create Business: {r.status_code}")
print(f"Response:\n{r.text}")

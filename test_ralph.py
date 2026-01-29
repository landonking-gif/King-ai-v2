#!/usr/bin/env python3
"""Test Ralph Loop via chat"""
import requests

r = requests.post("http://localhost:8000/api/chat/", json={"message": "run the ralph loop"})
print(f"Status: {r.status_code}")
print(f"Response:\n{r.json()}")

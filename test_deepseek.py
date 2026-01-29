#!/usr/bin/env python3
import requests

response = requests.post(
    'http://localhost:8000/api/chat',
    json={'message': 'Hello, what AI model are you? Please identify yourself.', 'user_id': 'test'}
)
print(response.text[:2000])

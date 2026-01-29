#!/usr/bin/env python3
"""Comprehensive King AI system verification."""
import requests

BASE = 'http://localhost:8000'
results = []

# Test 1: Health
print("Testing health endpoint...")
r = requests.get(f'{BASE}/api/health', timeout=10)
results.append(('Health', r.json().get('status') == 'healthy'))

# Test 2: Chat
print("Testing chat endpoint...")
r = requests.post(f'{BASE}/api/chat', json={'message': 'hello'}, timeout=30)
results.append(('Chat', len(r.json().get('response', '')) > 0))

# Test 3: Clean response (no think tags)
print("Testing clean response...")
r = requests.post(f'{BASE}/api/chat', json={'message': 'what can you do?'}, timeout=60)
results.append(('Clean Response', '<think>' not in r.json().get('response', '')))

# Test 4: Story status
print("Testing story status...")
r = requests.post(f'{BASE}/api/chat', json={'message': 'what story is running?'}, timeout=60)
resp = r.json().get('response', '').lower()
results.append(('Story Status', 'ralph' in resp or 'loop' in resp or 'running' in resp))

# Test 5: Conversation history
print("Testing conversation history...")
r = requests.get(f'{BASE}/api/conversation-history', timeout=30)
results.append(('DB History', len(r.json()) > 0))

# Test 6: Agents list
print("Testing agents list...")
r = requests.post(f'{BASE}/api/chat', json={'message': 'list agents'}, timeout=60)
resp = r.json().get('response', '').lower()
results.append(('Agent List', 'research' in resp or 'agent' in resp))

print('='*50)
print('KING AI VERIFICATION RESULTS')
print('='*50)
for name, passed in results:
    status = 'PASS' if passed else 'FAIL'
    print(f'{name}: {status}')
print('='*50)
all_passed = all(p for _, p in results)
print('OVERALL: ' + ('ALL PASSED' if all_passed else 'SOME FAILED'))

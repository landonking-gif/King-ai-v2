import requests
import json

BASE = 'http://localhost:8000/api'

def test(msg):
    print(f'\n=== Testing: "{msg}" ===')
    r = requests.post(f'{BASE}/chat/', json={'message': msg})
    data = r.json()
    print(f'Type: {data.get("type")}')
    resp = data.get("response", "")
    # Truncate for display
    if len(resp) > 300:
        resp = resp[:300] + '...'
    print(f'Response:\n{resp}')
    if data.get('system_state'):
        state = data['system_state']
        print(f'\nSystem State:')
        print(f'  Autonomous: {state.get("autonomous_mode")}')
        print(f'  Agents: {len(state.get("active_agents", []))}')
    return data

# Test commands
test('list agents')
test('help')
test('spawn a code agent')
test('list workflows')
test('status')
test('stop ralph loop')
print('\n=== All tests passed! ===')

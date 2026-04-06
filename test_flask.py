import sys; sys.path.insert(0,'.')
import json
from app import app

client = app.test_client()
print('Testing /health...')
r1 = client.get('/health')
print(r1.status_code, r1.get_json())
assert r1.status_code == 200

print('\nTesting /tasks...')
r2 = client.get('/tasks')
print(r2.status_code, [t['task_id'] for t in r2.get_json()])
assert r2.status_code == 200

print('\nTesting /reset/easy...')
r3 = client.post('/reset/easy')
print(r3.status_code, 'tick:', r3.get_json()['tick'])
assert r3.status_code == 200

print('\nTesting /step for easy (RestartService on web-3)...')
r4 = client.post('/step', json={'action_type': 'RestartService', 'target_id': 'web-3'})
data4 = r4.get_json()
print(r4.status_code)
print('score:', data4['reward']['score'])
print('feedback:', data4['reward']['feedback'])
assert r4.status_code == 200
assert data4['reward']['score'] == 1.0

print('\nFLASK ENDPOINTS OK')

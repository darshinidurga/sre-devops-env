from fastapi.testclient import TestClient
from app import app
import sys

client = TestClient(app)

print("Checking /health...")
r = client.get("/health")
assert r.status_code == 200, r.text

print("Checking /tasks...")
r = client.get("/tasks")
assert r.status_code == 200, r.text

print("Checking /reset/easy...")
r = client.post("/reset/easy")
assert r.status_code == 200, r.text

print("Checking /step (easy task)...")
r = client.post("/step", json={"action_type": "RestartService", "target_id": "web-3"})
assert r.status_code == 200, r.text
data = r.json()
assert data["reward"]["score"] == 1.0

print("App FASTAPI checks PASSED")

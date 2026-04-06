---
title: SRE DevOps Simulator
emoji: 🖥️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# SRE DevOps Simulator — OpenEnv Environment

## Description
An OpenEnv environment that simulates a company's cloud 
infrastructure. The AI agent acts as an on-call Site 
Reliability Engineer (SRE) that must diagnose and fix 
server incidents without taking the site offline.

## Real-World Motivation
Companies spend millions on cloud infrastructure. Training
AI to automatically diagnose and mitigate server outages
is a critical problem in enterprise tech.

## Environment Details

### Action Space
| Action | Description |
|--------|-------------|
| RestartService(server_id) | Restart a crashed server |
| ScaleUp(resource_type) | Add capacity for traffic |
| ScaleDown(resource_type) | Reduce capacity |
| RollbackDeployment(version) | Revert bad deployment |
| KillProcess(server_id) | Kill runaway process |
| FlushCache(cache_id) | Clear cache server |
| FailoverDatabase(replica_id) | Switch to replica DB |
| InvestigateLog(server_id) | Read detailed logs |

### Observation Space
| Field | Description |
|-------|-------------|
| tick | Current simulation tick |
| servers | Dict of server metrics (cpu, ram, status) |
| alerts | List of active alerts with severity |
| logs | Recent log entries |
| deployment_history | List of deployments |
| active_connections | Total cluster connections |
| site_uptime | Whether site is up |
| downtime_ticks | Ticks site has been down |

## Tasks

### Easy — "The Dead Server"
- **Difficulty:** Easy
- **Max Ticks:** 10
- **Scenario:** web-3 has crashed. Two red herring alerts 
  are firing. Agent must restart the correct server.
- **Expected Score:** 1.0 for correct first action

### Medium — "Traffic Tsunami"  
- **Difficulty:** Medium
- **Max Ticks:** 15
- **Scenario:** Black Friday traffic surge. API gateways 
  failing. Agent must scale up before database crashes.
- **Expected Score:** 0.8 for correct scaling actions

### Hard — "The Silent Killer"
- **Difficulty:** Hard
- **Max Ticks:** 15
- **Scenario:** Bad deployment v2.3.1 causing memory leak.
  CPU looks normal (red herring). Agent must investigate,
  rollback, and restart affected servers.
- **Expected Score:** 0.9 for full correct sequence

## Baseline Scores
| Task | Score |
|------|-------|
| easy | 1.00 |
| medium | 0.80 |
| hard | 0.30 |
| **Average** | **0.70** |

## Setup Instructions

### Prerequisites
```bash
pip install fastapi uvicorn pydantic openai requests
```

### Run Locally
```bash
# Start environment server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline inference (in new terminal)
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_hf_token_here
python inference.py
```

### Docker
```bash
docker build -t sre-devops-env .
docker run -p 7860:7860 sre-devops-env
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset/{task_id} | POST | Reset environment |
| /step | POST | Take action |
| /state | GET | Get current state |
| /tasks | GET | List all tasks |
| /docs | GET | API documentation |

## Environment Variables
| Variable | Description |
|----------|-------------|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN | Hugging Face API key |
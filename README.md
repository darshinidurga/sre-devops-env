---
title: SRE DevOps Simulator
emoji: 🖥️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - devops
  - reinforcement-learning
  - agents
---

# 🌐 SRE DevOps Simulator — OpenEnv Environment

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen.svg)]()
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)]()

## 📖 The Problem: AI in Enterprise Infrastructure
Modern enterprises lose millions of dollars for every hour their cloud infrastructure is offline. While Human Site Reliability Engineers (SREs) use complex reasoning to diagnose logs, ignore "red herring" alerts, and safely mitigate production outages, current AI agents lack a safe, deterministic sandbox to practice these high-stakes skills. 

We cannot train frontier models on live production databases. **They need a simulator.**

## 🎯 The Solution: SRE DevOps Simulator
This OpenEnv-compliant environment acts as a highly realistic, mathematically rigorous training ground for agentic AI. The AI model acts as the on-call SRE. It is given simulated server metrics, active alerts, and deployment histories, and must execute exact terminal-like commands to restore site stability without taking the infrastructure offline or bankrupting the company by over-provisioning servers.

This environment specifically evaluates an LLM's ability to:
1. **Filter Noise:** Ignore red herring alerts and focus on root causes.
2. **Sequential Reasoning:** Use tools like `InvestigateLog` to gather context before taking destructive actions like `RollbackDeployment`.
3. **Resource Management:** Balance scaling infrastructure to meet traffic demands without triggering spam/over-provisioning penalties.

---

## 🏗️ Environment Architecture

### The Action Space
The agent has access to 8 deterministic tools. Penalties are strictly enforced for redundant or illogical actions (e.g., attempting to rollback a deployment twice, or scaling a server already at maximum capacity).

| Action | Description | Complexity |
|--------|-------------|------------|
| `RestartService(server_id)` | Reboots a crashed node. | Low |
| `ScaleUp(resource_type)` | Provisions additional capacity for traffic spikes. | Medium |
| `ScaleDown(resource_type)` | Reduces capacity to save costs. | Medium |
| `RollbackDeployment(version)` | Reverts a bad code push to a stable state. | High |
| `KillProcess(server_id)` | Terminates a runaway process consuming CPU. | Low |
| `FlushCache(cache_id)` | Clears overloaded cache memory. | Low |
| `FailoverDatabase(replica_id)` | Promotes a replica DB if the primary fails. | High |
| `InvestigateLog(server_id)` | Reads detailed stdout/stderr logs for root cause analysis. | High |

### The Observation Space
At each `step()`, the agent receives a rich, JSON-structured state representing the live cluster:
* **`servers`**: Dict of real-time metrics (CPU %, RAM %, status, capacity).
* **`alerts`**: Active PagerDuty-style alerts (contains intentional noise/red herrings).
* **`logs`**: Recent system events.
* **`site_uptime`**: Boolean indicating if the customer-facing site is currently resolving.
* **`active_connections`**: Total cluster load, used for scaling scenarios.

---

## 🏆 Evaluation Tasks

The simulator includes three distinct scenarios designed to test different axes of agentic reasoning.

### 🟢 Easy: "The Dead Server"
* **Scenario:** `web-3` has crashed abruptly. Two critical but irrelevant "red herring" alerts are firing simultaneously.
* **Agent Objective:** Parse the noisy observation state, identify the genuinely dead server, and issue a targeted `RestartService` command.
* **Evaluation Focus:** Basic tool use and noise filtration.

### 🟡 Medium: "Traffic Tsunami"  
* **Scenario:** A massive traffic surge (e.g., Black Friday) is overwhelming the API gateways.
* **Agent Objective:** The agent must sequentially `ScaleUp` the infrastructure to meet the connection demand before the database crashes. 
* **Evaluation Focus:** The environment strictly penalizes "action spamming." The agent must scale appropriately without over-provisioning or blindly repeating commands.

### 🔴 Hard: "The Silent Killer"
* **Scenario:** A recent code deployment (`v2.3.1`) introduced a memory leak. CPU metrics look completely normal (a trap for naive models). 
* **Agent Objective:** The agent must recognize the RAM anomaly, use `InvestigateLog` to confirm the bad deployment, and successfully execute `RollbackDeployment(v2.3.0)`.
* **Evaluation Focus:** Multi-step reasoning, root cause analysis, and utilizing discovery tools before execution tools.

---

## 📊 Baseline Performance
This environment was tested using **Qwen/Qwen2.5-72B-Instruct** via the Hugging Face Serverless API. 

| Task | Final Score | Notes |
|------|-------|-------|
| **Easy** | `1.00` | Model correctly identified and restarted `web-3` on Tick 1. |
| **Medium** | `0.80` | Model successfully scaled infrastructure but struggled with optimal capacity limits. |
| **Hard** | `0.70` | Model identified the leak and initiated rollback, showing strong partial reasoning. |
| **Average** | **`0.83`** | *Validates environment provides meaningful, variable reward signals.* |

---

## ⚙️ Setup & Validation

### Prerequisites
```bash
pip install fastapi uvicorn pydantic openai requests openenv-core uv

##Run Locally (Development)

# Generate lock file
uv lock

# Start the OpenEnv server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In a separate terminal, run the agent:
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_hf_token
python inference.py

##docker deployment
docker build -t sre-devops-env .
docker run -p 7860:7860 sre-devops-env
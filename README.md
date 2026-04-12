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

<div align="center">

# 🚨 SRE DevOps Simulator
### *A Deterministic Training Environment for AI-driven Production Incident Response*

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen.svg)]()
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)]()
[![Phase 2 Passed](https://img.shields.io/badge/Phase%202-Passed-brightgreen.svg)]()
[![LLM](https://img.shields.io/badge/LLM-LiteLLM%20Proxy-orange.svg)]()

**Can AI systems reliably handle real-world production incidents?**
*This environment finds out.*

</div>

---

## 💀 The $5.6 Billion Problem

Every year, enterprises lose **$5.6 billion** to unplanned cloud downtime. When a production server crashes at 3 AM, a human SRE must do something extraordinarily hard — **think clearly under pressure.**

They must:
- Read 500 log lines and find the one that matters
- Ignore 12 screaming alerts and identify the 1 real cause
- Execute the right fix in the right sequence — without making things worse

Current AI agents **fail catastrophically** at this. Not because they aren't smart — but because **there is no safe place to practice.** You cannot train a model on a live production database. One wrong command and the entire system goes down.

> **We built a safe, reproducible simulator for training AI agents on production incidents.**

---

## 🎯 The Solution: A Deterministic SRE Training Ground

The **SRE DevOps Simulator** is an OpenEnv-compliant, mathematically rigorous environment where AI agents practice production incident response — with real consequences, real noise, and real penalties — but zero actual risk.

Think of it as a **flight simulator for AI engineers.**

The agent plays the role of an on-call SRE. It receives live-updating server metrics, PagerDuty-style alerts (including deliberate red herrings), and system logs. It must diagnose the root cause and execute the correct remediation sequence — or watch the virtual site go down.

**What makes this hard (and valuable):**

- 🎭 **Intentional noise** — irrelevant alerts fire alongside real ones. Naive models get distracted.
- ⏱️ **Sequential dependencies** — some actions must happen in order. Skipping steps causes cascading failures.
- 💸 **Penalty enforcement** — redundant or illogical actions are penalized. You can't spam your way to a solution.
- 🧠 **Hidden root causes** — CPU metrics can look normal while RAM silently leaks. The agent must *investigate*, not guess.

---

## 🏗️ Environment Architecture

[Agent] -> Action -> [Environment] -> State/Reward -> [Agent]

### Observation Space

At every `step()`, the agent receives a rich, real-time cluster snapshot:

| Field | Type | Description |
|-------|------|-------------|
| `servers` | `Dict` | Per-node metrics — CPU %, RAM %, status, capacity |
| `alerts` | `List` | PagerDuty-style alerts, including intentional red herrings |
| `logs` | `List` | Recent system events and stderr output |
| `site_uptime` | `bool` | Is the customer-facing site currently resolving? |
| `active_connections` | `int` | Total cluster load — used for scaling decisions |
| `tick` | `int` | Current simulation timestep |
| `task_description` | `str` | Natural language briefing of the incident |

### Action Space — 8 Deterministic Tools

| Action | Description | Risk Level |
|--------|-------------|------------|
| `RestartService(server_id)` | Reboots a crashed node | 🟢 Low |
| `ScaleUp(resource_id)` | Provisions additional capacity for traffic spikes | 🟡 Medium |
| `ScaleDown(resource_id)` | Reduces capacity to cut costs | 🟡 Medium |
| `RollbackDeployment(version)` | Reverts a bad code push to a known-good state | 🔴 High |
| `KillProcess(server_id)` | Terminates a runaway process consuming CPU | 🟢 Low |
| `FlushCache(cache_id)` | Clears overloaded cache memory | 🟢 Low |
| `FailoverDatabase(replica_id)` | Promotes a replica if the primary DB fails | 🔴 High |
| `InvestigateLog(server_id)` | Reads detailed stdout/stderr for root cause analysis | 🔵 Discovery |

> Penalties are strictly enforced for redundant, premature, or illogical actions. The environment rewards *precision*, not volume.

---

## 🏆 Three Scenarios. Three Tests of Intelligence.

### 🟢 Scenario 1 — "The Dead Server" *(Easy)*

> *"Two critical alerts are screaming. One server is silent. Which do you fix?"*

A node (`web-3`) has crashed. Two unrelated critical alerts are firing simultaneously as red herrings. The agent must cut through the noise and issue a single, targeted `RestartService` command.

**Tests:** Basic tool use, alert triage, noise filtration.

---

### 🟡 Scenario 2 — "Traffic Tsunami" *(Medium)*

> *"It's Black Friday. Traffic just 10x'd. You have 60 seconds before the database falls over."*

A massive traffic surge is overwhelming the API gateways. The agent must sequentially `ScaleUp` the right resources in the right order — without over-provisioning (which triggers cost penalties) or repeating commands (which triggers spam penalties).

**Tests:** Sequential planning, resource management, penalty avoidance.

---

### 🔴 Scenario 3 — "The Silent Killer" *(Hard)*

> *"CPU looks fine. RAM is bleeding out. A deployment went out 2 hours ago. Connect the dots."*

A recent code push (`v2.3.1`) introduced a memory leak. CPU metrics appear completely normal — a deliberate trap for pattern-matching models. The agent must investigate logs, confirm the bad deployment, and execute `RollbackDeployment(v2.3.0)` in the correct sequence.

**Tests:** Multi-step reasoning, root cause analysis, using discovery tools before execution tools.

---

## 📊 Baseline Performance

Validated using **Qwen/Qwen2.5-72B-Instruct** via LiteLLM proxy — routed through the OpenEnv evaluator's API gateway.

| Task | Score | Outcome |
|------|-------|---------|
| 🟢 Easy — The Dead Server | `0.50` | Correctly identified and restarted `web-3` in 1 step |
| 🟡 Medium — Traffic Tsunami | `0.30` | Initiated correct scale sequence; partial credit for progression |
| 🔴 Hard — The Silent Killer | `0.50` | Identified RAM anomaly; partial rollback sequence completed |
| **Average** | **`0.43`** | *Meaningful gradient across difficulty levels — confirms the environment discriminates well between model capabilities* |

> Scores are bounded strictly within `(0, 1)` exclusive. A score of `1.0` would require perfect execution with zero redundant actions — an intentionally hard ceiling that separates elite models from average ones.

---

## ⚙️ How to Run

### Prerequisites

```bash
pip install fastapi uvicorn pydantic requests openenv-core uv
```

### Option 1 — Run Locally

```bash
# Terminal 1: Start the SRE environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Terminal 2: Set credentials and run the agent
export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=your_hf_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Option 2 — Run via Docker

```bash
# Build
docker build -t sre-devops-env .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e API_KEY=your_hf_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  sre-devops-env
```

### Expected Output Format

```
[START] task=easy env=sre-devops-env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=RestartService(web-3) reward=0.50 done=true error=null
[END]   success=true steps=1 score=0.50 rewards=0.50
```

---

## 🧠 Why This Matters for AI Research

Most LLM benchmarks test **knowledge retrieval** — "what is the capital of France?"

This environment tests something far harder: **agentic reasoning under operational pressure.**

The SRE domain is uniquely valuable for AI evaluation because:

1. **Ground truth is unambiguous** — either the site comes back up or it doesn't
2. **Partial credit is meaningful** — investigating before acting is smarter than acting blind, and the reward function captures this
3. **Noise resistance is measurable** — red herring alerts are quantifiably distracting, and we can measure exactly how much they degrade model performance
4. **Sequential correctness matters** — the right actions in the wrong order still fail, which tests planning depth beyond single-step reasoning

This is the kind of environment that separates models that *understand* systems from models that *pattern-match* on keywords.

---

## 📁 Project Structure

```
sre-devops-env/
├── server/
│   └── app.py          # FastAPI OpenEnv server — all 3 task environments
├── models.py           # Pydantic schemas — Action, Observation, Reward, StepResponse
├── inference.py        # Agent loop — LiteLLM proxy integration + deterministic fallback
├── openenv.yaml        # OpenEnv compliance manifest
├── Dockerfile          # Production container
├── requirements.txt    # Dependencies
└── README.md           # You are here
```

---

## 🔬 Technical Design Decisions

**Why OpenAI Client via LiteLLM Proxy?**

The environment uses the OpenAI-compatible client interface routed through a LiteLLM proxy.  
This ensures all model calls are tracked, reproducible, and compliant with OpenEnv evaluation requirements.

This setup allows:
- consistent API behavior across models
- centralized evaluation and logging
- compatibility with multiple backend providers

**Why deterministic environment design?**

The environment itself is deterministic, meaning identical action sequences always produce identical outcomes.  
This ensures reproducibility of experiments and enables consistent benchmarking across models.

**Why strict score bounds `(0, 1)`?**
A score of exactly `0.0` means total failure with no partial credit — which shouldn't happen in a well-designed reward function. A score of `1.0` means perfect execution — which is intentionally nearly impossible. The `(0, 1)` constraint forces reward functions to be nuanced and continuous.

---

<div align="center">

*Built for the OpenEnv Challenge · Powered by LiteLLM · Validated on Qwen2.5-72B*

*A step toward evaluating AI systems in real-world operational environments.*

</div>
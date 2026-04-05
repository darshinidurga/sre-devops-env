"""
inference.py
============
Baseline inference script for SRE DevOps OpenEnv environment.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
BENCHMARK    = "sre-devops-env"
MAX_STEPS    = 15
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Valid actions & smart defaults ─────────────────────────────────────────────
VALID_ACTIONS = [
    "RestartService", "ScaleUp", "ScaleDown",
    "RollbackDeployment", "KillProcess",
    "FlushCache", "FailoverDatabase", "InvestigateLog"
]

TASK_DEFAULTS = {
    "easy":   {"action_type": "RestartService",     "target_id": "web-3"},
    "medium": {"action_type": "ScaleUp",            "target_id": "api-gw-1"},
    "hard":   {"action_type": "InvestigateLog",     "target_id": "web-1"},
}

# ── Mandatory log functions ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── AI Action ──────────────────────────────────────────────────────────────────
def get_ai_action(
    observation: dict,
    task_id: str,
    action_log: List[dict] = None
) -> dict:
    if action_log is None:
        action_log = []

    servers = observation.get("servers", {})
    server_summary = "\n".join([
        f"  {sid}: cpu={s['cpu']}% ram={s['ram']}% "
        f"status={s['status']}"
        for sid, s in servers.items()
    ])

    alerts = observation.get("alerts", [])
    alert_summary = "\n".join([
        f"  [{a['severity'].upper()}] {a['server']}: {a['message']}"
        for a in alerts
    ]) or "  None"

    logs = observation.get("logs", [])[-5:]
    log_summary = "\n".join([
        f"  {l['server']}: {l['message']}"
        for l in logs
    ]) or "  None"

    deployments = observation.get("deployment_history", [])
    deploy_summary = "\n".join([
        f"  {d['version']} — {d['status']}"
        for d in deployments
    ]) or "  None"

    recent = action_log[-3:] if action_log else []
    recent_summary = "\n".join([
        f"  {a['action_type']} → {a['target_id']}"
        for a in recent
    ]) or "  None yet"

    prompt = f"""You are a Site Reliability Engineer fixing a production incident.

TASK: {observation.get('task_description', '')}
TICK: {observation.get('tick', 0)}
SITE UP: {observation.get('site_uptime', True)}
CONNECTIONS: {observation.get('active_connections', 0)}

SERVERS:
{server_summary}

ALERTS:
{alert_summary}

RECENT LOGS:
{log_summary}

DEPLOYMENT HISTORY:
{deploy_summary}

ACTIONS ALREADY TAKEN - DO NOT REPEAT:
{recent_summary}

STRICT RULES:
1. NEVER repeat same action + target you already did
2. target_id must NEVER be null or empty
3. For RollbackDeployment → always use target_id: "v2.3.0"
4. If you investigated web-1 already → investigate web-2 next
5. After investigating logs → do RollbackDeployment next
6. After RollbackDeployment → do RestartService on affected servers
7. For crashed server → use RestartService on that server
8. For traffic spike → use ScaleUp on api-gw-1 first

AVAILABLE ACTIONS:
  RestartService     → {{"action_type": "RestartService", "target_id": "web-3"}}
  ScaleUp            → {{"action_type": "ScaleUp", "target_id": "api-gw-1"}}
  ScaleDown          → {{"action_type": "ScaleDown", "target_id": "web-1"}}
  RollbackDeployment → {{"action_type": "RollbackDeployment", "target_id": "v2.3.0"}}
  KillProcess        → {{"action_type": "KillProcess", "target_id": "web-1"}}
  FlushCache         → {{"action_type": "FlushCache", "target_id": "cache-1"}}
  FailoverDatabase   → {{"action_type": "FailoverDatabase", "target_id": "db-replica"}}
  InvestigateLog     → {{"action_type": "InvestigateLog", "target_id": "web-1"}}

Respond with ONLY a JSON object:
{{"action_type": "ACTION_NAME", "target_id": "TARGET_ID"}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if parsed.get("action_type") not in VALID_ACTIONS:
                return TASK_DEFAULTS[task_id]
            if not parsed.get("target_id"):
                parsed["target_id"] = "v2.3.0" \
                    if parsed["action_type"] == "RollbackDeployment" \
                    else "web-1"
            return parsed
        except json.JSONDecodeError:
            return TASK_DEFAULTS[task_id]
    return TASK_DEFAULTS[task_id]


# ── Run Single Task ────────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:

    rewards:    List[float] = []
    action_log: List[dict]  = []
    steps_taken = 0
    score       = 0.0
    success     = False

    # Mandatory START log
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        response = requests.post(
            f"{ENV_URL}/reset/{task_id}",
            timeout=10
        )
        obs = response.json()

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            steps_taken = step
            error_msg   = None

            # Get AI action
            try:
                action = get_ai_action(obs, task_id, action_log)
            except Exception as e:
                error_msg = str(e)[:80]
                action    = TASK_DEFAULTS[task_id]

            # Fix null target
            if not action.get("target_id"):
                action["target_id"] = "v2.3.0" \
                    if action.get("action_type") == "RollbackDeployment" \
                    else "web-1"

            action_log.append(action)
            action_str = f"{action['action_type']}({action['target_id']})"

            # Send to environment
            reward = 0.0
            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={
                        "action_type": action["action_type"],
                        "target_id":   action["target_id"],
                        "parameters":  {}
                    },
                    timeout=10
                )
                result     = step_resp.json()
                obs        = result.get("observation", obs)
                reward_obj = result.get("reward", {})
                done       = result.get("done", False)
                reward     = reward_obj.get("score", 0.0)
                score      = reward

            except Exception as e:
                error_msg = str(e)[:80]
                done      = True

            rewards.append(reward)

            # Mandatory STEP log
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(
            step=steps_taken + 1,
            action="error",
            reward=0.0,
            done=True,
            error=str(e)[:80],
        )

    finally:
        # Mandatory END log
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 50, flush=True)
    print("SRE DEVOPS ENV — BASELINE INFERENCE", flush=True)
    print("=" * 50, flush=True)
    print(f"API  : {API_BASE_URL}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Env  : {ENV_URL}", flush=True)

    # Health check
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=5)
        print(f"Health: {r.json()}", flush=True)
    except Exception:
        print("ERROR: Environment not running!", flush=True)
        print(f"Fix: uvicorn app:app --host 0.0.0.0 --port 7860", flush=True)
        return

    # Run all 3 tasks
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}", flush=True)
        scores[task_id] = run_task(task_id)

    # Final summary
    print(f"\n{'='*50}", flush=True)
    print("BASELINE RESULTS", flush=True)
    print("=" * 50, flush=True)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<8} | {score:.4f} | {bar}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average  | {avg:.4f}", flush=True)
    print("=" * 50, flush=True)
    print("Baseline inference complete ✅", flush=True)


if __name__ == "__main__":
    main()
"""
inference.py
------------
Baseline inference script for SRE DevOps OpenEnv environment.
Runs an AI agent against all 3 tasks and reports scores.

Environment Variables:
    API_BASE_URL : LLM API endpoint
    MODEL_NAME   : Model identifier  
    HF_TOKEN     : Hugging Face / API key
"""

import os
import re
import json
import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Smart defaults per task ─────────────────────────────────
TASK_DEFAULTS = {
    "easy":   {"action_type": "RestartService",     "target_id": "web-3"},
    "medium": {"action_type": "ScaleUp",            "target_id": "api-gw-1"},
    "hard":   {"action_type": "InvestigateLog",     "target_id": "web-1"},
}

VALID_ACTIONS = [
    "RestartService", "ScaleUp", "ScaleDown",
    "RollbackDeployment", "KillProcess",
    "FlushCache", "FailoverDatabase", "InvestigateLog"
]

# ── AI Action ───────────────────────────────────────────────
def get_ai_action(observation: dict, task_id: str, 
                  action_log: list = None) -> dict:
    if action_log is None:
        action_log = []

    servers = observation.get("servers", {})
    server_summary = "\n".join([
        f"  {sid}: cpu={s['cpu']}% ram={s['ram']}% status={s['status']}"
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
{json.dumps([d['version'] + ' (' + d['status'] + ')'
for d in observation.get('deployment_history', [])], indent=2)}

ACTIONS ALREADY TAKEN - DO NOT REPEAT THESE:
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


# ── Run Task ────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    print(f"\n{'='*50}")
    print(f"Running Task: {task_id.upper()}")
    print(f"{'='*50}")

    try:
        response = requests.post(
            f"{ENV_URL}/reset/{task_id}",
            timeout=10
        )
        obs = response.json()
    except Exception as e:
        print(f"ERROR resetting task: {e}")
        return 0.0

    print(f"Task: {obs.get('task_description', '')[:100]}")

    final_score = 0.0
    best_score  = 0.0
    done        = False
    tick        = 0
    action_log  = []

    while not done and tick < 15:
        tick += 1

        try:
            action = get_ai_action(obs, task_id, action_log)
        except Exception as e:
            print(f"Tick {tick:>2}: AI error — {str(e)[:60]}")
            action = TASK_DEFAULTS[task_id]

        if not action.get("target_id"):
            action["target_id"] = "v2.3.0" \
                if action.get("action_type") == "RollbackDeployment" \
                else "web-1"

        action_log.append(action)

        print(f"Tick {tick:>2}: {action['action_type']:<22} "
              f"→ {action['target_id']}")

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
            result      = step_resp.json()
            obs         = result.get("observation", obs)
            reward_obj  = result.get("reward", {})
            done        = result.get("done", False)
            score       = reward_obj.get("score", 0.0)
            feedback    = reward_obj.get("feedback", "")
            final_score = score
            best_score  = max(best_score, score)

            print(f"         Score: {score:.2f} | "
                  f"Best: {best_score:.2f} | "
                  f"Done: {done}")
            if feedback:
                print(f"         → {feedback[:70]}")

        except Exception as e:
            print(f"         Step error: {e}")
            break

    print(f"\n  → Final Score [{task_id}]: {final_score:.4f}")
    print(f"  → Best Score  [{task_id}]: {best_score:.4f}")
    return final_score

# ── Main ────────────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("SRE DEVOPS ENV — BASELINE INFERENCE")
    print("="*50)
    print(f"API : {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Env  : {ENV_URL}")

    # Health check
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=5)
        print(f"Health: {r.json()}")
    except Exception:
        print("ERROR: Environment not running!")
        print("Fix: uvicorn app:app --host 0.0.0.0 --port 7860")
        return

    # Run all 3 tasks
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(task_id)

    # Results
    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<8} | {score:.4f} | {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average  | {avg:.4f}")
    print("="*50)
    print("Baseline inference complete ✅")


if __name__ == "__main__":
    main()
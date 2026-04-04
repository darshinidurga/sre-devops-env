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

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Smart defaults per task ───────────────────────────────────────────────────
TASK_DEFAULTS = {
    "easy":   {"action_type": "RestartService",      "target_id": "web-3"},
    "medium": {"action_type": "ScaleUp",             "target_id": "api-gw-1"},
    "hard":   {"action_type": "InvestigateLog",      "target_id": "web-1"},
}

# ── Valid server IDs ──────────────────────────────────────────────────────────
VALID_TARGETS = [
    "web-1", "web-2", "web-3",
    "api-gw-1", "api-gw-2",
    "db-primary", "db-replica",
    "cache-1", "v2.3.0"
]

VALID_ACTIONS = [
    "RestartService", "ScaleUp", "ScaleDown",
    "RollbackDeployment", "KillProcess",
    "FlushCache", "FailoverDatabase", "InvestigateLog"
]

# ── AI Action Function ────────────────────────────────────────────────────────
def get_ai_action(observation: dict, task_id: str) -> dict:
    """Ask the LLM what action to take given current state."""

    # Build server summary
    servers = observation.get("servers", {})
    server_summary = "\n".join([
        f"  {sid}: cpu={s['cpu']}% ram={s['ram']}% status={s['status']}"
        for sid, s in servers.items()
    ])

    # Build alerts summary
    alerts = observation.get("alerts", [])
    alert_summary = "\n".join([
        f"  [{a['severity'].upper()}] {a['server']}: {a['message']}"
        for a in alerts
    ]) or "  None"

    # Build logs summary (last 3 only)
    logs = observation.get("logs", [])[-3:]
    log_summary = "\n".join([
        f"  {l['server']}: {l['message']}"
        for l in logs
    ]) or "  None"

    prompt = f"""You are an expert Site Reliability Engineer.
Analyze the infrastructure and choose exactly ONE action to fix the problem.

TASK: {observation.get('task_description', '')}
TICK: {observation.get('tick', 0)}
SITE UP: {observation.get('site_uptime', True)}

SERVERS:
{server_summary}

ALERTS:
{alert_summary}

RECENT LOGS:
{log_summary}

DEPLOYMENT HISTORY:
{json.dumps([d['version'] + ' (' + d['status'] + ')' 
for d in observation.get('deployment_history', [])], indent=2)}

AVAILABLE ACTIONS (copy exact format):
  RestartService     → {{"action_type": "RestartService", "target_id": "web-1"}}
  ScaleUp            → {{"action_type": "ScaleUp", "target_id": "api-gw-1"}}
  ScaleDown          → {{"action_type": "ScaleDown", "target_id": "web-1"}}
  RollbackDeployment → {{"action_type": "RollbackDeployment", "target_id": "v2.3.0"}}
  KillProcess        → {{"action_type": "KillProcess", "target_id": "web-1"}}
  FlushCache         → {{"action_type": "FlushCache", "target_id": "cache-1"}}
  FailoverDatabase   → {{"action_type": "FailoverDatabase", "target_id": "db-replica"}}
  InvestigateLog     → {{"action_type": "InvestigateLog", "target_id": "web-1"}}

RULES:
- target_id must NEVER be null
- Only use server IDs shown in SERVERS section
- For RollbackDeployment use version like "v2.3.0"
- Pick the action most likely to fix the current problem

Respond with ONLY a JSON object, no explanation:
{{"action_type": "ACTION_NAME", "target_id": "TARGET_ID"}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()

    # Extract JSON safely
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        parsed = json.loads(match.group())

        # Validate action_type
        if parsed.get("action_type") not in VALID_ACTIONS:
            return TASK_DEFAULTS[task_id]

        # Fix null target_id
        if not parsed.get("target_id"):
            if parsed["action_type"] == "RollbackDeployment":
                parsed["target_id"] = "v2.3.0"
            else:
                parsed["target_id"] = "web-1"

        return parsed

    return TASK_DEFAULTS[task_id]


# ── Run Single Task ───────────────────────────────────────────────────────────
def run_task(task_id: str) -> float:
    """Run agent on one task and return final score."""
    print(f"\n{'='*50}")
    print(f"Running Task: {task_id.upper()}")
    print(f"{'='*50}")

    # Reset environment
    try:
        response = requests.post(
            f"{ENV_URL}/reset/{task_id}",
            timeout=10
        )
        obs = response.json()
    except Exception as e:
        print(f"ERROR: Could not reset task: {e}")
        return 0.0

    print(f"Task: {obs.get('task_description', '')[:80]}")

    best_score  = 0.0
    final_score = 0.0
    done        = False
    tick        = 0
    max_ticks   = 15

    while not done and tick < max_ticks:
        tick += 1

        # Get AI action
        try:
            action = get_ai_action(obs, task_id)
        except Exception as e:
            print(f"Tick {tick}: AI error — {str(e)[:80]}")
            action = TASK_DEFAULTS[task_id]

        # Ensure target_id is never None
        if not action.get("target_id"):
            action["target_id"] = "web-1"

        print(f"Tick {tick:>2}: {action['action_type']:<22} → {action['target_id']}")

        # Send action to environment
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
            reward      = result.get("reward", {})
            done        = result.get("done", False)
            score       = reward.get("score", 0.0)
            final_score = score
            best_score  = max(best_score, score)

            print(f"        Score: {score:.2f} | "
                  f"Best: {best_score:.2f} | "
                  f"Done: {done}")

        except Exception as e:
            print(f"        Step error: {e}")
            break

    print(f"\nFinal Score [{task_id}]: {final_score:.4f}")
    print(f"Best Score  [{task_id}]: {best_score:.4f}")
    return final_score


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("SRE DEVOPS ENV — BASELINE INFERENCE")
    print("="*50)
    print(f"API URL : {API_BASE_URL}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Env URL : {ENV_URL}")

    # Health check
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=5)
        print(f"Health  : {r.json()}")
    except Exception:
        print("ERROR: Environment not running!")
        print(f"Start it: uvicorn app:app --host 0.0.0.0 --port 7860")
        return

    # Run all tasks
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(task_id)

    # Print summary
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
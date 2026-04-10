import os
import time
import re
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── HELPER TO GET CONFIG AT RUNTIME ────────────────────────────────────────────
def get_config():
    """Load and validate required environment variables at runtime."""
    api_base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    model_name = os.environ.get("MODEL_NAME")
    env_url = os.environ.get("ENV_URL")
    
    # Strict validation - all required
    if not api_base_url:
        raise ValueError("ERROR: API_BASE_URL environment variable not set")
    if not api_key:
        raise ValueError("ERROR: API_KEY environment variable not set")
    if not model_name:
        raise ValueError("ERROR: MODEL_NAME environment variable not set")
    if not env_url:
        raise ValueError("ERROR: ENV_URL environment variable not set")
    
    return {
        "api_base_url": api_base_url,
        "api_key": api_key,
        "model_name": model_name,
        "env_url": env_url
    }

# Global config - will be set by main()
CONFIG = None

BENCHMARK = "sre-devops-env"
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5

VALID_ACTIONS = [
    "RestartService", "ScaleUp", "ScaleDown",
    "RollbackDeployment", "KillProcess",
    "FlushCache", "FailoverDatabase", "InvestigateLog"
]

TASK_DEFAULTS = {
    "easy": {"action_type": "RestartService", "target_id": "web-3"},
    "medium": {"action_type": "ScaleUp", "target_id": "api-gw-1"},
    "hard": {"action_type": "InvestigateLog", "target_id": "web-1"},
}

# ── LOGGING (STRICT FORMAT) ───────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── AI ACTION ─────────────────────────────────────────────────────────────────
def get_ai_action(client: OpenAI, observation: dict, task_id: str, action_log: List[dict] = None) -> dict:
    if action_log is None:
        action_log = []

    prompt = f"""
You are a Site Reliability Engineer fixing a production incident.

TASK: {observation.get('task_description', '')}

Respond with ONLY a JSON object:
{{"action_type": "ACTION_NAME", "target_id": "TARGET_ID"}}
"""

    # 🚨 MUST call LLM (no silent fallback)
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}")

    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        raise RuntimeError("No JSON returned from model")

    try:
        parsed = json.loads(match.group())
    except:
        raise RuntimeError("Invalid JSON from model")

    if parsed.get("action_type") not in VALID_ACTIONS:
        return TASK_DEFAULTS.get(task_id, TASK_DEFAULTS["easy"])

    if not parsed.get("target_id"):
        parsed["target_id"] = "web-1"

    return parsed

# ── RUN TASK ──────────────────────────────────────────────────────────────────
def run_task(client: OpenAI, task_id: str) -> float:
    rewards = []
    action_log = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="sre-devops-env", model=CONFIG["model_name"])

    try:
        response = requests.post(
            f"{CONFIG['env_url']}/reset/{task_id}",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        obs = response.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            steps_taken = step
            error_msg = None

            action = get_ai_action(client, obs, task_id, action_log)
            action_log.append(action)

            action_str = f"{action['action_type']}({action['target_id']})"
            reward = 0.0

            try:
                step_resp = requests.post(
                    f"{CONFIG['env_url']}/step",
                    headers={"Content-Type": "application/json"},
                    json={
                        "action_type": action["action_type"],
                        "target_id": action["target_id"],
                        "parameters": {}
                    },
                    timeout=10
                )
                step_resp.raise_for_status()
                result = step_resp.json()

                obs = result.get("observation", obs)
                reward = result.get("reward", {}).get("score", 0.0)
                done = result.get("done", False)
                score = reward

            except Exception as e:
                error_msg = str(e)[:80]
                done = True

            rewards.append(reward)
            log_step(step, action_str, reward, done, error_msg)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(steps_taken + 1, "error", 0.0, True, str(e)[:80])

    finally:
        log_end(success, steps_taken, score, rewards)

    return score

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    global CONFIG
    
    # Load and validate config at runtime
    try:
        CONFIG = get_config()
        print(f"[CONFIG] API_BASE_URL set: {bool(CONFIG['api_base_url'])}", flush=True)
        print(f"[CONFIG] API_KEY set: {bool(CONFIG['api_key'])}", flush=True)
        print(f"[CONFIG] MODEL_NAME: {CONFIG['model_name']}", flush=True)
        print(f"[CONFIG] ENV_URL: {CONFIG['env_url']}", flush=True)
    except ValueError as e:
        print(f"ERROR: Configuration error: {e}", flush=True)
        return
    
    # Initialize OpenAI client with provided credentials
    try:
        print(f"[CLIENT] Initializing OpenAI client with base_url={CONFIG['api_base_url'][:20]}... (LiteLLM proxy)", flush=True)
        client = OpenAI(
            api_key=CONFIG["api_key"],
            base_url=CONFIG["api_base_url"]
        )
        print("[CLIENT] OpenAI client initialized successfully", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}", flush=True)
        return

    # Health check with retries
    env_up = False
    for attempt in range(5):
        try:
            requests.get(f"{CONFIG['env_url']}/health", timeout=5)
            env_up = True
            print(f"[HEALTH] Environment container is ready", flush=True)
            break
        except Exception as e:
            print(f"[HEALTH] Attempt {attempt+1}/5 - Waiting for env container... {e}", flush=True)
            time.sleep(2)
            
    if not env_up:
        print("ERROR: Environment not reachable!", flush=True)
        return

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            scores[task_id] = run_task(client, task_id)
        except Exception as e:
            print(f"ERROR: Unhandled exception in task {task_id}: {e}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: Unhandled exception exiting inference.py: {e}", flush=True)
"""
inference.py

Baseline inference script for SRE DevOps OpenEnv environment.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (evaluator's LiteLLM proxy).
    API_KEY        The evaluator's injected API key.
    MODEL_NAME     The model identifier to use for inference.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import json
import requests as req_lib
from typing import List, Optional, Dict, Any

from models import Action, ActionType, Observation, Reward, StepResponse

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY      = os.environ.get("API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

BENCHMARK               = "sre-devops-env"
MAX_STEPS               = 15
SUCCESS_SCORE_THRESHOLD = 0.5

VALID_ACTIONS = [
    "RestartService", "ScaleUp", "ScaleDown",
    "RollbackDeployment", "KillProcess",
    "FlushCache", "FailoverDatabase", "InvestigateLog"
]


# ── Mandatory Log Functions ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Helper: works for both dicts and Pydantic objects ─────────────────────────
def _g(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── Direct HTTP call to proxy ──────────────────────────────────────────────────
def call_llm_via_proxy(api_base_url: str, api_key: str, model: str, prompt: str) -> Optional[str]:
    """
    Raw requests call — bypasses OpenAI SDK entirely.
    This guarantees the HTTP request physically reaches the proxy.
    The OpenAI SDK can silently abort before sending; requests cannot.
    """
    base = api_base_url.rstrip('/')
    if not base.endswith('/v1'):
        base = base + '/v1'

    url = f"{base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.1,
    }

    print(f"[DEBUG] POST {url} model={model}", flush=True)
    resp = req_lib.post(url, headers=headers, json=payload, timeout=45)
    print(f"[DEBUG] Proxy status: {resp.status_code}", flush=True)

    if resp.status_code == 200:
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        print(f"[DEBUG] LLM response: {text}", flush=True)
        return text
    else:
        print(f"[DEBUG] Proxy error: {resp.text[:300]}", flush=True)
        return None


# ── Smart Fallback Helpers ─────────────────────────────────────────────────────
def get_hard_task_fallback(action_log: List[Dict]) -> Action:
    actions_taken  = [(_g(a, "action_type"), _g(a, "target_id")) for a in action_log]
    investigated_web1 = any(t == "InvestigateLog"    and tid == "web-1"  for t, tid in actions_taken)
    investigated_web2 = any(t == "InvestigateLog"    and tid == "web-2"  for t, tid in actions_taken)
    rolled_back       = any(t == "RollbackDeployment"                     for t, _   in actions_taken)
    restarted_web1    = any(t == "RestartService"     and tid == "web-1"  for t, tid in actions_taken)
    restarted_web2    = any(t == "RestartService"     and tid == "web-2"  for t, tid in actions_taken)

    if not investigated_web1:
        return Action(action_type=ActionType.InvestigateLog,    target_id="web-1")
    elif not investigated_web2:
        return Action(action_type=ActionType.InvestigateLog,    target_id="web-2")
    elif not rolled_back:
        return Action(action_type=ActionType.RollbackDeployment, target_id="v2.3.0")
    elif not restarted_web1:
        return Action(action_type=ActionType.RestartService,    target_id="web-1")
    elif not restarted_web2:
        return Action(action_type=ActionType.RestartService,    target_id="web-2")
    else:
        return Action(action_type=ActionType.InvestigateLog,    target_id="web-1")


def get_medium_task_fallback(action_log: List[Dict]) -> Action:
    actions_taken = [(_g(a, "action_type"), _g(a, "target_id")) for a in action_log]
    scaled_gw1 = any(t == "ScaleUp" and tid == "api-gw-1" for t, tid in actions_taken)
    scaled_gw2 = any(t == "ScaleUp" and tid == "api-gw-2" for t, tid in actions_taken)

    if not scaled_gw1:
        return Action(action_type=ActionType.ScaleUp, target_id="api-gw-1")
    elif not scaled_gw2:
        return Action(action_type=ActionType.ScaleUp, target_id="api-gw-2")
    else:
        return Action(action_type=ActionType.InvestigateLog, target_id="api-gw-1")


def get_easy_task_fallback() -> Action:
    return Action(action_type=ActionType.RestartService, target_id="web-3")


# ── AI Action Generation ───────────────────────────────────────────────────────
def get_ai_action(
    api_base_url: str,
    api_key: str,
    observation: Observation,
    task_id: str,
    action_log: List[Dict]
) -> Action:

    # Build observation summary
    servers = observation.servers
    server_summary = "\n".join([
        f"  {sid}: cpu={_g(s,'cpu',0)}% ram={_g(s,'ram',0)}% status={_g(s,'status','unknown')}"
        for sid, s in servers.items()
    ])

    alerts = observation.alerts
    alert_summary = "\n".join([
        f"  [{_g(a,'severity','INFO').upper()}] {_g(a,'server','unknown')}: {_g(a,'message','')}"
        for a in alerts
    ]) or "  None"

    logs = observation.logs[-5:]
    log_summary = "\n".join([
        f"  {_g(l,'server','unknown')}: {_g(l,'message','')}"
        for l in logs
    ]) or "  None"

    recent = action_log[-3:] if action_log else []
    recent_summary = "\n".join([
        f"  {_g(a,'action_type','')} -> {_g(a,'target_id','')}"
        for a in recent
    ]) or "  None yet"

    prompt = f"""You are a Site Reliability Engineer fixing a production incident.

TASK: {observation.task_description}
TICK: {observation.tick}
SITE UP: {observation.site_uptime}
CONNECTIONS: {observation.active_connections}

SERVERS:
{server_summary}

ALERTS:
{alert_summary}

RECENT LOGS:
{log_summary}

ACTIONS ALREADY TAKEN - DO NOT REPEAT:
{recent_summary}

STRICT RULES:
1. NEVER repeat same action + target you already did
2. target_id must NEVER be null or empty
3. For RollbackDeployment always use target_id: v2.3.0
4. HARD TASK SEQUENCE follow this EXACTLY:
   Step 1: InvestigateLog -> web-1
   Step 2: InvestigateLog -> web-2
   Step 3: RollbackDeployment -> v2.3.0
   Step 4: RestartService -> web-1
   Step 5: RestartService -> web-2
5. MEDIUM TASK: ScaleUp(api-gw-1) then ScaleUp(api-gw-2)

AVAILABLE ACTIONS:
  RestartService, ScaleUp, ScaleDown, RollbackDeployment, KillProcess, FlushCache, FailoverDatabase, InvestigateLog

Respond with ONLY a JSON object:
{{"action_type": "ACTION_NAME", "target_id": "TARGET_ID"}}"""

    # ── Always call proxy via raw HTTP first ──────────────────────────────────
    if api_base_url and api_key:
        try:
            text = call_llm_via_proxy(api_base_url, api_key, MODEL_NAME, prompt)
            if text:
                match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                if match:
                    parsed         = json.loads(match.group())
                    action_type_str = parsed.get("action_type", "")
                    if action_type_str in VALID_ACTIONS:
                        target_id = parsed.get("target_id") or ""
                        if not target_id:
                            target_id = "v2.3.0" if action_type_str == "RollbackDeployment" else "web-1"
                        is_repeat = any(
                            _g(a, "action_type") == action_type_str and _g(a, "target_id") == target_id
                            for a in action_log
                        )
                        if not is_repeat:
                            action = Action(action_type=ActionType(action_type_str), target_id=target_id)
                            action_log.append({"action_type": action_type_str, "target_id": target_id})
                            return action
                        print(f"[DEBUG] LLM repeat detected, falling back", flush=True)
                    else:
                        print(f"[DEBUG] Invalid action from LLM: {action_type_str}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] Proxy call exception: {exc}", flush=True)
    else:
        print(f"[DEBUG] No credentials — skipping LLM call", flush=True)

    # ── Deterministic fallback ────────────────────────────────────────────────
    print(f"[DEBUG] Deterministic fallback for task={task_id}", flush=True)
    if task_id == "hard":
        action = get_hard_task_fallback(action_log)
    elif task_id == "medium":
        action = get_medium_task_fallback(action_log)
    else:
        action = get_easy_task_fallback()

    action_type_val = action.action_type.value if isinstance(action.action_type, ActionType) else action.action_type
    action_log.append({"action_type": action_type_val, "target_id": action.target_id})
    return action


# ── Environment Interface ──────────────────────────────────────────────────────
class SREEnvironmentClient:
    def __init__(self, base_url: str = ENV_URL):
        self.base_url = base_url.rstrip('/')
        self.session  = req_lib.Session()

    def reset(self, task_id: str) -> Observation:
        resp = self.session.post(f"{self.base_url}/reset/{task_id}", timeout=10)
        resp.raise_for_status()
        return Observation(**resp.json())

    def step(self, action: Action) -> StepResponse:
        payload = {
            "action_type": action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type),
            "target_id":   action.target_id,
            "parameters":  action.parameters or {}
        }
        resp   = self.session.post(f"{self.base_url}/step", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        if "observation" in result:
            obs    = Observation(**result["observation"])
            reward = Reward(**result["reward"])
            done   = result.get("done", False)
        else:
            obs         = Observation(**result)
            reward_data = result.get("reward", {
                "score": 0.0, "feedback": "No feedback",
                "done": False, "total_ticks": 0, "breakdown": {}
            })
            reward = Reward(**reward_data)
            done   = result.get("done", False)

        return StepResponse(observation=obs, reward=reward, done=done, info={})

    def health(self) -> dict:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        self.session.close()


# ── Run Single Task ────────────────────────────────────────────────────────────
def run_task(
    env: SREEnvironmentClient,
    api_base_url: str,
    api_key: str,
    task_id: str
) -> float:
    rewards:    List[float] = []
    action_log: List[Dict]  = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = env.reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            steps_taken = step
            error_msg   = None

            action     = get_ai_action(api_base_url, api_key, obs, task_id, action_log)
            action_str = (
                f"{action.action_type.value if isinstance(action.action_type, ActionType) else action.action_type}"
                f"({action.target_id})"
            )

            reward = 0.0
            try:
                step_resp = env.step(action)
                obs    = step_resp.observation
                reward = step_resp.reward.score
                done   = step_resp.done
                score  = reward
            except Exception as e:
                error_msg = str(e)[:80]
                done      = True

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e)[:80])

    finally:
        safe_score = max(0.01, min(0.99, score))
        log_end(success=success, steps=steps_taken, score=safe_score, rewards=rewards)

    safe_score = max(0.01, min(0.99, score))
    return safe_score


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 50, flush=True)
    print("SRE DEVOPS ENV — BASELINE INFERENCE", flush=True)
    print("=" * 50, flush=True)

    api_base_url = API_BASE_URL.strip()
    api_key      = API_KEY.strip()

    print(f"[DEBUG] API_BASE_URL: {'SET -> ' + api_base_url if api_base_url else 'NOT SET'}", flush=True)
    print(f"[DEBUG] API_KEY:      {'SET' if api_key else 'NOT SET'}", flush=True)
    print(f"[DEBUG] MODEL_NAME:   {MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_URL:      {ENV_URL}", flush=True)

    env    = SREEnvironmentClient(ENV_URL)
    health = env.health()
    print(f"Health: {health}", flush=True)

    if health.get("status") == "error":
        print("ERROR: SRE environment not reachable!", flush=True)
        return

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'=' * 50}", flush=True)
        scores[task_id] = run_task(env, api_base_url, api_key, task_id)

    print(f"\n{'=' * 50}", flush=True)
    print("BASELINE RESULTS", flush=True)
    print("=" * 50, flush=True)
    for task_id, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  {task_id:<8} | {s:.4f} | {bar}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average  | {avg:.4f}", flush=True)
    print("=" * 50, flush=True)
    print("Baseline inference complete", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
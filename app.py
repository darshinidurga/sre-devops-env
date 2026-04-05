"""
app.py
------
Flask HTTP server for the SRE DevOps OpenEnv simulation environment.

Routes
------
GET  /health              — liveness probe
GET  /tasks               — list all tasks and their metadata
GET  /tasks/<task_id>     — info for a single task
POST /reset/<task_id>     — start a new episode  → Observation JSON
POST /step                — apply one action      → StepResponse JSON

Run locally
-----------
    python app.py

Or with a production WSGI server:
    waitress-serve --host=0.0.0.0 --port=7860 app:app

The ENV_PORT environment variable overrides the default port 7860.
"""

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, request, Response

# ---------------------------------------------------------------------------
# Path bootstrap — ensure repo root importable when running from any CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.environment import SREEnvironment  # noqa: E402

# ---------------------------------------------------------------------------
# Flask app + shared environment instance
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

env = SREEnvironment()


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error(message: str, status: int = 400) -> Response:
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — always returns 200 when the server is running."""
    return jsonify({
        "status": "ok",
        "current_task": env.current_task_id,
        "tick": env.current_tick,
        "done": env.is_done,
    })


@app.route("/tasks", methods=["GET"])
def list_tasks():
    """Return metadata for all registered tasks."""
    return jsonify(env.all_task_info())


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id: str):
    """Return metadata for a single task."""
    try:
        return jsonify(env.task_info(task_id))
    except ValueError as exc:
        return _error(str(exc), 404)


@app.route("/reset/<task_id>", methods=["POST", "GET"])
def reset(task_id: str):
    """
    Start a fresh episode for *task_id*.

    Returns the initial Observation as JSON.
    Accepts both POST (as inference.py uses) and GET for browser convenience.
    """
    try:
        observation = env.reset(task_id)
        return jsonify(observation)
    except ValueError as exc:
        return _error(str(exc), 404)
    except Exception as exc:
        return _error(f"Reset failed: {exc}", 500)


@app.route("/step", methods=["POST"])
def step():
    """
    Apply one action to the current episode.

    Expected JSON body::

        {
            "action_type": "RestartService",
            "target_id":   "web-3",
            "parameters":  {}          (optional)
        }

    Returns a StepResponse (observation + reward + done).
    """
    body = request.get_json(silent=True) or {}

    action_type = body.get("action_type", "").strip()
    target_id   = body.get("target_id",   "").strip()
    parameters  = body.get("parameters") or {}

    if not action_type:
        return _error("Missing required field: action_type")
    if not target_id:
        return _error("Missing required field: target_id")

    try:
        step_resp = env.step(
            action_type=action_type,
            target_id=target_id,
            parameters=parameters if parameters else None,
        )
        return jsonify(step_resp)
    except ValueError as exc:
        return _error(str(exc), 422)
    except RuntimeError as exc:
        return _error(str(exc), 409)
    except Exception as exc:
        return _error(f"Step failed: {exc}", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("ENV_PORT", "7860"))
    print(f"SRE OpenEnv starting on http://0.0.0.0:{port}")
    print("Routes: GET /health  GET /tasks  POST /reset/<task_id>  POST /step")
    app.run(host="0.0.0.0", port=port, debug=False)

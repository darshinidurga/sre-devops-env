"""
app.py
------
FastAPI web server for the SRE Cloud DevOps OpenEnv.

Endpoints
---------
  GET  /health            → liveness probe
  GET  /tasks             → list all available TaskInfo objects
  GET  /state             → current Observation (requires prior reset)
  POST /reset             → reset with body  {"task_id": "easy|medium|hard"}
  POST /reset/{task_id}   → reset shortcut via URL path param
  POST /step              → apply an Action, advance the simulation

Run
---
  uvicorn app:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path setup — ensure models.py and server/ are importable from repo root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SERVER = _ROOT / "server"
for _p in (_ROOT, _SERVER):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import (
    Action,
    Observation,
    StepResponse,
    TaskInfo,
)
from server.environment import SREEnvironment, _TASKS

# ---------------------------------------------------------------------------
# Global environment instance (single-user / single-episode server)
# ---------------------------------------------------------------------------
_env: SREEnvironment = SREEnvironment()
_env_initialised: bool = False   # True after the first successful reset()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SRE DevOps OpenEnv API",
    description=(
        "REST interface for the TechCorp SRE simulation environment. "
        "Call `/reset` first, then drive the simulation with `/step`."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body accepted by POST /reset."""
    task_id: str = "easy"


def _require_initialised() -> None:
    """Raise 400 if reset() has never been called."""
    if not _env_initialised:
        raise HTTPException(
            status_code=400,
            detail=(
                "Environment not initialised. "
                "Call POST /reset or POST /reset/{task_id} first."
            ),
        )


def _do_reset(task_id: str) -> Observation:
    """Shared reset logic used by both reset endpoints."""
    global _env_initialised

    if task_id not in _TASKS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown task_id '{task_id}'. "
                f"Valid values: {sorted(_TASKS.keys())}"
            ),
        )

    try:
        obs = _env.reset(task_id)
        _env_initialised = True
        return obs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Liveness probe",
    tags=["meta"],
)
def health() -> Dict[str, str]:
    """
    Returns a simple liveness signal.  Use this to verify the server is up
    before calling any simulation endpoints.
    """
    return {"status": "ok", "environment": "sre-devops-env"}


@app.get(
    "/tasks",
    response_model=List[TaskInfo],
    summary="List all available tasks",
    tags=["meta"],
)
def list_tasks() -> List[TaskInfo]:
    """
    Returns metadata for every built-in task/scenario
    (easy, medium, hard) including name, description,
    difficulty, and max_ticks.
    """
    return list(_TASKS.values())


@app.post(
    "/reset",
    response_model=Observation,
    summary="Reset environment (body)",
    tags=["environment"],
)
def reset_via_body(body: ResetRequest) -> Observation:
    """
    Reset the environment to a fresh episode for the given task.

    **Body**
    ```json
    { "task_id": "easy" }
    ```
    Valid values for `task_id`: `easy`, `medium`, `hard`.

    Returns the initial **Observation** for the new episode.
    """
    return _do_reset(body.task_id)


@app.post(
    "/reset/{task_id}",
    response_model=Observation,
    summary="Reset environment (path param)",
    tags=["environment"],
)
def reset_via_path(
    task_id: str = FPath(
        ...,
        description="Task difficulty preset: easy | medium | hard",
        example="medium",
    ),
) -> Observation:
    """
    Shortcut to reset with the task supplied in the URL path.

    Example: `POST /reset/hard`

    Returns the initial **Observation** for the new episode.
    """
    return _do_reset(task_id)


@app.post(
    "/step",
    response_model=StepResponse,
    summary="Apply action and advance the simulation",
    tags=["environment"],
)
def step(action: Action) -> StepResponse:
    """
    Apply a single **Action** to the environment, advance the simulation
    by one tick, and return a **StepResponse** containing:

    - `observation` — updated cluster state
    - `reward`      — score, breakdown, and human-readable feedback
    - `done`        — whether the episode has ended
    - `info`        — diagnostic details (action result, done reason, etc.)

    **Example body**
    ```json
    {
      "action_type": "RestartService",
      "target_id":   "web-1",
      "parameters":  {}
    }
    ```

    Returns **400** if `/reset` has not been called yet, or **409** if the
    episode is already done.
    """
    _require_initialised()

    if _env.is_done:
        raise HTTPException(
            status_code=409,
            detail=(
                "Episode is already done. "
                "Call POST /reset to start a new episode."
            ),
        )

    try:
        return _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/state",
    response_model=Observation,
    summary="Get current observation without advancing time",
    tags=["environment"],
)
def get_state() -> Observation:
    """
    Return the current **Observation** without consuming a tick.

    Useful for inspecting the cluster state between steps or after a reset.

    Returns **400** if `/reset` has not been called yet.
    """
    _require_initialised()

    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        log_level="info",
    )

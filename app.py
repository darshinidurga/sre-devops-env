"""
app.py
------
FastAPI server for the SRE DevOps OpenEnv environment.

Endpoints
---------
GET  /health
GET  /tasks
GET  /tasks/{task_id}
POST /reset
POST /reset/{task_id}
POST /step
GET  /state
GET  /docs  (auto-generated)
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from server.environment import SREEnvironment

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="SRE DevOps OpenEnv",
    description="SRE Cloud DevOps Simulator for OpenEnv Hackathon",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SREEnvironment()

# ── Request models ─────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    action_type: str
    target_id:   str
    parameters:  Optional[Dict[str, Any]] = {}

# ── Endpoints ──────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe."""
    return {
        "status": "ok",
        "environment": "sre-devops-env",
        "current_task": env.current_task_id,
        "tick": env.current_tick,
        "done": env.is_done,
    }


@app.get("/tasks")
def list_tasks():
    """List all tasks and metadata."""
    return env.all_task_info()


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Get metadata for a single task."""
    try:
        return env.task_info(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/reset")
def reset_post(body: ResetRequest):
    """Reset environment with task_id in body."""
    try:
        return env.reset(body.task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{task_id}")
def reset_by_path(task_id: str):
    """Reset environment with task_id in URL."""
    try:
        return env.reset(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reset/{task_id}")
def reset_by_path_get(task_id: str):
    """Reset environment GET version for browser testing."""
    try:
        return env.reset(task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    """Apply one action to current episode."""
    if not body.action_type:
        raise HTTPException(status_code=400, detail="Missing action_type")
    if not body.target_id:
        raise HTTPException(status_code=400, detail="Missing target_id")
    try:
        return env.step(
            action_type=body.action_type,
            target_id=body.target_id,
            parameters=body.parameters,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Get current environment state."""
    try:
        if env.current_task_id is None:
            raise HTTPException(
                status_code=409,
                detail="No active episode. Call /reset first."
            )
        return env.reset(env.current_task_id) \
            if env._sim.state is None \
            else env._sim._build_observation().model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ENV_PORT", "7860"))
    print(f"SRE OpenEnv starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
"""
server/environment.py
---------------------
Thin adapter over SRESimulator for the SRE DevOps OpenEnv environment.

Wraps SRESimulator to provide the public API that app.py expects:
    env = SREEnvironment()
    obs  = env.reset("easy")          → Observation (Pydantic model)
    resp = env.step(action)           → StepResponse (Pydantic model)
    obs  = env.state()                → Observation

All heavy simulation + grading logic lives in simulator.py and tasks/*.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SERVER_DIR, ".."))
for _p in (_REPO_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import Action, ActionType, Observation, StepResponse, TaskInfo  # noqa: E402
from simulator import SRESimulator                                            # noqa: E402


class SREEnvironment:
    """
    Public-facing environment adapter.

    Delegates all simulation, action handling, and grading to
    :class:`~simulator.SRESimulator` and the per-task modules.

    Attributes exposed for app.py:
        current_task_id  → active task id string, or None
        current_tick     → ticks elapsed this episode
        is_done          → True when the episode has ended
    """

    def __init__(self) -> None:
        self._sim = SRESimulator()

    # ------------------------------------------------------------------
    # Public API — returns Pydantic models (NOT dicts)
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Reset to a fresh episode for *task_id*.

        Parameters
        ----------
        task_id : str
            One of ``"easy"``, ``"medium"``, ``"hard"``.

        Returns
        -------
        Observation
            Initial observation for the new episode.
        """
        return self._sim.reset(task_id)

    def step(self, action: Action) -> StepResponse:
        """
        Apply *action* and return the step result.

        Parameters
        ----------
        action : Action
            Validated Action object.

        Returns
        -------
        StepResponse

        Raises
        ------
        RuntimeError
            If reset() has not been called first, or episode is done.
        """
        return self._sim.step(action)

    def state(self) -> Observation:
        """Return the current observation without advancing the simulation."""
        if self._sim.state is None:
            raise RuntimeError("Call reset() before state().")
        return self._sim._build_observation()

    # ------------------------------------------------------------------
    # Convenience properties (used by app.py)
    # ------------------------------------------------------------------

    @property
    def current_task_id(self) -> Optional[str]:
        """Active task id, or None before the first reset."""
        return self._sim.task_id

    @property
    def current_tick(self) -> int:
        """Ticks elapsed in the current episode."""
        return self._sim.tick

    @property
    def is_done(self) -> bool:
        """True when the episode has ended."""
        return self._sim.done

    def __repr__(self) -> str:
        return (
            f"SREEnvironment(task={self.current_task_id!r}, "
            f"tick={self.current_tick}, done={self.is_done})"
        )

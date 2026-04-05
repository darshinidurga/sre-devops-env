"""
server/environment.py
---------------------
HTTP-level adapter around SRESimulator.

Converts raw HTTP request data (dicts) into typed model objects and
serialises Pydantic responses back to plain dicts for JSON transport.

Used by app.py — keeps Flask route handlers thin and free of simulation
logic.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, Optional

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SERVER_DIR, ".."))
for _p in (_REPO_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import Action, ActionType          # noqa: E402
from server.simulator import SRESimulator      # noqa: E402


class SREEnvironment:
    """
    Thin HTTP adapter over :class:`~server.simulator.SRESimulator`.

    All methods accept and return plain Python dicts so Flask can serialise
    them directly to JSON without extra conversion steps.
    """

    def __init__(self) -> None:
        self._sim = SRESimulator()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Dict[str, Any]:
        """
        Reset the simulator to a fresh episode for *task_id*.

        Returns
        -------
        dict
            Serialised :class:`~models.Observation`.
        """
        obs = self._sim.reset(task_id)
        return obs.model_dump()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action_type: str,
        target_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply an action and return the full step response.

        Parameters
        ----------
        action_type:
            One of the :class:`~models.ActionType` string values.
        target_id:
            The resource the action targets.
        parameters:
            Optional free-form action parameters.

        Returns
        -------
        dict
            Serialised :class:`~models.StepResponse`.

        Raises
        ------
        ValueError
            If the environment has not been reset yet, or *action_type* is invalid.
        """
        if self._sim.state is None:
            raise ValueError("Environment not initialised — call reset() first.")

        action = Action(
            action_type=action_type,
            target_id=target_id,
            parameters=parameters,
        )
        step_resp = self._sim.step(action)
        return step_resp.model_dump()

    # ------------------------------------------------------------------
    # task_info
    # ------------------------------------------------------------------

    def task_info(self, task_id: str) -> Dict[str, Any]:
        """Return serialised TaskInfo for the given task_id."""
        try:
            from server.tasks import TASK_REGISTRY
        except ModuleNotFoundError:
            from tasks import TASK_REGISTRY  # type: ignore[no-redef]

        tid = task_id.strip().lower()
        if tid not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id {task_id!r}")
        return TASK_REGISTRY[tid].get_task_info().model_dump()

    # ------------------------------------------------------------------
    # all_task_info
    # ------------------------------------------------------------------

    def all_task_info(self) -> Dict[str, Any]:
        """Return serialised TaskInfo for all registered tasks."""
        try:
            from server.tasks import TASK_REGISTRY
        except ModuleNotFoundError:
            from tasks import TASK_REGISTRY  # type: ignore[no-redef]

        return {
            tid: module.get_task_info().model_dump()
            for tid, module in TASK_REGISTRY.items()
        }

    # ------------------------------------------------------------------
    # Properties that mirror SRESimulator state (for Flask health checks)
    # ------------------------------------------------------------------

    @property
    def current_task_id(self) -> Optional[str]:
        return self._sim.task_id

    @property
    def current_tick(self) -> int:
        return self._sim.tick

    @property
    def is_done(self) -> bool:
        return self._sim.done

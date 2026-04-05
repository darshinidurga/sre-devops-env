"""
server/graders.py
-----------------
Central grading system for the SRE OpenEnv simulation environment.

Dispatches to the correct per-task ``grade()`` function based on a
``task_id`` string, normalises scores, and provides both an OO and a
standalone-function interface so callers (environment.py, inference.py,
tests) can use whichever style suits them.

Task signatures (for reference)
--------------------------------
easy   : grade(action_history)                              → float
medium : grade(action_history, final_state, ticks_survived) → float
hard   : grade(action_history, final_state, ticks_used)     → float

All task graders already return values in [0.0, 1.0]; ``validate_score``
performs a defensive clamp + type check as a safety net.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path bootstrap — ensure repo root and server/ are importable
# ---------------------------------------------------------------------------
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SERVER_DIR, ".."))
for _p in (_REPO_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import per-task grade functions
# ---------------------------------------------------------------------------
from tasks.easy   import grade as _grade_easy    # noqa: E402
from tasks.medium import grade as _grade_medium  # noqa: E402
from tasks.hard   import grade as _grade_hard    # noqa: E402
from models import Action                        # noqa: E402

# ---------------------------------------------------------------------------
# Supported task IDs
# ---------------------------------------------------------------------------
SUPPORTED_TASKS = frozenset({"easy", "medium", "hard"})


# ---------------------------------------------------------------------------
# TaskGrader
# ---------------------------------------------------------------------------


class TaskGrader:
    """
    Central dispatcher that routes an episode's data to the appropriate
    per-task grader and normalises the resulting score.

    Usage
    -----
    ::

        grader = TaskGrader()

        # Grade a single episode
        score = grader.grade_episode(
            task_id="hard",
            action_history=actions,
            final_state=state,
            ticks_used=7,
        )

        # Grade all tasks from a results dict
        scores = grader.get_all_scores({
            "easy":   {"action_history": [...], "final_state": {...}, "ticks_used": 2},
            "medium": {"action_history": [...], "final_state": {...}, "ticks_used": 10},
            "hard":   {"action_history": [...], "final_state": {...}, "ticks_used": 6},
        })
    """

    # ------------------------------------------------------------------
    # grade_episode
    # ------------------------------------------------------------------

    def grade_episode(
        self,
        task_id: str,
        action_history: List[Action],
        final_state: Dict[str, Any],
        ticks_used: int,
    ) -> float:
        """
        Route the episode to the correct task grader and return a score.

        Parameters
        ----------
        task_id:
            One of ``"easy"``, ``"medium"``, or ``"hard"``.
        action_history:
            Ordered list of :class:`~models.Action` objects from the episode.
        final_state:
            The state dict at episode end (from ``simulate_tick`` or
            equivalent).  Ignored for ``easy`` (which only needs actions).
        ticks_used:
            Number of ticks consumed during the episode.

        Returns
        -------
        float
            Normalised score in [0.0, 1.0].

        Raises
        ------
        ValueError
            If ``task_id`` is not supported.
        """
        tid = task_id.strip().lower()
        if tid not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task_id {task_id!r}. "
                f"Must be one of: {sorted(SUPPORTED_TASKS)}"
            )

        if tid == "easy":
            raw = _grade_easy(action_history, final_state, ticks_used)

        elif tid == "medium":
            raw = _grade_medium(
                action_history,
                final_state,
                ticks_survived=ticks_used,
            )

        else:  # "hard"
            raw = _grade_hard(
                action_history,
                final_state,
                ticks_used=ticks_used,
            )

        return self.validate_score(raw)

    # ------------------------------------------------------------------
    # get_all_scores
    # ------------------------------------------------------------------

    def get_all_scores(
        self,
        results_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Grade every task present in *results_dict* and return a score map.

        Parameters
        ----------
        results_dict:
            Mapping of ``task_id → episode_data`` where each ``episode_data``
            is a dict with the following keys:

            ``action_history`` : list[Action]
                Actions taken during the episode.
            ``final_state`` : dict
                Environment state at episode end.
            ``ticks_used`` : int
                Ticks consumed (defaults to ``0`` if missing).

        Returns
        -------
        dict[str, float]
            ``{task_id: score}`` for every task_id present in the input.

        Notes
        -----
        Tasks present in the input but not in ``SUPPORTED_TASKS`` are
        silently skipped and will not appear in the output.
        """
        scores: Dict[str, float] = {}
        for task_id, episode_data in results_dict.items():
            tid = task_id.strip().lower()
            if tid not in SUPPORTED_TASKS:
                continue  # skip unknown tasks gracefully

            action_history: List[Action] = episode_data.get("action_history", [])
            final_state: Dict[str, Any] = episode_data.get("final_state", {})
            ticks_used: int = int(episode_data.get("ticks_used", 0))

            scores[tid] = self.grade_episode(
                task_id=tid,
                action_history=action_history,
                final_state=final_state,
                ticks_used=ticks_used,
            )

        return scores

    # ------------------------------------------------------------------
    # validate_score
    # ------------------------------------------------------------------

    @staticmethod
    def validate_score(score: float) -> float:
        """
        Ensure *score* is a valid float clamped to [0.0, 1.0].

        Parameters
        ----------
        score:
            The raw score returned by a task grader.

        Returns
        -------
        float
            The score clamped to [0.0, 1.0].

        Raises
        ------
        TypeError
            If *score* is not an ``int`` or ``float``.
        ValueError
            If *score* is ``NaN`` or ``±Inf``.
        """
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"Score must be a numeric float, got {type(score).__name__!r}: {score!r}"
            )

        import math
        if not math.isfinite(score):
            raise ValueError(
                f"Score must be a finite number, got {score!r}"
            )

        return round(float(min(max(score, 0.0), 1.0)), 4)


# ---------------------------------------------------------------------------
# Standalone convenience function (used by inference.py directly)
# ---------------------------------------------------------------------------

_default_grader = TaskGrader()


def run_grader(
    task_id: str,
    action_history: List[Action],
    final_state: Dict[str, Any],
    ticks_used: int,
) -> float:
    """
    Standalone function wrapper around :meth:`TaskGrader.grade_episode`.

    Intended for direct use in ``inference.py`` and other scripts that
    prefer a simple function call over instantiating a class.

    Parameters
    ----------
    task_id:
        One of ``"easy"``, ``"medium"``, or ``"hard"``.
    action_history:
        Ordered list of :class:`~models.Action` objects.
    final_state:
        Environment state dict at episode end.
    ticks_used:
        Number of ticks consumed.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    return _default_grader.grade_episode(
        task_id=task_id,
        action_history=action_history,
        final_state=final_state,
        ticks_used=ticks_used,
    )

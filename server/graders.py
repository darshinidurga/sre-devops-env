"""
server/graders.py
-----------------
Central grading system for the SRE DevOps OpenEnv environment.
Routes grading to the correct task-specific grader.
"""

from __future__ import annotations

import math
import sys
import os
from typing import Any, Dict, List, Union

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SERVER_DIR = os.path.abspath(os.path.dirname(__file__))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# ── Import models ─────────────────────────────────────────────────────────────
from models import Action, ActionType

# ── Import task graders ───────────────────────────────────────────────────────
# Using server.tasks.* style as requested
try:
    from server.tasks.easy   import grade as easy_grade,   get_task_info as easy_info
    from server.tasks.medium import grade as medium_grade, get_task_info as medium_info
    from server.tasks.hard   import grade as hard_grade,   get_task_info as hard_info
except ImportError:
    # Fallback for different execution environments
    from tasks.easy   import grade as easy_grade,   get_task_info as easy_info
    from tasks.medium import grade as medium_grade, get_task_info as medium_info
    from tasks.hard   import grade as hard_grade,   get_task_info as hard_info


# ── Helper for action normalization ──────────────────────────────────────────
def _ensure_action_objects(action_history: List[Union[Action, Dict[str, Any]]]) -> List[Action]:
    """Convert a list of dicts or Actions into a list of Actions."""
    normalized = []
    for a in action_history:
        if isinstance(a, Action):
            normalized.append(a)
        elif isinstance(a, dict):
            # Normalizing dict to Action object so Enum comparisons work
            normalized.append(Action(
                action_type=a["action_type"],
                target_id=a.get("target_id", ""),
                parameters=a.get("parameters")
            ))
        else:
            raise TypeError(f"Expected Action or dict, got {type(a)}")
    return normalized


# ── TaskGrader Class ──────────────────────────────────────────────────────────
class TaskGrader:
    """
    Central grader that routes scoring to the correct task grader.
    """

    TASK_IDS = ["easy", "medium", "hard"]

    def grade_episode(
        self,
        task_id: str,
        action_history: List[Union[Action, Dict[str, Any]]],
        final_state: Dict[str, Any],
        ticks_used: int,
    ) -> float:
        """
        Grade a completed episode for the given task.
        """
        # Normalization (fixes 0.0/0.05 score issues when receiving dicts)
        actions = _ensure_action_objects(action_history)

        # Debug info as requested
        print(f"Grading {task_id}: {len(actions)} actions")
        print(f"Actions: {[(a.action_type, a.target_id) for a in actions]}")

        # Routing logic (calls task-specific graders)
        score = self._route(task_id, actions, final_state, ticks_used)
        score = self.validate_score(score)

        print(f"Final score: {score}")
        return score

    def _route(
        self,
        task_id: str,
        action_history: List[Action],
        final_state: Dict[str, Any],
        ticks_used: int,
    ) -> float:
        """Route to the correct task grader."""
        tid = task_id.strip().lower()
        
        if tid == "easy":
            return easy_grade(
                action_history=action_history,
                final_state=final_state,
                ticks_used=ticks_used,
            )
        elif tid == "medium":
            return medium_grade(
                action_history=action_history,
                final_state=final_state,
                ticks_survived=ticks_used
            )
        elif tid == "hard":
            return hard_grade(
                action_history=action_history,
                final_state=final_state,
                ticks_used=ticks_used
            )
        else:
            # Raise ValueError for unknown tasks as requested by validation habits
            raise ValueError(f"Unknown task_id '{task_id}'")

    def get_all_scores(
        self,
        results_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Grade all tasks from a results dictionary.
        """
        scores = {}
        for task_id, episode_data in results_dict.items():
            try:
                scores[task_id] = self.grade_episode(
                    task_id=task_id,
                    action_history=episode_data.get("action_history", []),
                    final_state=episode_data.get("final_state", {}),
                    ticks_used=episode_data.get("ticks_used", 0),
                )
            except ValueError:
                # Still Return 0.0 for unknown tasks in the bulk grader
                scores[task_id] = 0.0
        return scores

    @staticmethod
    def validate_score(score: float) -> float:
        """Ensure score is always between 0.0 and 1.0."""
        if not isinstance(score, (int, float)):
            raise TypeError(f"Score must be a number, got {type(score)}: {score}")
        
        if not math.isfinite(score):
            return 0.0
            
        return round(float(min(max(score, 0.0), 1.0)), 4)

    def list_tasks(self):
        """Return TaskInfo for all tasks."""
        return [
            easy_info(),
            medium_info(),
            hard_info(),
        ]


# ── Standalone function ───────────────────────────────────────────────────────
def run_grader(
    task_id: str,
    action_history: List[Union[Action, Dict[str, Any]]],
    final_state: Dict[str, Any],
    ticks_used: int,
) -> float:
    """Standalone version of TaskGrader.grade_episode()."""
    grader = TaskGrader()
    return grader.grade_episode(
        task_id=task_id,
        action_history=action_history,
        final_state=final_state,
        ticks_used=ticks_used
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing graders...\n")

    # Test easy grader with a dict to verify normalization
    print("=" * 40)
    print("TEST: Easy — correct first action (dict)")
    test_actions = [
        {"action_type": "RestartService", "target_id": "web-3"}
    ]
    test_score = run_grader("easy", test_actions, {}, 1)
    print(f"Expected: 1.0 | Got: {test_score}")
    assert test_score == 1.0, f"Easy grader failed: {test_score}"

    # Test medium grader
    print("\n" + "=" * 40)
    print("TEST: Medium — scale up")
    med_actions = [
        {"action_type": "ScaleUp", "target_id": "api-gw-1"}
    ]
    test_score = run_grader("medium", med_actions, {}, 10)
    print(f"Expect score > 0.0 | Got: {test_score}")

    print("\n✅ All grader tests complete!")
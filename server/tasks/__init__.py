"""
server/tasks/__init__.py
------------------------
Task registry — maps task_id strings to their module so the simulator and
grader can look up the correct setup/grade functions without conditionals
scattered across the codebase.
"""

from . import easy, medium, hard  # noqa: F401

# Central registry: task_id -> module with setup_scenario / grade / get_task_info
TASK_REGISTRY = {
    "easy":   easy,
    "medium": medium,
    "hard":   hard,
}

__all__ = ["TASK_REGISTRY", "easy", "medium", "hard"]

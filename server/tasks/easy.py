"""
server/tasks/easy.py
--------------------
Task: "The Dead Server"
Task ID: "easy"

Scenario
--------
web-3 is completely offline. The agent must identify and restart it as quickly
as possible. Two additional alerts (api-gw-1 high CPU, cache-1 high memory) are
present as red-herrings to test whether the agent focuses on the critical issue
rather than getting distracted by non-critical warnings.

Public API
----------
- setup_scenario()  → dict   — initial environment state
- grade(action_history) → float — score in [0.0, 1.0]
- get_task_info()   → TaskInfo
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Import models from the parent-level models.py
# ---------------------------------------------------------------------------
# The directory layout is:
#   sre-devops-env/
#     models.py          ← we want this
#     server/
#       tasks/
#         easy.py        ← this file
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models import (  # noqa: E402
    Action,
    ActionType,
    Alert,
    AlertSeverity,
    Deployment,
    Server,
    ServerStatus,
    TaskInfo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ID = "easy"
TASK_NAME = "The Dead Server"
TASK_DESCRIPTION = (
    "web-3 has gone completely offline and is serving no traffic. "
    "Other alerts are present in the environment (elevated CPU on api-gw-1 "
    "and high memory on cache-1) but these are non-critical. "
    "Your goal is to restore service by restarting web-3 as quickly as possible."
)
TASK_DIFFICULTY = "easy"
TASK_MAX_TICKS = 20


# ---------------------------------------------------------------------------
# setup_scenario
# ---------------------------------------------------------------------------


def setup_scenario() -> Dict[str, Any]:
    """
    Return the initial state dict for "The Dead Server" scenario.

    The dict contains:
      - ``servers``     : dict[server_id, Server]
      - ``alerts``      : list[Alert]
      - ``deployment``  : Deployment
    """
    servers: Dict[str, Server] = {
        # Completely offline — the target of the task
        "web-3": Server(
            id="web-3",
            cpu=0.0,
            ram=0.0,
            status=ServerStatus.offline,
            active_connections=0,
            version="v2.3.0",
        ),
        # Healthy web nodes
        "web-1": Server(
            id="web-1",
            cpu=30.0,
            ram=40.0,
            status=ServerStatus.healthy,
            active_connections=120,
            version="v2.3.0",
        ),
        "web-2": Server(
            id="web-2",
            cpu=30.0,
            ram=40.0,
            status=ServerStatus.healthy,
            active_connections=115,
            version="v2.3.0",
        ),
        # RED HERRING — elevated CPU but not the root cause
        "api-gw-1": Server(
            id="api-gw-1",
            cpu=75.0,
            ram=45.0,
            status=ServerStatus.degraded,
            active_connections=300,
            version="v2.3.0",
        ),
        # RED HERRING — high memory but not the root cause
        "cache-1": Server(
            id="cache-1",
            cpu=20.0,
            ram=80.0,
            status=ServerStatus.degraded,
            active_connections=0,
            version="v2.3.0",
        ),
        # Healthy database primary
        "db-primary": Server(
            id="db-primary",
            cpu=25.0,
            ram=35.0,
            status=ServerStatus.healthy,
            active_connections=50,
            version="v2.3.0",
        ),
    }

    alerts: List[Alert] = [
        Alert(
            id="ALT-001",
            severity=AlertSeverity.critical,
            message="web-3 is OFFLINE - no response",
            server="web-3",
            tick=0,
        ),
        Alert(
            id="ALT-002",
            severity=AlertSeverity.warning,
            message="api-gw-1 elevated CPU 75%",
            server="api-gw-1",
            tick=0,
        ),
        Alert(
            id="ALT-003",
            severity=AlertSeverity.info,
            message="cache-1 memory usage high 80%",
            server="cache-1",
            tick=0,
        ),
    ]

    deployment = Deployment(
        version="v2.3.0",
        status="active",
        age_mins=180.0,
    )

    return {
        "servers": servers,
        "alerts": alerts,
        "deployment": deployment,
    }


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------


def grade(
    action_history: List[Action],
    final_state: dict = None,  # accepted but unused for easy task
    ticks_used: int = 0,       # accepted but unused for easy task
) -> float:
    """
    Score the agent's performance based on its action history.

    Scoring rubric
    --------------
    1.0  — RestartService on web-3 was the VERY FIRST action
    0.7  — RestartService on web-3 was performed, but not as first action
           (agent did some investigation/other actions before finding the issue)
    0.5  — RestartService on web-3 was performed after 5 or more other actions
           (agent took too long to find the root cause)
    0.3  — Agent tried restarting a wrong server before eventually fixing web-3
    0.0  — web-3 was never restarted

    Parameters
    ----------
    action_history:
        Ordered list of Action objects taken during the episode (earliest first).
    final_state:
        Ignored for this task (accepted for API consistency with medium/hard).
    ticks_used:
        Ignored for this task (accepted for API consistency with medium/hard).

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    if not action_history:
        return 0.0

    def _is_restart_web3(action: Action) -> bool:
        """Return True regardless of whether action_type is stored as enum or str."""
        atype = action.action_type
        # Normalise: could be ActionType enum member OR plain string
        atype_str = atype.value if hasattr(atype, "value") else str(atype)
        return atype_str == "RestartService" and action.target_id == "web-3"

    def _is_action_type(action: Action, type_str: str) -> bool:
        """Generic helper to compare action_type against a string name."""
        atype = action.action_type
        atype_str = atype.value if hasattr(atype, "value") else str(atype)
        return atype_str == type_str

    # Find the index of the first RestartService targeting web-3
    restart_web3_idx: int | None = None
    for idx, action in enumerate(action_history):
        if _is_restart_web3(action):
            restart_web3_idx = idx
            break

    # Agent never restarted web-3
    if restart_web3_idx is None:
        return 0.0

    # Perfect: web-3 restarted immediately as the very first action
    if restart_web3_idx == 0:
        return 1.0

    # Inspect actions taken before the correct restart
    prior_actions = action_history[:restart_web3_idx]

    tried_wrong_server = any(
        _is_action_type(a, "RestartService") and a.target_id != "web-3"
        for a in prior_actions
    )

    # Worst case: tried wrong server before eventually fixing web-3
    if tried_wrong_server:
        return 0.3

    # Slow case: 5 or more actions taken before finding the right fix
    if len(prior_actions) >= 5:
        return 0.5

    # Not first, but found it reasonably quickly without wrong restarts
    return 0.7


# ---------------------------------------------------------------------------
# get_task_info
# ---------------------------------------------------------------------------


def get_task_info() -> TaskInfo:
    """Return metadata describing this task."""
    return TaskInfo(
        task_id=TASK_ID,
        name=TASK_NAME,
        description=TASK_DESCRIPTION,
        difficulty=TASK_DIFFICULTY,
        max_ticks=TASK_MAX_TICKS,
    )

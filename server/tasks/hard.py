"""
server/tasks/hard.py
--------------------
Task: "The Silent Killer"
Task ID: "hard"

Scenario
--------
Deployment v2.3.1 was pushed 12 minutes ago and introduced a memory leak.
web-1 and web-2 RAM climbs 8 % every tick and will crash when it exceeds 95 %.
CPU looks normal — that is a deliberate red herring to make the memory leak
non-obvious. db-primary emits slow-query logs, also a red herring.

The correct remediation is:
  1. Investigate logs on web-1 and web-2 (reveals OOM exceptions)
  2. RollbackDeployment to v2.3.0   (stops the leak)
  3. RestartService on web-1 and web-2 (clears accumulated RAM)

Public API
----------
- setup_scenario()                       → dict
- simulate_tick(state, tick_number)      → dict  (mutates state in-place, returns it)
- grade(action_history, final_state, ticks_used) → float  [0.0, 1.0]
- get_task_info()                        → TaskInfo
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import models from the repo-root models.py
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
    LogEntry,
    Server,
    ServerStatus,
    TaskInfo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ID = "hard"
TASK_NAME = "The Silent Killer"
TASK_DESCRIPTION = (
    "Deployment v2.3.1 (pushed 12 minutes ago) introduced a memory leak. "
    "web-1 and web-2 RAM is climbing steadily while CPU stays deceptively normal. "
    "db-primary slow-query warnings are a red herring. "
    "You must investigate the web servers, roll back to v2.3.0, and restart "
    "the affected nodes before they run out of memory and go offline."
)
TASK_DIFFICULTY = "hard"
TASK_MAX_TICKS = 20

# RAM thresholds
_RAM_INCREASE_PER_TICK: float = 8.0
_RAM_CRASH_THRESHOLD: float = 95.0
_RAM_AFTER_RESTART: float = 30.0

# Deployment versions
_CURRENT_VERSION = "v2.3.1"
_STABLE_VERSION = "v2.3.0"


# ---------------------------------------------------------------------------
# setup_scenario
# ---------------------------------------------------------------------------


def setup_scenario() -> Dict[str, Any]:
    """
    Return the initial state dict for "The Silent Killer" scenario.

    The dict contains:
      - ``servers``            : dict[server_id, Server]
      - ``alerts``             : list[Alert]
      - ``logs``               : list[LogEntry]
      - ``deployment_history`` : list[Deployment]  (most recent last)
      - ``_meta``              : internal mutable bookkeeping (not for agents)
    """
    servers: Dict[str, Server] = {
        # Leaking — high RAM, deceptively normal CPU
        "web-1": Server(
            id="web-1",
            cpu=45.0,
            ram=72.0,
            status=ServerStatus.degraded,
            active_connections=140,
            version=_CURRENT_VERSION,
        ),
        "web-2": Server(
            id="web-2",
            cpu=43.0,
            ram=74.0,
            status=ServerStatus.degraded,
            active_connections=135,
            version=_CURRENT_VERSION,
        ),
        # Healthy — unaffected
        "web-3": Server(
            id="web-3",
            cpu=30.0,
            ram=35.0,
            status=ServerStatus.healthy,
            active_connections=120,
            version=_CURRENT_VERSION,
        ),
        # RED HERRING — slow queries but not the root cause
        "db-primary": Server(
            id="db-primary",
            cpu=38.0,
            ram=50.0,
            status=ServerStatus.degraded,
            active_connections=80,
            version=_CURRENT_VERSION,
        ),
    }

    alerts: List[Alert] = [
        Alert(
            id="ALT-001",
            severity=AlertSeverity.warning,
            message="web-1 RAM 72% and rising",
            server="web-1",
            tick=0,
        ),
        Alert(
            id="ALT-002",
            severity=AlertSeverity.warning,
            message="web-2 RAM 74% and rising",
            server="web-2",
            tick=0,
        ),
        Alert(
            id="ALT-003",
            severity=AlertSeverity.info,
            message="db-primary slow queries detected",
            server="db-primary",
            tick=0,
        ),
        Alert(
            id="ALT-004",
            severity=AlertSeverity.info,
            message="CPU usage normal across all servers",
            server="web-1",
            tick=0,
        ),
    ]

    logs: List[LogEntry] = [
        LogEntry(
            tick=0,
            server="web-1",
            level="ERROR",
            message="OutOfMemoryException: heap allocation failed in worker thread",
        ),
        LogEntry(
            tick=0,
            server="web-2",
            level="ERROR",
            message="OutOfMemoryException: heap allocation failed in worker thread",
        ),
        LogEntry(
            tick=0,
            server="db-primary",
            level="WARN",
            message="Slow query detected: SELECT * FROM sessions (took 4200ms)",
        ),
        LogEntry(
            tick=0,
            server="web-1",
            level="WARN",
            message="Memory pressure increasing — GC pauses 850ms",
        ),
        LogEntry(
            tick=0,
            server="web-2",
            level="WARN",
            message="Memory pressure increasing — GC pauses 910ms",
        ),
    ]

    deployment_history: List[Deployment] = [
        # Last stable version
        Deployment(version=_STABLE_VERSION, status="superseded", age_mins=600.0),
        # Current bad deployment — most recent
        Deployment(version=_CURRENT_VERSION, status="active", age_mins=12.0),
    ]

    return {
        "servers": servers,
        "alerts": alerts,
        "logs": logs,
        "deployment_history": deployment_history,
        # Internal bookkeeping (not exposed to agents):
        # leak_stopped  — True once RollbackDeployment is applied
        # restarted     — set of server IDs that have been restarted
        "_meta": {
            "leak_stopped": False,
            "restarted": set(),
        },
    }


# ---------------------------------------------------------------------------
# simulate_tick
# ---------------------------------------------------------------------------


def simulate_tick(state: Dict[str, Any], tick_number: int) -> Dict[str, Any]:
    """
    Advance the environment by one tick.

    Effects
    -------
    - web-1 and web-2 RAM each increase by 8 % unless the leak has been stopped
      (i.e. ``RollbackDeployment`` was applied earlier).
    - If a server's RAM exceeds 95 % it transitions to ``offline`` (cpu → 0, ram → 0).
    - Servers that have been ``RestartService``d already have their RAM at 30 %
      and are not further inflated.

    Parameters
    ----------
    state:
        The mutable state dict produced by :func:`setup_scenario` (or a previous
        tick). This function modifies it **in-place** and also returns it.
    tick_number:
        The current tick index (1-indexed from the caller's perspective).

    Returns
    -------
    dict
        The same ``state`` dict after applying tick effects.
    """
    meta: Dict[str, Any] = state.get("_meta", {})
    leak_stopped: bool = meta.get("leak_stopped", False)
    restarted: set = meta.get("restarted", set())

    servers: Dict[str, Server] = state["servers"]
    new_logs: List[LogEntry] = []

    for server_id in ("web-1", "web-2"):
        server = servers.get(server_id)
        if server is None:
            continue

        # Already offline — nothing to do
        if server.status == ServerStatus.offline:
            continue

        # Restarted servers are no longer leaking
        if server_id in restarted:
            continue

        # Apply memory leak only if rollback hasn't been done
        if not leak_stopped:
            new_ram = min(server.ram + _RAM_INCREASE_PER_TICK, 100.0)

            # Server crashes when RAM exceeds threshold
            if new_ram > _RAM_CRASH_THRESHOLD:
                servers[server_id] = Server(
                    id=server_id,
                    cpu=0.0,
                    ram=0.0,
                    status=ServerStatus.offline,
                    active_connections=0,
                    version=server.version,
                )
                new_logs.append(LogEntry(
                    tick=tick_number,
                    server=server_id,
                    level="CRITICAL",
                    message=(
                        f"{server_id} OOM-killed — RAM exceeded {_RAM_CRASH_THRESHOLD}%. "
                        "Server is now offline."
                    ),
                ))
            else:
                servers[server_id] = Server(
                    id=server_id,
                    cpu=server.cpu,
                    ram=new_ram,
                    status=ServerStatus.degraded if new_ram < 90.0 else ServerStatus.critical,
                    active_connections=server.active_connections,
                    version=server.version,
                )
                new_logs.append(LogEntry(
                    tick=tick_number,
                    server=server_id,
                    level="WARN",
                    message=f"{server_id} RAM now at {new_ram:.1f}% — memory leak continuing",
                ))

    # Append new log entries
    state["logs"] = state.get("logs", []) + new_logs

    return state


# ---------------------------------------------------------------------------
# Helpers used by grade()
# ---------------------------------------------------------------------------


def _investigated(action_history: List[Action], server_id: str) -> bool:
    """Return True if InvestigateLog was called on *server_id*."""
    return any(
        a.action_type == ActionType.InvestigateLog and a.target_id == server_id
        for a in action_history
    )


def _rollback_done(action_history: List[Action]) -> bool:
    """Return True if RollbackDeployment targeting v2.3.0 appears in history."""
    return any(
        a.action_type == ActionType.RollbackDeployment
        and (a.target_id == _STABLE_VERSION or a.target_id == "v2.3.0")
        for a in action_history
    )


def _restarted(action_history: List[Action], server_id: str) -> bool:
    """Return True if RestartService was applied to *server_id*."""
    return any(
        a.action_type == ActionType.RestartService and a.target_id == server_id
        for a in action_history
    )


def _site_stayed_online(final_state: Dict[str, Any]) -> bool:
    """
    Return True if neither web-1 nor web-2 crashed (went offline) during the episode.

    We check the *final* server states: if both are still online (healthy,
    degraded, or critical) then the site never went fully dark.
    """
    servers: Dict[str, Server] = final_state.get("servers", {})
    for sid in ("web-1", "web-2"):
        s = servers.get(sid)
        if s is not None and s.status == ServerStatus.offline:
            return False
    return True


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------


def grade(
    action_history: List[Action],
    final_state: Dict[str, Any],
    ticks_used: int,
) -> float:
    """
    Score the agent using an additive rubric.

    Rubric
    ------
    +0.15  InvestigateLog called on web-1 OR web-2
    +0.15  InvestigateLog called on BOTH web-1 AND web-2
    +0.25  RollbackDeployment to v2.3.0 performed
    +0.20  RestartService on BOTH web-1 AND web-2
    +0.15  Site never went fully offline (web-1 and web-2 both stayed up)
    +0.10  All of the above accomplished in < 8 ticks (speed bonus)

    Total possible: 1.00  — clamped to [0.0, 1.0].

    Parameters
    ----------
    action_history:
        Ordered list of Action objects taken during the episode.
    final_state:
        The state dict at episode end (from :func:`simulate_tick` or equivalent).
    ticks_used:
        Number of ticks consumed.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    score: float = 0.0

    investigated_web1 = _investigated(action_history, "web-1")
    investigated_web2 = _investigated(action_history, "web-2")
    rollback_applied = _rollback_done(action_history)
    restarted_web1 = _restarted(action_history, "web-1")
    restarted_web2 = _restarted(action_history, "web-2")
    site_online = _site_stayed_online(final_state)

    # +0.15 — investigated at least one of the leaking servers
    if investigated_web1 or investigated_web2:
        score += 0.15

    # +0.15 — investigated both leaking servers
    if investigated_web1 and investigated_web2:
        score += 0.15

    # +0.25 — rolled back the bad deployment
    if rollback_applied:
        score += 0.25

    # +0.20 — restarted both affected web servers
    if restarted_web1 and restarted_web2:
        score += 0.20

    # +0.15 — neither web-1 nor web-2 crashed during the episode
    if site_online:
        score += 0.15

    # +0.10 — speed bonus: all milestones achieved in under 8 ticks
    all_done = (
        (investigated_web1 and investigated_web2)
        and rollback_applied
        and (restarted_web1 and restarted_web2)
        and site_online
    )
    if all_done and ticks_used < 8:
        score += 0.10

    return round(min(max(score, 0.0), 1.0), 4)


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

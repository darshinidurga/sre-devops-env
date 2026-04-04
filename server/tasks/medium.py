"""
server/tasks/medium.py
----------------------
Task: "Traffic Tsunami"
Task ID: "medium"

Scenario
--------
A Black Friday-style traffic surge is hitting the cluster. All servers are
healthy but under growing load. Active connections climb 1 500 per tick.
API gateways are first to buckle (cpu +3%/tick → offline at 95%); if
connections exceed 15 000 the database crashes. The agent must scale up
the gateways and web servers fast enough to keep the system alive for
10 full ticks.

Public API
----------
- setup_scenario()                            → dict
- simulate_tick(state, tick_number)           → dict  (mutates + returns state)
- grade(action_history, final_state,
        ticks_survived)                       → float [0.0, 1.0]
- get_task_info()                             → TaskInfo
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

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

TASK_ID = "medium"
TASK_NAME = "Traffic Tsunami"
TASK_DESCRIPTION = (
    "A Black Friday traffic surge is hammering the cluster. "
    "Active connections are rising 1 500 every tick. "
    "API gateways will buckle first — their CPU climbs 3% per tick and they "
    "go offline above 95%. If total connections exceed 15 000, db-primary "
    "crashes. Scale up the gateways and web tier quickly enough to keep the "
    "system alive for 10 ticks."
)
TASK_DIFFICULTY = "medium"
TASK_MAX_TICKS = 15

# Tick dynamics
_CONNECTIONS_PER_TICK: int = 1500
_GW_CPU_INCREASE_PER_TICK: float = 3.0
_GW_CPU_OFFLINE_THRESHOLD: float = 95.0
_DB_CRASH_CONNECTION_THRESHOLD: int = 15_000

# API-gateway IDs — used throughout for consistency
_GW_IDS = ("api-gw-1", "api-gw-2")

# Web-server IDs
_WEB_IDS = ("web-1", "web-2", "web-3")


# ---------------------------------------------------------------------------
# setup_scenario
# ---------------------------------------------------------------------------


def setup_scenario() -> Dict[str, Any]:
    """
    Return the initial state dict for "Traffic Tsunami".

    Keys
    ----
    servers             : dict[server_id, Server]
    alerts              : list[Alert]
    logs                : list[LogEntry]
    deployment_history  : list[Deployment]
    active_connections  : int  — cluster-wide connection counter
    _meta               : internal bookkeeping (not exposed to agents)
    """
    servers: Dict[str, Server] = {
        # API gateways — hot and climbing
        "api-gw-1": Server(
            id="api-gw-1",
            cpu=89.0,
            ram=70.0,
            status=ServerStatus.critical,
            active_connections=8000,
            version="v2.3.0",
        ),
        "api-gw-2": Server(
            id="api-gw-2",
            cpu=85.0,
            ram=68.0,
            status=ServerStatus.critical,
            active_connections=7500,
            version="v2.3.0",
        ),
        # Web tier — healthy but working hard
        "web-1": Server(
            id="web-1",
            cpu=75.0,
            ram=65.0,
            status=ServerStatus.degraded,
            active_connections=1200,
            version="v2.3.0",
        ),
        "web-2": Server(
            id="web-2",
            cpu=75.0,
            ram=65.0,
            status=ServerStatus.degraded,
            active_connections=1100,
            version="v2.3.0",
        ),
        "web-3": Server(
            id="web-3",
            cpu=75.0,
            ram=65.0,
            status=ServerStatus.degraded,
            active_connections=1050,
            version="v2.3.0",
        ),
        # Database — near connection limit
        "db-primary": Server(
            id="db-primary",
            cpu=60.0,
            ram=78.0,
            status=ServerStatus.degraded,
            active_connections=1150,
            version="v2.3.0",
        ),
    }

    alerts: List[Alert] = [
        Alert(
            id="ALT-001",
            severity=AlertSeverity.critical,
            message="API gateway CPU at 89% - traffic spike",
            server="api-gw-1",
            tick=0,
        ),
        Alert(
            id="ALT-002",
            severity=AlertSeverity.warning,
            message="Database connections near limit",
            server="db-primary",
            tick=0,
        ),
        Alert(
            id="ALT-003",
            severity=AlertSeverity.critical,
            message="Active connections 12000 - Black Friday surge",
            server="api-gw-1",
            tick=0,
        ),
    ]

    logs: List[LogEntry] = [
        LogEntry(
            tick=0,
            server="api-gw-1",
            level="WARN",
            message="Request queue depth 4200 — approaching saturation",
        ),
        LogEntry(
            tick=0,
            server="api-gw-2",
            level="WARN",
            message="Request queue depth 3800 — load balancer pressure high",
        ),
        LogEntry(
            tick=0,
            server="db-primary",
            level="WARN",
            message="Connection pool 91% full — 1365/1500 connections in use",
        ),
    ]

    deployment_history: List[Deployment] = [
        Deployment(version="v2.2.9", status="superseded", age_mins=2880.0),
        Deployment(version="v2.3.0", status="active", age_mins=720.0),
    ]

    return {
        "servers": servers,
        "alerts": alerts,
        "logs": logs,
        "deployment_history": deployment_history,
        "active_connections": 12_000,
        # Internal bookkeeping — tracks which gateways have been scaled up so
        # their CPU drift is suppressed.
        "_meta": {
            "scaled_gateways": set(),  # server IDs where ScaleUp was applied
            "scaled_web": set(),       # web server IDs where ScaleUp was applied
        },
    }


# ---------------------------------------------------------------------------
# simulate_tick
# ---------------------------------------------------------------------------


def simulate_tick(state: Dict[str, Any], tick_number: int) -> Dict[str, Any]:
    """
    Advance the environment by one tick.

    Effects (in order)
    ------------------
    1. active_connections += 1 500
    2. Each api-gw that hasn't been scaled up: cpu += 3 %
       - If cpu ≥ 95 % → server goes offline
    3. If active_connections > 15 000 → db-primary crashes (offline)

    A ScaleUp action applied before this tick (recorded in
    ``_meta["scaled_gateways"]``) suppresses the gateway's CPU drift for that
    tick — the gateway's CPU stabilises instead of climbing.

    Parameters
    ----------
    state:
        Mutable state dict from :func:`setup_scenario` or a previous tick.
        Modified in-place and returned.
    tick_number:
        Current tick index (1-indexed from the caller).

    Returns
    -------
    dict
        The updated state dict.
    """
    meta: Dict[str, Any] = state.get("_meta", {})
    scaled_gateways: set = meta.get("scaled_gateways", set())

    new_logs: List[LogEntry] = []
    servers: Dict[str, Server] = state["servers"]

    # 1. Connections climb
    state["active_connections"] = state.get("active_connections", 0) + _CONNECTIONS_PER_TICK

    # 2. API gateway CPU drift
    for gw_id in _GW_IDS:
        gw = servers.get(gw_id)
        if gw is None or gw.status == ServerStatus.offline:
            continue

        if gw_id in scaled_gateways:
            # Scaled up — CPU pressure relieved, drift halted
            # Bring it down slightly to show the scale-out taking effect
            stabilised_cpu = max(gw.cpu - 5.0, 45.0)
            servers[gw_id] = Server(
                id=gw_id,
                cpu=stabilised_cpu,
                ram=gw.ram,
                status=ServerStatus.degraded if stabilised_cpu < 85.0 else ServerStatus.critical,
                active_connections=gw.active_connections,
                version=gw.version,
            )
        else:
            new_cpu = gw.cpu + _GW_CPU_INCREASE_PER_TICK
            if new_cpu >= _GW_CPU_OFFLINE_THRESHOLD:
                servers[gw_id] = Server(
                    id=gw_id,
                    cpu=0.0,
                    ram=0.0,
                    status=ServerStatus.offline,
                    active_connections=0,
                    version=gw.version,
                )
                new_logs.append(LogEntry(
                    tick=tick_number,
                    server=gw_id,
                    level="CRITICAL",
                    message=(
                        f"{gw_id} CPU reached {new_cpu:.1f}% — "
                        "process crash, gateway OFFLINE"
                    ),
                ))
            else:
                servers[gw_id] = Server(
                    id=gw_id,
                    cpu=new_cpu,
                    ram=gw.ram,
                    status=ServerStatus.critical,
                    active_connections=gw.active_connections,
                    version=gw.version,
                )
                new_logs.append(LogEntry(
                    tick=tick_number,
                    server=gw_id,
                    level="ERROR",
                    message=(
                        f"{gw_id} CPU now {new_cpu:.1f}% — "
                        "still absorbing traffic surge"
                    ),
                ))

    # 3. Database crash on connection overload
    if state["active_connections"] > _DB_CRASH_CONNECTION_THRESHOLD:
        db = servers.get("db-primary")
        if db is not None and db.status != ServerStatus.offline:
            servers["db-primary"] = Server(
                id="db-primary",
                cpu=0.0,
                ram=0.0,
                status=ServerStatus.offline,
                active_connections=0,
                version=db.version,
            )
            new_logs.append(LogEntry(
                tick=tick_number,
                server="db-primary",
                level="CRITICAL",
                message=(
                    f"db-primary connection pool exhausted "
                    f"({state['active_connections']} active connections) — "
                    "database OFFLINE"
                ),
            ))

    state["logs"] = state.get("logs", []) + new_logs
    return state


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------


def grade(
    action_history: List[Action],
    final_state: Dict[str, Any],
    ticks_survived: int,
) -> float:
    """
    Score the agent using an additive rubric.

    Rubric
    ------
    +0.30  ScaleUp on any api-gateway within the first 5 actions
    +0.20  ScaleUp on any web server within the first 7 actions
    +0.30  ticks_survived == 10 (system fully survived the surge)
    +0.10  no RestartService called on a server that started healthy
    +0.10  no ScaleUp called on db-primary (wrong target)

    Total possible: 1.00 — clamped to [0.0, 1.0].

    Parameters
    ----------
    action_history:
        Ordered list of Action objects taken during the episode.
    final_state:
        State dict at episode end.
    ticks_survived:
        Number of ticks the system stayed alive.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    score: float = 0.0

    # +0.30 — ScaleUp on a gateway in first 5 actions
    first5 = action_history[:5]
    scaled_gw_early = any(
        a.action_type == ActionType.ScaleUp and a.target_id in _GW_IDS
        for a in first5
    )
    if scaled_gw_early:
        score += 0.30

    # +0.20 — ScaleUp on a web server in first 7 actions
    first7 = action_history[:7]
    scaled_web_early = any(
        a.action_type == ActionType.ScaleUp and a.target_id in _WEB_IDS
        for a in first7
    )
    if scaled_web_early:
        score += 0.20

    # +0.30 — survived all 10 ticks
    if ticks_survived >= 10:
        score += 0.30

    # +0.10 — no RestartService on a server that started healthy
    # healthy starters = web-1, web-2, web-3 (degraded under load but not failed)
    # We reward the agent for not thrashing healthy servers
    _initially_unhealthy = {*_GW_IDS, "db-primary"}  # these were critical/degraded
    restarted_healthy = any(
        a.action_type == ActionType.RestartService
        and a.target_id not in _initially_unhealthy
        for a in action_history
    )
    if not restarted_healthy:
        score += 0.10

    # +0.10 — no ScaleUp targeting the database (wrong escalation path)
    scaled_db = any(
        a.action_type == ActionType.ScaleUp and a.target_id == "db-primary"
        for a in action_history
    )
    if not scaled_db:
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

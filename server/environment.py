"""
server/environment.py
---------------------
Main OpenEnv environment class for the SRE Cloud DevOps simulation.

The ``SREEnvironment`` follows the standard Gym-style interface:

    env = SREEnvironment()
    obs = env.reset("medium")
    while True:
        action = agent.act(obs)
        step_response = env.step(action)
        obs = step_response.observation
        if step_response.done:
            break

Three built-in difficulty presets (easy / medium / hard) configure the
starting noise level, maximum episode length, and task objectives.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path fix — works when called from repo root or from inside server/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # …/server/
_ROOT = _HERE.parent                      # …/sre-devops-env/
for _p in (_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models import (  # noqa: E402
    Action,
    ActionType,
    Alert,
    AlertSeverity,
    Deployment,
    LogEntry,
    Observation,
    Reward,
    Server,
    ServerStatus,
    StepResponse,
    TaskInfo,
)
from simulator import TechCorpSimulator  # noqa: E402
from graders import TaskGrader            # noqa: E402

_grader = TaskGrader()

# Task modules (loaded lazily in reset())
import server.tasks.easy   as _task_easy    # noqa: E402
import server.tasks.medium as _task_medium  # noqa: E402
import server.tasks.hard   as _task_hard    # noqa: E402

_TASK_MODULES = {
    "easy":   _task_easy,
    "medium": _task_medium,
    "hard":   _task_hard,
}

# ---------------------------------------------------------------------------
# Task catalogue — canonical descriptions and limits
# ---------------------------------------------------------------------------

_TASKS: Dict[str, TaskInfo] = {
    "easy": TaskInfo(
        task_id="easy",
        name="The Dead Server",
        description=(
            "web-3 has crashed completely and is OFFLINE. "
            "Two red herring alerts are firing (cache-1 high memory, "
            "api-gw-1 warning) but they are NOT the problem. "
            "Find and restart the correct crashed server: web-3."
        ),
        difficulty="easy",
        max_ticks=10,
    ),
    "medium": TaskInfo(
        task_id="medium",
        name="Traffic Tsunami",
        description=(
            "Black Friday traffic surge detected. Active connections "
            "rising 1500 per tick. API gateways CPU climbing 3% per "
            "tick and will crash above 95%. Database crashes if "
            "connections exceed 15000. Scale up API gateways and "
            "web servers before the system collapses. You have 10 ticks."
        ),
        difficulty="medium",
        max_ticks=10,
    ),
    "hard": TaskInfo(
        task_id="hard",
        name="The Silent Killer",
        description=(
            "Deployment v2.3.1 (pushed 12 mins ago) introduced a "
            "memory leak. web-1 and web-2 RAM rising 8% per tick. "
            "CPU looks normal - RED HERRING. db-primary slow queries "
            "- RED HERRING. You must: (1) InvestigateLog on web-1 "
            "and web-2, (2) RollbackDeployment to v2.3.0, "
            "(3) RestartService on web-1 and web-2."
        ),
        difficulty="hard",
        max_ticks=15,
    ),
}

# ---------------------------------------------------------------------------
# Thresholds reused across reward calculation
# ---------------------------------------------------------------------------

_CPU_WARN  = 80.0
_RAM_WARN  = 85.0
_LOW_LOAD  = 40.0    # CPU below this is considered "low load"


# ---------------------------------------------------------------------------
# Helper: build a clean state snapshot dict from the simulator
# ---------------------------------------------------------------------------

def _snap(sim: TechCorpSimulator) -> Dict[str, Any]:
    """Thin wrapper around ``TechCorpSimulator.get_state()``."""
    return sim.get_state()


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class SREEnvironment:
    """
    OpenEnv-compatible SRE simulation environment.

    Parameters
    ----------
    seed : int, optional
        Random seed forwarded to the simulator for reproducibility.
        Currently informational only (simulator uses the global ``random``
        module).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed          = seed
        self._sim: Optional[TechCorpSimulator] = None
        self._task: Optional[TaskInfo]         = None

        # Episode bookkeeping
        self._episode_tick: int         = 0
        self._downtime_ticks: int       = 0
        self._consecutive_up: int       = 0   # for "hard" victory condition
        self._done: bool                = False

        # Reward-shaping state
        self._prev_alerts: List[Alert]          = []
        # Full ordered list of Action objects — passed to the task grader every step
        self._action_full_history: List[Action] = []
        # Deque kept only for the per-step repeat-penalty heuristic
        self._action_repeat_window: Deque[str]  = deque(maxlen=3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> Observation:
        """
        Reset the environment to a fresh initial state for the given task.

        Loads task info from ``server/tasks/{task_id}.py`` so that
        descriptions, max_ticks, and scenario state are always in sync
        with the task module definitions.

        Parameters
        ----------
        task_id : str
            One of ``"easy"``, ``"medium"``, or ``"hard"``.

        Returns
        -------
        Observation
            The initial observation for the new episode.

        Raises
        ------
        ValueError
            If ``task_id`` is not a recognised preset.
        """
        if task_id not in _TASKS:
            raise ValueError(
                f"Unknown task_id {task_id!r}. Choose from: {sorted(_TASKS)}"
            )

        # Load task metadata from the canonical task module
        self._task = _TASK_MODULES[task_id].get_task_info()
        # Override with our environment-level descriptions so they stay consistent
        self._task = _TASKS[task_id]

        # Fresh simulator
        self._sim              = TechCorpSimulator()

        # Reset all episode counters
        self._episode_tick     = 0
        self._downtime_ticks   = 0
        self._consecutive_up   = 0
        self._done             = False
        self._prev_alerts           = []

        # Reset action history — full list for grader, window for repeat detection
        self._action_full_history   = []
        self._action_repeat_window  = deque(maxlen=3)

        # Apply scenario-specific starting conditions
        self._apply_scenario_preset(task_id)

        # Capture the initial alert set so step() can diff against it
        self._prev_alerts = self._sim.generate_alerts()

        return self._build_observation()

    def step(self, action: Action) -> StepResponse:
        """
        Apply *action* to the environment, advance by one tick, and return
        the resulting ``StepResponse``.

        Parameters
        ----------
        action : Action
            A validated ``Action`` model instance.

        Returns
        -------
        StepResponse

        Raises
        ------
        RuntimeError
            If ``reset()`` has not been called first.
        """
        if self._sim is None or self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        # --- snapshot state BEFORE the action ----------------------------
        prev_state = _snap(self._sim)
        prev_alerts = list(self._prev_alerts)
        prev_site_up = self._sim.is_site_up()

        # --- apply action -------------------------------------------------
        action_result = self._sim.apply_action(action)

        # --- advance time -------------------------------------------------
        self._sim.tick()
        self._episode_tick += 1

        # --- snapshot state AFTER ----------------------------------------
        new_state  = _snap(self._sim)
        new_alerts = self._sim.generate_alerts()
        site_up    = self._sim.is_site_up()

        # Downtime / consecutive-up tracking
        if not site_up:
            self._downtime_ticks += 1
            self._consecutive_up  = 0
        else:
            self._downtime_ticks  = 0
            self._consecutive_up += 1

        # Update alert cache for next step's diff
        self._prev_alerts = new_alerts

        # --- record action (full list for grader, window for repeat check) -
        self._action_full_history.append(action)
        action_key = f"{action.action_type}::{action.target_id}"
        self._action_repeat_window.append(action_key)

        # --- check done ---------------------------------------------------
        done, done_reason = self._check_done(site_up)
        self._done = done

        # --- calculate reward via task grader ----------------------------
        reward = self._calculate_reward(
            action       = action,
            prev_state   = prev_state,
            new_state    = new_state,
            prev_alerts  = prev_alerts,
            new_alerts   = new_alerts,
            prev_site_up = prev_site_up,
            site_up      = site_up,
            done         = done,
        )

        # --- build observation -------------------------------------------
        obs = self._build_observation()

        return StepResponse(
            observation = obs,
            reward      = reward,
            done        = done,
            info        = {
                "action_result" : action_result,
                "done_reason"   : done_reason,
                "episode_tick"  : self._episode_tick,
                "downtime_ticks": self._downtime_ticks,
                "consecutive_up": self._consecutive_up,
                "actions_taken" : len(self._action_full_history),
            },
        )

    def state(self) -> Observation:
        """
        Return the current environment observation without advancing time.

        Returns
        -------
        Observation

        Raises
        ------
        RuntimeError
            If ``reset()`` has not been called first.
        """
        if self._sim is None:
            raise RuntimeError("Call reset() before state().")
        return self._build_observation()

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        action: Action,
        prev_state: Dict[str, Any],
        new_state: Dict[str, Any],
        prev_alerts: List[Alert],
        new_alerts: List[Alert],
        prev_site_up: bool,
        site_up: bool,
        done: bool,
    ) -> Reward:
        """
        Compute the step reward.

        Score source
        ------------
        The primary score comes from ``TaskGrader.grade_episode()`` which
        calls the per-task grader (easy/medium/hard) on the *full* action
        history accumulated so far.  This ensures the score reflects
        cumulative correctness not just last-step heuristics.

        Step-level adjustments (applied on top, then clamped)
        -------------------------------------------------------
        -0.50  Site went completely down this step
        -0.20  Same action+target repeated ≥3 times in last 3 steps
        -0.10  ScaleUp on a role whose average CPU was already below 40 %

        Feedback string summarises every action taken so far (narrated),
        plus any step-level penalty that fired.
        """
        assert self._task is not None

        task_id = self._task.task_id
        target  = action.target_id
        atype   = ActionType(action.action_type)

        # ── 1. Task-grader score (cumulative over full action history) ────
        try:
            grader_score = _grader.grade_episode(
                task_id        = task_id,
                action_history = self._action_full_history,
                final_state    = new_state,
                ticks_used     = self._episode_tick,
            )
        except Exception:
            grader_score = 0.0

        score     = grader_score
        breakdown: Dict[str, float] = {"grader_score": round(grader_score, 4)}
        penalties: List[str]        = []

        # ── 2. Step-level penalties (score only, never end episode) ───────

        # Penalty: site went down this step
        if prev_site_up and not site_up:
            breakdown["site_down"] = -0.50
            score -= 0.50
            penalties.append("Site went DOWN this step (-0.50)")

        # Penalty: repeated same action+target 3× in last 3 steps
        repeat_key   = f"{action.action_type}::{target}"
        repeat_count = sum(1 for k in self._action_repeat_window if k == repeat_key)
        if repeat_count >= 3:
            breakdown["repeat_penalty"] = -0.20
            score -= 0.20
            penalties.append(
                f"Repeated '{action.action_type}' on '{target}' "
                f"{repeat_count}× (-0.20)"
            )

        # Penalty: unnecessary ScaleUp under low average CPU
        prev_servers: Dict[str, Server] = prev_state["servers"]
        if atype == ActionType.ScaleUp and target in prev_servers:
            prefix = target.split("-")[0]   # e.g.  "api" from "api-gw-1"
            role_servers = [
                s for sid, s in prev_servers.items()
                if sid.startswith(prefix)
            ]
            avg_cpu = (
                sum(s.cpu for s in role_servers) / len(role_servers)
                if role_servers else 0.0
            )
            if avg_cpu < _LOW_LOAD:
                breakdown["unnecessary_scaleup"] = -0.10
                score -= 0.10
                penalties.append(
                    f"ScaleUp on '{target}' while avg CPU was "
                    f"{avg_cpu:.1f}% < {_LOW_LOAD}% (-0.10)"
                )

        # ── 3. Clamp ──────────────────────────────────────────────────────
        score = round(max(0.0, min(1.0, score)), 4)

        # ── 4. Rich feedback narrative ────────────────────────────────────
        feedback = self._build_feedback(penalties)

        return Reward(
            score       = score,
            breakdown   = breakdown,
            feedback    = feedback,
            done        = done,
            total_ticks = self._episode_tick,
        )

    def _build_feedback(self, penalties: List[str]) -> str:
        """
        Narrate what the agent has done so far in plain English,
        followed by any step-level penalty descriptions.
        """
        # Summarise action history as a readable sentence
        history = self._action_full_history
        if not history:
            summary = "No actions taken yet."
        else:
            parts: List[str] = []
            for act in history:
                aname = act.action_type
                tid   = act.target_id
                if aname == "RestartService":
                    parts.append(f"restarted {tid}")
                elif aname == "InvestigateLog":
                    parts.append(f"investigated logs on {tid}")
                elif aname == "RollbackDeployment":
                    parts.append(f"rolled back deployment to {tid}")
                elif aname == "ScaleUp":
                    parts.append(f"scaled up {tid}")
                elif aname == "ScaleDown":
                    parts.append(f"scaled down {tid}")
                elif aname == "KillProcess":
                    parts.append(f"killed process on {tid}")
                elif aname == "FlushCache":
                    parts.append(f"flushed cache on {tid}")
                elif aname == "FailoverDatabase":
                    parts.append(f"failed over database to {tid}")
                else:
                    parts.append(f"{aname} on {tid}")
            summary = "Agent has: " + ", ".join(parts) + "."

        if penalties:
            return summary + " | Penalties: " + "; ".join(penalties)
        return summary

    # ------------------------------------------------------------------
    # Episode termination logic
    # ------------------------------------------------------------------

    def _check_done(self, site_up: bool) -> Tuple[bool, str]:
        """
        Evaluate whether the episode should end.

        Episode ends ONLY when:
          1. ``self._episode_tick >= self._task.max_ticks``  (budget exhausted)
          2. ``site_uptime`` is False                        (site is down)

        Bad actions, repeated actions, and low reward scores do NOT end the
        episode early — penalties are reflected in the score only.  The agent
        always gets to try a different action on the next tick.

        Returns
        -------
        Tuple[bool, str]
            (done, reason_string)
        """
        assert self._task is not None

        # 1. Tick-limit exhausted
        if self._episode_tick >= self._task.max_ticks:
            return True, f"max_ticks ({self._task.max_ticks}) reached"

        # 2. Site completely down
        if not site_up:
            return True, (
                f"Site is DOWN (site_uptime=False) at tick {self._episode_tick}"
            )

        # Episode continues — agent may take another action
        return False, ""

    def _check_task_success(self) -> Tuple[bool, str]:
        """
        Return (True, reason) if the current task's victory condition is met.
        """
        assert self._sim  is not None
        assert self._task is not None

        task_id = self._task.task_id
        state   = _snap(self._sim)
        servers: Dict[str, Server] = state["servers"]
        alerts  = self._sim.generate_alerts()

        if task_id == "easy":
            # Win: web-3 is back online (no longer offline)
            web3 = servers.get("web-3")
            web3_online = (
                web3 is not None
                and web3.status not in (ServerStatus.offline, "offline")
            )
            no_critical = not any(
                a.severity == AlertSeverity.critical
                and a.server == "web-3"
                for a in alerts
            )
            if web3_online and no_critical:
                return True, "web-3 restored — The Dead Server task complete!"

        elif task_id == "medium":
            # Win: site still up after surviving max_ticks OR
            # all gateways no longer critical and site is up
            site_up = self._sim.is_site_up()
            gw_ok = all(
                servers.get(gid) is not None
                and servers[gid].status not in (ServerStatus.offline, "offline")
                for gid in ("api-gw-1", "api-gw-2")
            )
            if site_up and gw_ok and self._consecutive_up >= 5:
                return True, (
                    "API gateways stabilised and site survived the surge "
                    "— Traffic Tsunami task complete!"
                )

        elif task_id == "hard":
            # Win: web-1 and web-2 both online + rollback performed
            # We detect rollback indirectly: both web servers are at a
            # healthy/degraded status and not offline
            web1 = servers.get("web-1")
            web2 = servers.get("web-2")
            both_online = (
                web1 is not None
                and web2 is not None
                and web1.status not in (ServerStatus.offline, "offline")
                and web2.status not in (ServerStatus.offline, "offline")
            )
            if both_online and self._consecutive_up >= 3:
                return True, (
                    "web-1 and web-2 stable — The Silent Killer resolved!"
                )

        return False, ""

    # ------------------------------------------------------------------
    # Scenario presets
    # ------------------------------------------------------------------

    def _apply_scenario_preset(self, task_id: str) -> None:
        """
        Inject starting conditions that match the task module definitions.

        easy   — web-3 OFFLINE; api-gw-1 degraded (red herring);
                 cache-1 degraded high RAM (red herring)
        medium — api-gw-1/2 at CRITICAL CPU (89%/85%); web tier degraded;
                 connections already at 12 000
        hard   — web-1/2 have high RAM (72%/74%, leaking 8%/tick);
                 db-primary degraded slow queries (red herring);
                 deployment v2.3.1 marked as active/leaking
        """
        assert self._sim is not None
        from simulator import ServerStatus as _SS

        if task_id == "easy":
            # web-3 crashed — the target the agent must restart
            srv = self._sim._servers.get("web-3")
            if srv:
                srv.status = _SS.offline
                srv.cpu = 0.0
                srv.ram = 0.0
                srv.active_connections = 0

            # RED HERRING: api-gw-1 elevated CPU
            srv = self._sim._servers.get("api-gw-1")
            if srv:
                srv.cpu = 75.0
                srv.ram = 45.0
                srv._recompute_status()

            # RED HERRING: cache-1 high memory
            srv = self._sim._servers.get("cache-1")
            if srv:
                srv.cpu = 20.0
                srv.ram = 80.0
                srv._recompute_status()

        elif task_id == "medium":
            # API gateways at critical CPU — climbing 3%/tick
            for sid, cpu in [("api-gw-1", 89.0), ("api-gw-2", 85.0)]:
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.cpu = cpu
                    srv.ram = 70.0 if sid == "api-gw-1" else 68.0
                    srv._recompute_status()

            # Web tier degraded under load
            for sid in ("web-1", "web-2", "web-3"):
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.cpu = 75.0
                    srv.ram = 65.0
                    srv.active_connections = (
                        1200 if sid == "web-1" else
                        1100 if sid == "web-2" else 1050
                    )
                    srv._recompute_status()

            # Database near connection limit
            srv = self._sim._servers.get("db-primary")
            if srv:
                srv.cpu = 60.0
                srv.ram = 78.0
                srv.active_connections = 1150
                srv._recompute_status()

        elif task_id == "hard":
            # web-1 and web-2: leaking memory — RAM rising 8%/tick
            for sid, ram in [("web-1", 72.0), ("web-2", 74.0)]:
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.cpu = 45.0 if sid == "web-1" else 43.0
                    srv.ram = ram
                    srv.version = "v2.3.1"
                    srv._recompute_status()

            # RED HERRING: db-primary slow queries
            srv = self._sim._servers.get("db-primary")
            if srv:
                srv.cpu = 38.0
                srv.ram = 50.0
                srv.version = "v2.3.1"
                srv._recompute_status()

            # Mark the bad deployment in history
            self._sim._record_deployment("v2.3.0", "superseded", age_mins=600.0)
            self._sim._record_deployment("v2.3.1", "active", age_mins=12.0)
            self._sim._current_version = "v2.3.1"
            self._sim._previous_version = "v2.3.0"

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """
        Construct a fully-typed ``Observation`` from the current simulator state.
        """
        assert self._sim  is not None
        assert self._task is not None

        state   = _snap(self._sim)
        alerts  = self._sim.generate_alerts()
        site_up = self._sim.is_site_up()

        downtime = 0 if site_up else self._downtime_ticks

        return Observation(
            tick               = self._sim.current_tick,
            servers            = state["servers"],
            alerts             = alerts,
            logs               = state["logs"],
            deployment_history = state["deployment_history"],
            active_connections = state["total_connections"],
            site_uptime        = site_up,
            downtime_ticks     = downtime,
            task_id            = self._task.task_id,
            task_description   = self._task.description,
        )

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def task(self) -> Optional[TaskInfo]:
        """The currently loaded task, or ``None`` before the first reset."""
        return self._task

    @property
    def episode_tick(self) -> int:
        """Steps elapsed in the current episode."""
        return self._episode_tick

    @property
    def is_done(self) -> bool:
        """``True`` if the current episode has ended."""
        return self._done

    def __repr__(self) -> str:
        task_id = self._task.task_id if self._task else "None"
        return (
            f"SREEnvironment(task={task_id!r}, "
            f"tick={self._episode_tick}, done={self._done})"
        )

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



# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

_TASKS: Dict[str, TaskInfo] = {
    "easy": TaskInfo(
        task_id="easy",
        name="Basic Health Recovery",
        description=(
            "One or more servers have elevated CPU/RAM. Bring the cluster back "
            "to a fully healthy state (all servers healthy, no critical alerts) "
            "within the time limit. No fault injection beyond normal noise."
        ),
        difficulty="easy",
        max_ticks=30,
    ),
    "medium": TaskInfo(
        task_id="medium",
        name="Cascading Failure Response",
        description=(
            "The deployment is degraded: several servers are critical or offline. "
            "Restore site availability and stabilise CPU/RAM across all tiers "
            "(web, api, db). You must also handle at least one database failover "
            "without losing site_uptime."
        ),
        difficulty="medium",
        max_ticks=60,
    ),
    "hard": TaskInfo(
        task_id="hard",
        name="Full Outage — War Room",
        description=(
            "The site is partially down and a bad deployment is causing errors. "
            "You must: (1) rollback the deployment, (2) restore all offline servers, "
            "(3) maintain site_uptime for at least 10 consecutive ticks, "
            "all within the tick budget."
        ),
        difficulty="hard",
        max_ticks=100,
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
        self._action_history: Deque[str]        = deque(maxlen=10)  # last 10 action types
        self._action_repeat_window: Deque[str]  = deque(maxlen=3)   # last 3 full action keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> Observation:
        """
        Reset the environment to a fresh initial state for the given task.

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

        self._task             = _TASKS[task_id]
        self._sim              = TechCorpSimulator()
        self._episode_tick     = 0
        self._downtime_ticks   = 0
        self._consecutive_up   = 0
        self._done             = False
        self._prev_alerts      = []
        self._action_history.clear()
        self._action_repeat_window.clear()

        # Difficulty-specific scenario setup
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

        # Update action history for repeat-penalty detection
        action_key = f"{action.action_type}::{action.target_id}"
        self._action_history.append(action.action_type)
        self._action_repeat_window.append(action_key)

        # --- check done ---------------------------------------------------
        done, done_reason = self._check_done(site_up)
        self._done = done

        # --- calculate reward --------------------------------------------
        reward = self._calculate_reward(
            action      = action,
            prev_state  = prev_state,
            new_state   = new_state,
            prev_alerts = prev_alerts,
            new_alerts  = new_alerts,
            prev_site_up= prev_site_up,
            site_up     = site_up,
            done        = done,
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
        Compute the scalar reward for a single step.

        Scoring components
        ------------------
        Positive:
          +0.20  Action resolved ≥1 critical alert that existed before the step
          +0.15  Cluster is measurably more stable (fewer servers in bad status)
          +0.10  Action targeted an actually-problematic server

        Negative:
          -0.15  Restarted a server that was already healthy
          -0.20  Same action+target repeated ≥3 times consecutively
          -0.50  Site went completely down this step
          -0.10  ScaleUp issued while average CPU across the role was already low

        Baseline:
          +0.05  Awarded every step the site is up (encourages uptime maintenance)

        The raw sum is clamped to [0.0, 1.0].

        Returns
        -------
        Reward
        """
        score     = 0.0
        breakdown: Dict[str, float] = {}
        reasons:   List[str]        = []

        atype  = ActionType(action.action_type)
        target = action.target_id

        prev_servers: Dict[str, Server] = prev_state["servers"]
        new_servers:  Dict[str, Server] = new_state["servers"]

        # ── baseline: uptime reward ──────────────────────────────────────
        if site_up:
            breakdown["uptime_bonus"] = 0.05
            score += 0.05
            reasons.append("Site is up (+0.05)")

        # ── positive: resolved critical alert ───────────────────────────
        prev_crit_servers = {
            a.server for a in prev_alerts
            if a.severity == AlertSeverity.critical
        }
        new_crit_servers = {
            a.server for a in new_alerts
            if a.severity == AlertSeverity.critical
        }
        resolved_crits = prev_crit_servers - new_crit_servers

        if resolved_crits:
            breakdown["resolved_critical"] = 0.20
            score += 0.20
            reasons.append(
                f"Resolved critical alert(s) on {resolved_crits} (+0.20)"
            )

        # ── positive: cluster more stable ────────────────────────────────
        def _bad_count(servers: Dict[str, Server]) -> int:
            """Count servers not in 'healthy' status."""
            return sum(
                1 for s in servers.values()
                if s.status not in (ServerStatus.healthy, "healthy")
            )

        prev_bad = _bad_count(prev_servers)
        new_bad  = _bad_count(new_servers)

        if new_bad < prev_bad:
            breakdown["stability_gain"] = 0.15
            score += 0.15
            reasons.append(
                f"Cluster more stable: bad server count {prev_bad}→{new_bad} (+0.15)"
            )

        # ── positive: action targeted a problematic server ───────────────
        target_srv = prev_servers.get(target)
        target_was_bad = (
            target_srv is not None
            and target_srv.status not in (ServerStatus.healthy, "healthy")
        )
        if target_was_bad:
            breakdown["targeted_problem"] = 0.10
            score += 0.10
            reasons.append(
                f"Action targeted a non-healthy server '{target}' (+0.10)"
            )

        # ── negative: restarted a healthy server ─────────────────────────
        if atype == ActionType.RestartService:
            if target_srv is not None and target_srv.status in (
                ServerStatus.healthy, "healthy"
            ):
                breakdown["restart_healthy"] = -0.15
                score -= 0.15
                reasons.append(
                    f"Restarted already-healthy server '{target}' (-0.15)"
                )

        # ── negative: same action+target repeated ≥3 times lately ───────
        repeat_key  = f"{action.action_type}::{target}"
        repeat_count = sum(1 for k in self._action_repeat_window if k == repeat_key)
        # Note: the current action was already appended before this call,
        # so repeat_count == 3 means this is the 3rd+ consecutive repeat.
        if repeat_count >= 3:
            breakdown["repeat_penalty"] = -0.20
            score -= 0.20
            reasons.append(
                f"Same action+target '{repeat_key}' repeated {repeat_count}× (-0.20)"
            )

        # ── negative: site went down this step ───────────────────────────
        if prev_site_up and not site_up:
            breakdown["site_down"] = -0.50
            score -= 0.50
            reasons.append("Site went completely DOWN this step (-0.50)")

        # ── negative: unnecessary ScaleUp under low load ─────────────────
        if atype == ActionType.ScaleUp and target in prev_servers:
            role = None
            # Retrieve role from simulator internal state if possible; else
            # fall back to heuristic from server id prefix.
            for sid, srv in prev_servers.items():
                if sid == target:
                    # Server model has no role field; use id prefix heuristic
                    if target.startswith("web"):
                        role = "web"
                    elif target.startswith("api"):
                        role = "api"
                    elif target.startswith("db"):
                        role = "db"
                    elif target.startswith("cache"):
                        role = "cache"
                    break

            if role is not None:
                role_servers = [
                    s for sid, s in prev_servers.items()
                    if sid.startswith(role.split("-")[0])
                ]
                avg_cpu = (
                    sum(s.cpu for s in role_servers) / len(role_servers)
                    if role_servers else 0.0
                )
                if avg_cpu < _LOW_LOAD:
                    breakdown["unnecessary_scaleup"] = -0.10
                    score -= 0.10
                    reasons.append(
                        f"ScaleUp on '{target}' while avg CPU was low "
                        f"({avg_cpu:.1f}% < {_LOW_LOAD}%) (-0.10)"
                    )

        # ── clamp and assemble ───────────────────────────────────────────
        score = max(0.0, min(1.0, score))

        feedback = "; ".join(reasons) if reasons else "No significant events this step."

        return Reward(
            score       = round(score, 4),
            breakdown   = {k: round(v, 4) for k, v in breakdown.items()},
            feedback    = feedback,
            done        = done,
            total_ticks = self._episode_tick,
        )

    # ------------------------------------------------------------------
    # Episode termination logic
    # ------------------------------------------------------------------

    def _check_done(self, site_up: bool) -> Tuple[bool, str]:
        """
        Evaluate whether the episode should end.

        Returns
        -------
        Tuple[bool, str]
            (done, reason_string)
        """
        assert self._task is not None

        # 1. Tick limit reached
        if self._episode_tick >= self._task.max_ticks:
            return True, f"max_ticks ({self._task.max_ticks}) reached"

        # 2. Site completely down (all web or all DB offline)
        if not site_up and self._downtime_ticks >= 5:
            return True, (
                f"Site has been down for {self._downtime_ticks} consecutive ticks"
            )

        # 3. Task-specific success conditions
        done, reason = self._check_task_success()
        if done:
            return True, reason

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
            # Win: all servers healthy, no critical alerts
            all_healthy  = all(
                s.status in (ServerStatus.healthy, "healthy")
                for s in servers.values()
            )
            no_critical  = not any(
                a.severity == AlertSeverity.critical for a in alerts
            )
            if all_healthy and no_critical:
                return True, "All servers healthy and no critical alerts — task complete!"

        elif task_id == "medium":
            # Win: site up + no critical alerts + at least one DB failover occurred
            site_up     = self._sim.is_site_up()
            no_critical = not any(
                a.severity == AlertSeverity.critical for a in alerts
            )
            if site_up and no_critical and self._consecutive_up >= 5:
                return True, (
                    "Site stable for 5+ ticks with no critical alerts — task complete!"
                )

        elif task_id == "hard":
            # Win: site up for ≥10 consecutive ticks + all servers not offline
            no_offline = all(
                s.status not in (ServerStatus.offline, "offline")
                for s in servers.values()
            )
            if self._consecutive_up >= 10 and no_offline:
                return True, (
                    "Site up for 10+ consecutive ticks with all nodes online "
                    "— war room resolved!"
                )

        return False, ""

    # ------------------------------------------------------------------
    # Scenario presets
    # ------------------------------------------------------------------

    def _apply_scenario_preset(self, task_id: str) -> None:
        """
        Inject starting conditions appropriate for the chosen difficulty.

        Mutates the freshly-created ``self._sim`` directly via its internal
        ``_servers`` dict (white-box access within the same package).
        """
        assert self._sim is not None
        from simulator import ServerStatus as _SS  # local alias for clarity

        if task_id == "easy":
            # Elevate one web server and one API gateway to degraded
            for sid in ("web-2", "api-gw-1"):
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.cpu = 82.0
                    srv.ram = 78.0
                    srv._recompute_status()

        elif task_id == "medium":
            # Take web-3 and db-replica offline; push others to critical
            for sid in ("web-3", "db-replica"):
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.status = _SS.offline

            for sid in ("web-1", "api-gw-2"):
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.cpu = 91.0
                    srv.ram = 88.0
                    srv._recompute_status()

        elif task_id == "hard":
            # Take web-2, web-3, and db-primary offline; everything else critical
            for sid in ("web-2", "web-3", "db-primary"):
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.status = _SS.offline

            for sid in ("web-1", "api-gw-1", "api-gw-2", "cache-1"):
                srv = self._sim._servers.get(sid)
                if srv:
                    srv.cpu = 93.0
                    srv.ram = 91.0
                    srv._recompute_status()

            # Also mark the current deployment version as "failed" to hint rollback
            self._sim._record_deployment("v2.4.1", "failed", age_mins=2.0)

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

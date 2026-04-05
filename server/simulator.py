"""
server/simulator.py
-------------------
Core simulation engine for the SRE OpenEnv environment.

``SRESimulator`` manages a single episode: it loads a task scenario, applies
agent actions to mutate the server state, advances the simulation clock, and
returns structured Observation / StepResponse objects after each step.

Supported tasks and their unique behaviours
------------------------------------------
easy   — no per-tick simulation; just applies action effects and grades.
medium — calls ``medium.simulate_tick()`` after every action (connections
         rise, gateway CPU climbs, db may crash).
hard   — calls ``hard.simulate_tick()`` after every action (web-1/web-2 RAM
         climbs until rollback + restart remediation is applied).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _restart_server(servers: Dict[str, Server], target_id: str) -> Optional[str]:
    """Reset a server to a healthy idle state. Returns feedback string or None."""
    srv = servers.get(target_id)
    if srv is None:
        return f"Unknown server '{target_id}' — no action taken."
    was_offline = srv.status == ServerStatus.offline
    servers[target_id] = Server(
        id=target_id,
        cpu=25.0,
        ram=30.0,
        status=ServerStatus.healthy,
        active_connections=max(srv.active_connections - 100, 50),
        version=srv.version,
    )
    if was_offline:
        return f"{target_id} restarted and is back ONLINE."
    return f"{target_id} restarted successfully (was already running)."


# ---------------------------------------------------------------------------
# SRESimulator
# ---------------------------------------------------------------------------

class SRESimulator:
    """
    Stateful single-episode simulator.

    Typical flow::

        sim = SRESimulator()
        obs = sim.reset("easy")          # loads scenario, returns Observation
        step = sim.step(action)          # returns StepResponse
        ...                              # repeat until step.done is True
    """

    def __init__(self) -> None:
        self.task_id:       Optional[str]  = None
        self._task_module:  Any            = None
        self._task_info:    Optional[TaskInfo] = None
        self.state:         Optional[Dict[str, Any]] = None
        self.action_history: List[Action] = []
        self.tick:          int  = 0
        self.done:          bool = False
        self.downtime_ticks: int = 0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Load a task scenario and return the initial Observation.

        Parameters
        ----------
        task_id : {"easy", "medium", "hard"}

        Returns
        -------
        Observation
        """
        tid = task_id.strip().lower()
        try:
            from server.tasks import TASK_REGISTRY
        except ModuleNotFoundError:
            from tasks import TASK_REGISTRY  # type: ignore[no-redef]

        if tid not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id {task_id!r}. "
                f"Available: {sorted(TASK_REGISTRY.keys())}"
            )

        self.task_id        = tid
        self._task_module   = TASK_REGISTRY[tid]
        self._task_info     = self._task_module.get_task_info()
        self.state          = self._task_module.setup_scenario()
        self.action_history = []
        self.tick           = 0
        self.done           = False
        self.downtime_ticks = 0

        return self._build_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResponse:
        """
        Apply *action*, advance the simulation, and return a StepResponse.

        Parameters
        ----------
        action : Action

        Returns
        -------
        StepResponse
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new one.")

        self.tick += 1
        self.action_history.append(action)

        # 1. Apply action effects to state
        feedback = self._apply_action(action)

        # 2. Advance tick simulation (medium / hard only)
        sim_tick = getattr(self._task_module, "simulate_tick", None)
        if callable(sim_tick):
            sim_tick(self.state, self.tick)

        # 3. Update downtime counter
        site_up = self._is_site_up()
        if site_up:
            self.downtime_ticks = 0
        else:
            self.downtime_ticks += 1

        # 4. Compute grade for entire history so far
        score = self._compute_score()

        # 5. Check episode termination
        self.done = self._is_done(score)

        # 6. Build structured response
        obs    = self._build_observation()
        reward = Reward(
            score=score,
            breakdown={"task_score": score},
            feedback=feedback,
            done=self.done,
            total_ticks=self.tick,
        )

        return StepResponse(observation=obs, reward=reward, done=self.done, info={
            "tick": self.tick,
            "task_id": self.task_id,
            "site_up": site_up,
        })

    # ------------------------------------------------------------------
    # _apply_action
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> str:
        """
        Mutate ``self.state`` based on the action and return a feedback string.
        """
        atype   = action.action_type
        target  = action.target_id
        servers: Dict[str, Server] = self.state["servers"]
        meta:   Dict[str, Any]     = self.state.get("_meta", {})

        # ── EASY: "The Dead Server" ────────────────────────────────────────
        if self.task_id == "easy":
            if atype == ActionType.RestartService:
                if target == "web-3":
                    msg = _restart_server(servers, "web-3")
                    return f"RESOLVED: {msg} The Dead Server is back online."
                else:
                    _restart_server(servers, target)
                    return (
                        f"Restarted {target} — but web-3 is STILL OFFLINE. "
                        "That is the critical alert to address!"
                    )
            if atype == ActionType.InvestigateLog:
                return (
                    f"Logs on {target} examined. "
                    "Note: web-3 shows NO RESPONSE on any port — it is completely offline. "
                    "Restart web-3 to resolve the incident."
                )
            return (
                f"{atype} on {target} executed. "
                "web-3 remains offline — restart it to fix the incident."
            )

        # ── MEDIUM: "Traffic Tsunami" ──────────────────────────────────────
        if self.task_id == "medium":
            gw_ids  = {"api-gw-1", "api-gw-2"}
            web_ids = {"web-1", "web-2", "web-3"}

            if atype == ActionType.ScaleUp:
                if target in gw_ids:
                    meta.setdefault("scaled_gateways", set()).add(target)
                    meta["gateway_fixed"] = True
                    return (
                        f"ScaleUp applied to {target}. CPU drift suppressed — "
                        "capacity is expanding to absorb the traffic surge."
                    )
                if target in web_ids:
                    meta.setdefault("scaled_web", set()).add(target)
                    return (
                        f"Web tier scaled up on {target}. "
                        "Ensure gateways are also scaled to handle incoming load."
                    )
                return f"ScaleUp on {target} — this resource does not benefit from scaling in this scenario."

            if atype == ActionType.KillProcess:
                if target in gw_ids:
                    meta.setdefault("scaled_gateways", set()).add(target)
                    meta["gateway_fixed"] = True
                    return (
                        f"Runaway process killed on {target}. "
                        "CPU load dropping — gateway is recovering."
                    )
                return f"KillProcess on {target} executed. No runaway process found there."

            if atype == ActionType.RestartService:
                msg = _restart_server(servers, target)
                return f"{msg} Connections are still rising — scale up gateways to keep up."

            if atype == ActionType.InvestigateLog:
                if target in gw_ids:
                    return (
                        f"Logs on {target}: request queue depth critical, "
                        "CPU-bound. Scale up or kill the runaway route-worker process."
                    )
                return f"Logs on {target} examined. Focus on the overloaded gateways."

            return (
                f"{atype} on {target} executed. "
                "Traffic still rising — prioritise ScaleUp on api-gw-1 and api-gw-2."
            )

        # ── HARD: "The Silent Killer" ──────────────────────────────────────
        if self.task_id == "hard":
            leak_stopped: bool = meta.get("leak_stopped", False)
            restarted: set     = meta.get("restarted", set())

            if atype == ActionType.InvestigateLog:
                if target in ("web-1", "web-2"):
                    return (
                        f"CRITICAL FINDING on {target}: OutOfMemoryException — "
                        "heap exhaustion from deployment v2.3.1. "
                        "Rollback to v2.3.0 and restart affected servers."
                    )
                if target == "db-primary":
                    return (
                        "db-primary slow queries are a downstream symptom of the web-tier "
                        "memory pressure — fix the root cause first."
                    )
                return f"Logs on {target} examined. Focus on web-1 and web-2 OOM errors."

            if atype == ActionType.RollbackDeployment:
                if target in ("v2.3.0", "v2.3.1"):
                    meta["leak_stopped"] = True
                    # Update deployment history to reflect rollback
                    history: List[Deployment] = self.state.get("deployment_history", [])
                    history.append(Deployment(version="v2.3.0", status="active", age_mins=0.0))
                    return (
                        "Deployment v2.3.1 rolled back to v2.3.0. "
                        "Memory leak HALTED. Now restart web-1 and web-2 to clear accumulated RAM."
                    )
                return f"RollbackDeployment to {target} — version not found in history."

            if atype == ActionType.RestartService:
                if target in ("web-1", "web-2"):
                    meta.setdefault("restarted", set()).add(target)
                    msg = _restart_server(servers, target)
                    if not leak_stopped:
                        return (
                            f"{msg} WARNING: memory leak is still active (v2.3.1 is running). "
                            "RAM will climb again — rollback the deployment first!"
                        )
                    return f"{msg} RAM cleared to baseline. Memory leak is stopped."
                msg = _restart_server(servers, target)
                return f"{msg}"

            return (
                f"{atype} on {target} executed. "
                "Focus: InvestigateLog web-1/web-2, then RollbackDeployment, then RestartService."
            )

        # Fallback (should not happen with validated task_id)
        return f"{atype} on {target} executed."

    # ------------------------------------------------------------------
    # _compute_score
    # ------------------------------------------------------------------

    def _compute_score(self) -> float:
        """Grade the current action history and return a normalised score."""
        try:
            try:
                from server.graders import run_grader
            except ModuleNotFoundError:
                from graders import run_grader  # type: ignore[no-redef]

            return run_grader(
                task_id=self.task_id,
                action_history=self.action_history,
                final_state=self.state,
                ticks_used=self.tick,
            )
        except Exception as exc:
            print(f"[simulator] grade error: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # _is_done
    # ------------------------------------------------------------------

    def _is_done(self, score: float) -> bool:
        """Return True when the episode should terminate."""
        max_ticks = self._task_info.max_ticks if self._task_info else 20
        if self.tick >= max_ticks:
            return True

        servers = self.state["servers"]
        meta    = self.state.get("_meta", {})

        if self.task_id == "easy":
            # Done once web-3 is back online
            return servers.get("web-3", Server(
                id="web-3", cpu=0, ram=0,
                status=ServerStatus.offline,
                active_connections=0, version=""
            )).status != ServerStatus.offline

        if self.task_id == "medium":
            # Done when system collapses completely
            gw1_down = servers.get("api-gw-1", Server(id="api-gw-1", cpu=0, ram=0, status=ServerStatus.offline, active_connections=0, version="")).status == ServerStatus.offline
            gw2_down = servers.get("api-gw-2", Server(id="api-gw-2", cpu=0, ram=0, status=ServerStatus.offline, active_connections=0, version="")).status == ServerStatus.offline
            db_down  = servers.get("db-primary", Server(id="db-primary", cpu=0, ram=0, status=ServerStatus.offline, active_connections=0, version="")).status == ServerStatus.offline
            if gw1_down and gw2_down and db_down:
                return True
            return False

        if self.task_id == "hard":
            # Done once rollback applied AND both affected servers restarted
            restarted: set = meta.get("restarted", set())
            leak_stopped: bool = meta.get("leak_stopped", False)
            return leak_stopped and "web-1" in restarted and "web-2" in restarted

        return False

    # ------------------------------------------------------------------
    # _is_site_up
    # ------------------------------------------------------------------

    def _is_site_up(self) -> bool:
        """Return True if the externally-visible site is reachable."""
        servers = self.state.get("servers", {})
        # Site is up if at least one web server is online
        web_ids = {"web-1", "web-2", "web-3"}
        return any(
            s.status != ServerStatus.offline
            for sid, s in servers.items()
            if sid in web_ids
        )

    # ------------------------------------------------------------------
    # _build_observation
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct an Observation from the current simulation state."""
        state   = self.state
        servers = state["servers"]

        # Deployment history — normalise easy's single Deployment vs list
        if "deployment_history" in state:
            deployment_history = state["deployment_history"]
        elif "deployment" in state:
            deployment_history = [state["deployment"]]
        else:
            deployment_history = []

        # Cluster-wide active connections
        if "active_connections" in state:
            active_connections = state["active_connections"]
        else:
            active_connections = sum(
                s.active_connections for s in servers.values()
            )

        logs   = state.get("logs", [])
        alerts = state.get("alerts", [])

        site_up = self._is_site_up()

        return Observation(
            tick=self.tick,
            servers=servers,
            alerts=alerts,
            logs=logs[-20:],           # cap to last 20 log entries for brevity
            deployment_history=deployment_history,
            active_connections=active_connections,
            site_uptime=site_up,
            downtime_ticks=0 if site_up else self.downtime_ticks,
            task_id=self.task_id,
            task_description=self._task_info.description,
        )

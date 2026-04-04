"""
server/simulator.py
-------------------
Simulates the fake "TechCorp" cloud infrastructure for the SRE OpenEnv.

Cluster layout
~~~~~~~~~~~~~~
  Web tier  : web-1, web-2, web-3
  API tier   : api-gw-1, api-gw-2
  Database   : db-primary (RW), db-replica (RO)
  Cache      : cache-1

The simulator is intentionally *not* thread-safe; call it from a single
thread / coroutine and wrap with a lock if you need concurrency.
"""

from __future__ import annotations

import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path fix: allow ``from models import ...`` when running from repo root or
# from inside the server/ sub-directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # …/server/
_ROOT = _HERE.parent                              # …/sre-devops-env/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import (  # noqa: E402  (after sys.path patch)
    Action,
    ActionType,
    Alert,
    AlertSeverity,
    Deployment,
    LogEntry,
    Server,
    ServerStatus,
)


# ---------------------------------------------------------------------------
# Internal mutable server state (not a Pydantic model — mutated in-place)
# ---------------------------------------------------------------------------

class _ServerState:
    """Mutable runtime state for one simulated server node."""

    # Thresholds that drive status classification
    CPU_DEGRADED  = 70.0
    CPU_CRITICAL  = 90.0
    RAM_DEGRADED  = 75.0
    RAM_CRITICAL  = 90.0

    def __init__(
        self,
        server_id: str,
        cpu: float,
        ram: float,
        connections: int,
        version: str,
        role: str,
    ) -> None:
        self.id               = server_id
        self.cpu              = cpu
        self.ram              = ram
        self.active_connections = connections
        self.version          = version
        self.role             = role          # "web" | "api" | "db" | "cache"
        self.status           = ServerStatus.healthy
        self._recompute_status()

    # ------------------------------------------------------------------ helpers

    def _recompute_status(self) -> None:
        """Derive ServerStatus from current metrics."""
        if self.status == ServerStatus.offline:
            return  # offline stays offline until a RestartService
        if self.cpu >= self.CPU_CRITICAL or self.ram >= self.RAM_CRITICAL:
            self.status = ServerStatus.critical
        elif self.cpu >= self.CPU_DEGRADED or self.ram >= self.RAM_DEGRADED:
            self.status = ServerStatus.degraded
        else:
            self.status = ServerStatus.healthy

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, value))

    # ------------------------------------------------------------------ mutations

    def fluctuate(self, tick: int) -> None:
        """Apply random per-tick metric drift (skips offline nodes)."""
        if self.status == ServerStatus.offline:
            return

        # Slow sinusoidal + noise baseline keeps metrics interesting
        phase = (tick * 0.1 + hash(self.id) % 100) % (2 * 3.14159)
        import math
        sine_factor = math.sin(phase) * 3.0

        cpu_delta = random.gauss(sine_factor, 4.0)
        ram_delta = random.gauss(0.5, 2.0)          # RAM drifts upward slowly

        self.cpu = self._clamp(self.cpu + cpu_delta)
        self.ram = self._clamp(self.ram + ram_delta)

        # Connection churn (web/api nodes see higher traffic)
        if self.role in ("web", "api"):
            conn_delta = random.randint(-5, 12)
        else:
            conn_delta = random.randint(-2, 5)
        self.active_connections = max(0, self.active_connections + conn_delta)

        self._recompute_status()

    def to_model(self) -> Server:
        """Return an immutable Pydantic Server snapshot."""
        return Server(
            id=self.id,
            cpu=round(self.cpu, 2),
            ram=round(self.ram, 2),
            status=self.status,
            active_connections=self.active_connections,
            version=self.version,
        )


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class TechCorpSimulator:
    """
    Simulates TechCorp's fake cloud infrastructure.

    Usage
    -----
    >>> sim = TechCorpSimulator()
    >>> sim.tick()
    >>> alerts = sim.generate_alerts()
    >>> state  = sim.get_state()
    >>> up     = sim.is_site_up()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise all server nodes with realistic baseline metrics."""

        self._tick: int = 0
        self._alert_counter: int = 0
        self._deployment_history: List[Deployment] = []
        self._log_buffer: List[LogEntry] = []

        # Current active version across the fleet
        self._current_version: str = "v2.4.1"
        self._previous_version: str = "v2.3.9"

        # Track which DB is primary (may be swapped by FailoverDatabase)
        self._db_primary_id: str = "db-primary"
        self._db_replica_id: str = "db-replica"

        # ---- initialise servers ----------------------------------------
        self._servers: Dict[str, _ServerState] = {}

        specs: List[Tuple[str, float, float, int, str]] = [
            # id,          cpu,  ram,  conns,  role
            ("web-1",      35.0, 42.0, 120,    "web"),
            ("web-2",      38.0, 45.0, 130,    "web"),
            ("web-3",      33.0, 40.0, 110,    "web"),
            ("api-gw-1",   28.0, 38.0, 80,     "api"),
            ("api-gw-2",   30.0, 36.0, 75,     "api"),
            ("db-primary", 22.0, 55.0, 40,     "db"),
            ("db-replica", 18.0, 50.0, 20,     "db"),
            ("cache-1",    15.0, 30.0, 60,     "cache"),
        ]

        for sid, cpu, ram, conns, role in specs:
            self._servers[sid] = _ServerState(
                server_id=sid,
                cpu=cpu,
                ram=ram,
                connections=conns,
                version=self._current_version,
                role=role,
            )

        # Record the initial deployment
        self._record_deployment(self._current_version, "active", age_mins=45.0)

        self._emit_log("simulator", "INFO",
                       "TechCorpSimulator initialised — 8 nodes online.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_alert_id(self) -> str:
        self._alert_counter += 1
        return f"alert-{self._tick:04d}-{self._alert_counter:03d}"

    def _emit_log(self, server_id: str, level: str, message: str) -> None:
        self._log_buffer.append(
            LogEntry(tick=self._tick, server=server_id, level=level, message=message)
        )

    def _record_deployment(
        self, version: str, status: str, age_mins: float = 0.0
    ) -> None:
        dep = Deployment(version=version, status=status, age_mins=age_mins)
        self._deployment_history.append(dep)
        # Keep only the last 20 deployment records
        if len(self._deployment_history) > 20:
            self._deployment_history = self._deployment_history[-20:]

    def _age_deployments(self, minutes_per_tick: float = 1.0) -> None:
        """Increment age_mins on every deployment record each tick."""
        aged: List[Deployment] = []
        for dep in self._deployment_history:
            aged.append(
                Deployment(
                    version=dep.version,
                    status=dep.status,
                    age_mins=round(dep.age_mins + minutes_per_tick, 2),
                )
            )
        self._deployment_history = aged

    def _server(self, server_id: str) -> Optional[_ServerState]:
        return self._servers.get(server_id)

    def _servers_by_role(self, role: str) -> List[_ServerState]:
        return [s for s in self._servers.values() if s.role == role]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> int:
        """
        Advance the simulation by one time step.

        - Increments the tick counter.
        - Applies random metric fluctuations to every online server.
        - Ages all deployment records by 1 simulated minute.
        - Gradually increases cluster-wide traffic (connection growth).

        Returns
        -------
        int
            The new tick value after the advance.
        """
        self._tick += 1
        self._log_buffer.clear()   # fresh log buffer each tick

        # Metric fluctuation
        for srv in self._servers.values():
            srv.fluctuate(self._tick)

        # Slow traffic growth: every 10 ticks add a small connection ramp
        if self._tick % 10 == 0:
            for srv in self._servers_by_role("web"):
                srv.active_connections += random.randint(0, 8)
            for srv in self._servers_by_role("api"):
                srv.active_connections += random.randint(0, 4)

        # Rare random failure injection (1 % chance per tick per server)
        for srv in self._servers.values():
            if srv.status != ServerStatus.offline and random.random() < 0.01:
                srv.status = ServerStatus.offline
                self._emit_log(srv.id, "ERROR",
                               f"[FAULT INJECTION] {srv.id} has gone offline unexpectedly.")

        self._age_deployments()

        self._emit_log("simulator", "INFO", f"Tick {self._tick} completed.")
        return self._tick

    def generate_alerts(self) -> List[Alert]:
        """
        Inspect current server metrics and produce ``Alert`` objects for any
        threshold breaches or offline nodes.

        Rules
        -----
        - CPU > 80 % and server online  → warning (CPU > 95 % → critical)
        - RAM > 85 % and server online  → warning (RAM > 95 % → critical)
        - Server offline                → critical alert

        Returns
        -------
        List[Alert]
            Alerts raised at the current tick (may be empty).
        """
        alerts: List[Alert] = []

        for srv in self._servers.values():
            # Offline alert
            if srv.status == ServerStatus.offline:
                alerts.append(Alert(
                    id=self._next_alert_id(),
                    severity=AlertSeverity.critical,
                    message=f"{srv.id} is OFFLINE and not serving traffic.",
                    server=srv.id,
                    tick=self._tick,
                ))
                continue

            # CPU alerts
            if srv.cpu > 95.0:
                alerts.append(Alert(
                    id=self._next_alert_id(),
                    severity=AlertSeverity.critical,
                    message=(
                        f"{srv.id} CPU critically high: {srv.cpu:.1f}% "
                        "(threshold: 95%)"
                    ),
                    server=srv.id,
                    tick=self._tick,
                ))
            elif srv.cpu > 80.0:
                alerts.append(Alert(
                    id=self._next_alert_id(),
                    severity=AlertSeverity.warning,
                    message=(
                        f"{srv.id} CPU elevated: {srv.cpu:.1f}% "
                        "(threshold: 80%)"
                    ),
                    server=srv.id,
                    tick=self._tick,
                ))

            # RAM alerts
            if srv.ram > 95.0:
                alerts.append(Alert(
                    id=self._next_alert_id(),
                    severity=AlertSeverity.critical,
                    message=(
                        f"{srv.id} RAM critically high: {srv.ram:.1f}% "
                        "(threshold: 95%)"
                    ),
                    server=srv.id,
                    tick=self._tick,
                ))
            elif srv.ram > 85.0:
                alerts.append(Alert(
                    id=self._next_alert_id(),
                    severity=AlertSeverity.warning,
                    message=(
                        f"{srv.id} RAM elevated: {srv.ram:.1f}% "
                        "(threshold: 85%)"
                    ),
                    server=srv.id,
                    tick=self._tick,
                ))

        return alerts

    def apply_action(self, action: Action) -> Dict[str, Any]:
        """
        Apply an agent action to the simulated infrastructure.

        Parameters
        ----------
        action:
            A validated ``Action`` model instance.

        Returns
        -------
        Dict[str, Any]
            A result dict with at minimum ``{"success": bool, "message": str}``.
            Some actions return additional diagnostic keys.
        """
        atype = ActionType(action.action_type)
        tid   = action.target_id
        params = action.parameters or {}

        handler = {
            ActionType.RestartService     : self._action_restart_service,
            ActionType.ScaleUp            : self._action_scale_up,
            ActionType.ScaleDown          : self._action_scale_down,
            ActionType.RollbackDeployment : self._action_rollback_deployment,
            ActionType.KillProcess        : self._action_kill_process,
            ActionType.FlushCache         : self._action_flush_cache,
            ActionType.FailoverDatabase   : self._action_failover_database,
            ActionType.InvestigateLog     : self._action_investigate_log,
        }.get(atype)

        if handler is None:
            return {"success": False, "message": f"Unknown action type: {atype!r}"}

        return handler(tid, params)

    # ------------------------------------------------------------------ actions

    def _action_restart_service(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        RestartService — bring a server back online and reset its metrics
        to a healthy baseline.
        """
        srv = self._server(target_id)
        if srv is None:
            return {"success": False, "message": f"Server '{target_id}' not found."}

        prev_status = srv.status.value if hasattr(srv.status, "value") else str(srv.status)
        srv.cpu    = random.uniform(10.0, 30.0)
        srv.ram    = random.uniform(20.0, 45.0)
        srv.status = ServerStatus.healthy
        srv.active_connections = max(0, srv.active_connections // 3)

        msg = (
            f"RestartService: {target_id} restarted successfully "
            f"(was {prev_status}). CPU={srv.cpu:.1f}%, RAM={srv.ram:.1f}%."
        )
        self._emit_log(target_id, "INFO", msg)
        return {"success": True, "message": msg, "new_status": "healthy"}

    def _action_scale_up(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ScaleUp — reduce load on servers of the same role as target_id by
        distributing traffic across a (simulated) additional instance.
        """
        srv = self._server(target_id)
        if srv is None:
            return {"success": False, "message": f"Server '{target_id}' not found."}

        role = srv.role
        peers = self._servers_by_role(role)
        reduction = params.get("load_reduction_pct", 15.0)

        for peer in peers:
            if peer.status != ServerStatus.offline:
                peer.cpu  = peer._clamp(peer.cpu  - random.uniform(reduction * 0.8, reduction * 1.2))
                peer.ram  = peer._clamp(peer.ram  - random.uniform(reduction * 0.3, reduction * 0.6))
                peer.active_connections = max(0, int(peer.active_connections * 0.75))
                peer._recompute_status()

        msg = (
            f"ScaleUp: applied to {len(peers)} '{role}' node(s). "
            f"Load reduced by ~{reduction:.0f}%."
        )
        self._emit_log(target_id, "INFO", msg)
        return {"success": True, "message": msg, "affected_servers": [p.id for p in peers]}

    def _action_scale_down(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ScaleDown — consolidate load onto fewer nodes, increasing per-node
        utilisation slightly (cost-saving operation).
        """
        srv = self._server(target_id)
        if srv is None:
            return {"success": False, "message": f"Server '{target_id}' not found."}

        role = srv.role
        peers = self._servers_by_role(role)
        increase = params.get("load_increase_pct", 12.0)

        online_peers = [p for p in peers if p.status != ServerStatus.offline]
        for peer in online_peers:
            peer.cpu = peer._clamp(peer.cpu + random.uniform(increase * 0.7, increase * 1.3))
            peer.ram = peer._clamp(peer.ram + random.uniform(increase * 0.2, increase * 0.5))
            peer.active_connections = int(peer.active_connections * 1.2)
            peer._recompute_status()

        msg = (
            f"ScaleDown: consolidated load across {len(online_peers)} '{role}' "
            f"node(s). Per-node utilisation increased by ~{increase:.0f}%."
        )
        self._emit_log(target_id, "WARN", msg)
        return {"success": True, "message": msg, "affected_servers": [p.id for p in online_peers]}

    def _action_rollback_deployment(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        RollbackDeployment — revert all servers to the previous version and
        record the rollback in deployment history.
        """
        rollback_to = params.get("version", self._previous_version)

        if rollback_to == self._current_version:
            return {
                "success": False,
                "message": (
                    f"RollbackDeployment: target version '{rollback_to}' is "
                    "already the active version."
                ),
            }

        old_version = self._current_version
        self._previous_version = old_version
        self._current_version  = rollback_to

        # Update version string on all servers
        for srv in self._servers.values():
            srv.version = rollback_to
            # Rollback causes a brief CPU/RAM spike then stabilises
            if srv.status != ServerStatus.offline:
                srv.cpu = srv._clamp(srv.cpu + random.uniform(-5.0, 10.0))
                srv.ram = srv._clamp(srv.ram + random.uniform(-5.0, 5.0))
                srv._recompute_status()

        self._record_deployment(old_version, "rolled_back")
        self._record_deployment(rollback_to, "active", age_mins=0.0)

        msg = (
            f"RollbackDeployment: all servers rolled back from "
            f"{old_version} → {rollback_to}."
        )
        self._emit_log("simulator", "WARN", msg)
        return {
            "success": True,
            "message": msg,
            "from_version": old_version,
            "to_version": rollback_to,
        }

    def _action_kill_process(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        KillProcess — forcibly terminate a runaway process on target_id,
        reducing RAM by ~20 % and slightly lowering CPU.
        """
        srv = self._server(target_id)
        if srv is None:
            return {"success": False, "message": f"Server '{target_id}' not found."}
        if srv.status == ServerStatus.offline:
            return {
                "success": False,
                "message": f"KillProcess: {target_id} is offline, cannot kill process.",
            }

        pid = params.get("pid", random.randint(1000, 65535))
        ram_before = srv.ram
        cpu_before = srv.cpu

        ram_freed = random.uniform(15.0, 25.0)
        srv.ram = srv._clamp(srv.ram - ram_freed)
        srv.cpu = srv._clamp(srv.cpu - random.uniform(5.0, 15.0))
        srv._recompute_status()

        msg = (
            f"KillProcess: PID {pid} terminated on {target_id}. "
            f"RAM {ram_before:.1f}% → {srv.ram:.1f}% "
            f"(freed {ram_freed:.1f}%). "
            f"CPU {cpu_before:.1f}% → {srv.cpu:.1f}%."
        )
        self._emit_log(target_id, "INFO", msg)
        return {
            "success": True,
            "message": msg,
            "pid": pid,
            "ram_freed_pct": round(ram_freed, 2),
        }

    def _action_flush_cache(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        FlushCache — wipe the cache-1 node, resetting its metrics to a cold
        baseline.  target_id should be ``"cache-1"`` but the action is
        gracefully rejected for any other target.
        """
        cache = self._server("cache-1")
        if cache is None:
            return {"success": False, "message": "cache-1 not found in cluster."}
        if target_id != "cache-1":
            return {
                "success": False,
                "message": (
                    f"FlushCache: target '{target_id}' is not the cache server. "
                    "Use target_id='cache-1'."
                ),
            }
        if cache.status == ServerStatus.offline:
            return {
                "success": False,
                "message": "FlushCache: cache-1 is offline, cannot flush.",
            }

        cache.cpu              = random.uniform(5.0, 15.0)
        cache.ram              = random.uniform(10.0, 20.0)
        cache.active_connections = 0
        cache.status           = ServerStatus.healthy

        msg = (
            "FlushCache: cache-1 flushed successfully. "
            f"CPU={cache.cpu:.1f}%, RAM={cache.ram:.1f}%, connections=0."
        )
        self._emit_log("cache-1", "INFO", msg)
        return {"success": True, "message": msg}

    def _action_failover_database(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        FailoverDatabase — promote the replica to primary and demote the
        current primary to replica (or offline if it is already down).
        """
        primary = self._server(self._db_primary_id)
        replica = self._server(self._db_replica_id)

        if primary is None or replica is None:
            return {
                "success": False,
                "message": "FailoverDatabase: could not locate both DB nodes.",
            }
        if replica.status == ServerStatus.offline:
            return {
                "success": False,
                "message": (
                    "FailoverDatabase: replica is offline — "
                    "failover would result in data loss. Aborting."
                ),
            }

        old_primary_id = self._db_primary_id
        old_replica_id = self._db_replica_id

        # Swap logical roles (ids stay the same, roles swap)
        primary.role, replica.role = replica.role, primary.role

        # After failover the (now) primary takes on a promotion spike
        replica.cpu = replica._clamp(replica.cpu + random.uniform(10.0, 20.0))
        replica.ram = replica._clamp(replica.ram + random.uniform(5.0, 10.0))
        replica.active_connections += primary.active_connections
        replica._recompute_status()

        # The demoted node drains connections
        primary.active_connections = max(0, primary.active_connections // 4)
        if primary.status != ServerStatus.offline:
            primary.cpu = primary._clamp(primary.cpu - random.uniform(10.0, 15.0))
            primary._recompute_status()

        # Swap tracker IDs
        self._db_primary_id, self._db_replica_id = old_replica_id, old_primary_id

        msg = (
            f"FailoverDatabase: {old_replica_id} promoted to PRIMARY; "
            f"{old_primary_id} demoted to REPLICA."
        )
        self._emit_log("simulator", "WARN", msg)
        return {
            "success": True,
            "message": msg,
            "new_primary": self._db_primary_id,
            "new_replica": self._db_replica_id,
        }

    def _action_investigate_log(
        self, target_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        InvestigateLog — synthesise a detailed diagnostic log dump for the
        target server based on its current metrics.
        """
        srv = self._server(target_id)
        if srv is None:
            return {"success": False, "message": f"Server '{target_id}' not found."}

        status_str = (
            srv.status.value if hasattr(srv.status, "value") else str(srv.status)
        )

        # Build a realistic log tail
        lines: List[str] = [
            f"=== Diagnostic log for {target_id} (tick {self._tick}) ===",
            f"  status            : {status_str}",
            f"  version           : {srv.version}",
            f"  cpu_util          : {srv.cpu:.2f}%",
            f"  ram_util          : {srv.ram:.2f}%",
            f"  active_connections: {srv.active_connections}",
        ]

        # Append contextual messages based on health
        if srv.status == ServerStatus.offline:
            lines += [
                "  [CRIT] Node is OFFLINE. No recent log data available.",
                "  [CRIT] Check hardware / hypervisor / network path.",
            ]
        else:
            if srv.cpu > 80:
                lines.append(
                    f"  [WARN] High CPU detected ({srv.cpu:.1f}%). "
                    "Possible runaway process or traffic spike."
                )
            if srv.ram > 85:
                lines.append(
                    f"  [WARN] High RAM detected ({srv.ram:.1f}%). "
                    "Consider KillProcess or ScaleUp."
                )
            lines += [
                f"  [INFO] Last GC cycle: {random.randint(1, 30)}s ago.",
                f"  [INFO] p99 response time: {random.randint(50, 500)}ms.",
                f"  [INFO] Error rate (1 min): {random.uniform(0.0, 5.0):.2f}%.",
                f"  [INFO] Disk I/O wait: {random.uniform(0.1, 8.0):.2f}%.",
            ]

        log_text = "\n".join(lines)

        # Also emit as a structured LogEntry so it appears in the tick buffer
        self._emit_log(target_id, "INFO",
                       f"InvestigateLog requested — see 'log_detail' in response.")

        return {
            "success": True,
            "message": f"Diagnostic log retrieved for {target_id}.",
            "server_id": target_id,
            "log_detail": log_text,
        }

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """
        Return a complete snapshot of the simulated cluster.

        Returns
        -------
        Dict[str, Any]
            Keys:
            - ``"tick"``                  : current tick
            - ``"servers"``               : dict[server_id → Server model]
            - ``"deployment_history"``    : list[Deployment]
            - ``"logs"``                  : list[LogEntry] (since last tick)
            - ``"db_primary"``            : id of the current primary DB node
            - ``"db_replica"``            : id of the current replica DB node
            - ``"total_connections"``     : sum of connections across cluster
            - ``"site_up"``               : bool from is_site_up()
        """
        server_models = {sid: srv.to_model() for sid, srv in self._servers.items()}
        total_conns   = sum(srv.active_connections for srv in self._servers.values())

        return {
            "tick"              : self._tick,
            "servers"           : server_models,
            "deployment_history": list(self._deployment_history),
            "logs"              : list(self._log_buffer),
            "db_primary"        : self._db_primary_id,
            "db_replica"        : self._db_replica_id,
            "total_connections" : total_conns,
            "site_up"           : self.is_site_up(),
        }

    def is_site_up(self) -> bool:
        """
        Return ``True`` if at least one web server and at least one database
        node are online (not offline).

        Even a single healthy web node + any DB node is enough for the site
        to be considered reachable (degraded but serving).
        """
        web_online = any(
            s.status != ServerStatus.offline
            for s in self._servers_by_role("web")
        )
        db_online = any(
            s.status != ServerStatus.offline
            for s in self._servers.values()
            if s.role == "db"
        )
        return web_online and db_online

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        """Current simulation tick (read-only)."""
        return self._tick

    @property
    def current_version(self) -> str:
        """Active deployment version across the fleet."""
        return self._current_version

    @property
    def deployment_history(self) -> List[Deployment]:
        """Read-only view of deployment history (copy)."""
        return list(self._deployment_history)

    @property
    def log_buffer(self) -> List[LogEntry]:
        """Read-only view of the current tick's log buffer (copy)."""
        return list(self._log_buffer)

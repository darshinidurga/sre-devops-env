"""
models.py
---------
Pydantic data models for the SRE Cloud DevOps OpenEnv simulation environment.

These models represent the full observation/action/reward contract between the
environment and any agent (human or ML) that interacts with it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """All discrete action types an agent can take inside the environment."""

    RestartService = "RestartService"
    ScaleUp = "ScaleUp"
    ScaleDown = "ScaleDown"
    RollbackDeployment = "RollbackDeployment"
    KillProcess = "KillProcess"
    FlushCache = "FlushCache"
    FailoverDatabase = "FailoverDatabase"
    InvestigateLog = "InvestigateLog"


class AlertSeverity(str, Enum):
    """Severity levels for environment alerts, ordered by increasing urgency."""

    info = "info"
    warning = "warning"
    critical = "critical"


class ServerStatus(str, Enum):
    """Aggregate health status of a server instance."""

    healthy = "healthy"
    degraded = "degraded"
    critical = "critical"
    offline = "offline"


# ---------------------------------------------------------------------------
# Action models
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """
    A single action submitted by an agent to the environment.

    Attributes
    ----------
    action_type:
        The high-level operation to perform.
    target_id:
        Identifier of the resource the action targets (e.g. server ID,
        service name, deployment version, process PID, cache key, …).
    parameters:
        Optional free-form key/value map for action-specific arguments
        (e.g. ``{"replicas": 3}`` for ScaleUp, ``{"signal": "SIGKILL"}``
        for KillProcess).
    """

    action_type: ActionType
    target_id: str = Field(..., min_length=1, description="ID of the resource to act on")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional action-specific parameters",
    )

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Alert models
# ---------------------------------------------------------------------------


class Alert(BaseModel):
    """
    An event/notification emitted by the environment.

    Attributes
    ----------
    id:
        Unique alert identifier (e.g. UUID or monotonic counter string).
    severity:
        How urgent the alert is.
    message:
        Human-readable description of the alert condition.
    server:
        ID of the server that produced the alert.
    tick:
        Simulation tick at which the alert was raised.
    """

    id: str = Field(..., min_length=1, description="Unique alert identifier")
    severity: AlertSeverity
    message: str = Field(..., min_length=1)
    server: str = Field(..., min_length=1, description="Source server ID")
    tick: int = Field(..., ge=0, description="Simulation tick when the alert was raised")

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Server models
# ---------------------------------------------------------------------------


class Server(BaseModel):
    """
    Snapshot of a single server's runtime state.

    Attributes
    ----------
    id:
        Unique server identifier (e.g. ``"web-01"``, ``"db-primary"``).
    cpu:
        CPU utilisation percentage in the range [0, 100].
    ram:
        RAM utilisation percentage in the range [0, 100].
    status:
        Aggregate health classification.
    active_connections:
        Number of currently open client connections.
    version:
        Version string of the running application/service (e.g. ``"2.4.1"``).
    """

    id: str = Field(..., min_length=1, description="Unique server identifier")
    cpu: float = Field(..., ge=0.0, le=100.0, description="CPU utilisation (%)")
    ram: float = Field(..., ge=0.0, le=100.0, description="RAM utilisation (%)")
    status: ServerStatus
    active_connections: int = Field(..., ge=0, description="Number of open connections")
    version: str = Field(..., min_length=1, description="Running service version string")

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Deployment models
# ---------------------------------------------------------------------------


class Deployment(BaseModel):
    """
    Metadata for a single deployment revision.

    Attributes
    ----------
    version:
        Version string that was deployed (e.g. ``"v1.3.0"``).
    status:
        Current lifecycle status (e.g. ``"active"``, ``"rolling"``,
        ``"failed"``, ``"rolled_back"``).
    age_mins:
        How many minutes ago this deployment was started.
    """

    version: str = Field(..., min_length=1, description="Deployed version string")
    status: str = Field(..., min_length=1, description="Deployment lifecycle status")
    age_mins: float = Field(..., ge=0.0, description="Deployment age in minutes")


# ---------------------------------------------------------------------------
# Log models
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    """
    A single structured log line emitted by a server.

    Attributes
    ----------
    tick:
        Simulation tick at which the log line was emitted.
    server:
        ID of the server that emitted the log.
    level:
        Log level string (e.g. ``"INFO"``, ``"WARN"``, ``"ERROR"``).
    message:
        The log message body.
    """

    tick: int = Field(..., ge=0, description="Simulation tick at log emission")
    server: str = Field(..., min_length=1, description="Source server ID")
    level: str = Field(..., min_length=1, description="Log level (INFO/WARN/ERROR/…)")
    message: str = Field(..., min_length=1, description="Log message body")


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    Full environment observation returned to the agent after each step.

    Attributes
    ----------
    tick:
        Current simulation tick (monotonically increasing integer).
    servers:
        Mapping of server ID → Server snapshot for every node in the cluster.
    alerts:
        List of alerts active (or newly raised) at this tick.
    logs:
        Structured log entries emitted since the last step.
    deployment_history:
        Ordered list of recent deployments (most recent last).
    active_connections:
        Aggregate number of active connections across the entire cluster.
    site_uptime:
        ``True`` if the externally-visible site/service is reachable.
    downtime_ticks:
        Number of consecutive ticks the site has been unreachable (0 if up).
    task_id:
        Identifier for the current task/scenario being evaluated.
    task_description:
        Human-readable description of the goal the agent must accomplish.
    """

    tick: int = Field(..., ge=0, description="Current simulation tick")
    servers: Dict[str, Server] = Field(
        ...,
        description="server_id → Server snapshot mapping",
    )
    alerts: List[Alert] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    deployment_history: List[Deployment] = Field(default_factory=list)
    active_connections: int = Field(..., ge=0, description="Cluster-wide active connections")
    site_uptime: bool = Field(..., description="Whether the public site is currently reachable")
    downtime_ticks: int = Field(
        ..., ge=0, description="Consecutive ticks the site has been down"
    )
    task_id: str = Field(..., min_length=1, description="Active task/scenario identifier")
    task_description: str = Field(..., min_length=1, description="Human-readable task goal")

    @model_validator(mode="after")
    def downtime_consistent_with_uptime(self) -> "Observation":
        """downtime_ticks must be 0 when the site is reported as up."""
        if self.site_uptime and self.downtime_ticks > 0:
            raise ValueError(
                "downtime_ticks must be 0 when site_uptime is True; "
                f"got downtime_ticks={self.downtime_ticks}"
            )
        return self


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class Reward(BaseModel):
    """
    Reward signal returned after each environment step.

    Attributes
    ----------
    score:
        Normalised scalar reward in [0.0, 1.0] for this step.
    breakdown:
        Named sub-rewards that sum/contribute to ``score``
        (e.g. ``{"uptime": 0.4, "latency": 0.3, "cost": 0.3}``).
    feedback:
        Natural-language explanation of the reward (for display / debugging).
    done:
        ``True`` if the episode has ended (task success, failure, or timeout).
    total_ticks:
        Total number of ticks elapsed in the episode so far.
    """

    score: float = Field(..., ge=0.0, le=1.0, description="Normalised step reward [0, 1]")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Named sub-reward components",
    )
    feedback: str = Field(..., min_length=1, description="Human-readable reward explanation")
    done: bool = Field(..., description="Whether the episode has terminated")
    total_ticks: int = Field(..., ge=0, description="Total ticks elapsed in this episode")

    @field_validator("breakdown")
    @classmethod
    def breakdown_values_finite(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure no sub-reward value is NaN or ±Inf."""
        import math

        for key, val in v.items():
            if not math.isfinite(val):
                raise ValueError(
                    f"breakdown['{key}'] must be a finite float, got {val!r}"
                )
        return v


# ---------------------------------------------------------------------------
# Step response model
# ---------------------------------------------------------------------------


class StepResponse(BaseModel):
    """
    The complete payload returned by the environment's ``step()`` call.

    Attributes
    ----------
    observation:
        Updated environment state after applying the action.
    reward:
        Reward signal for the previous action.
    done:
        Episode termination flag (mirrors ``reward.done`` for convenience).
    info:
        Auxiliary diagnostic information (e.g. debug data, metrics).
    """

    observation: Observation
    reward: Reward
    done: bool = Field(..., description="Episode termination flag")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic / debug information",
    )

    @model_validator(mode="after")
    def done_matches_reward(self) -> "StepResponse":
        """``done`` must agree with ``reward.done`` to avoid silent bugs."""
        if self.done != self.reward.done:
            raise ValueError(
                f"StepResponse.done ({self.done}) must match reward.done ({self.reward.done})"
            )
        return self


# ---------------------------------------------------------------------------
# Task info model
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """
    Metadata describing a scenario/task available in the environment.

    Attributes
    ----------
    task_id:
        Unique identifier for the task (e.g. ``"db-failover-001"``).
    name:
        Short display name (e.g. ``"Database Failover Under Load"``).
    description:
        Detailed description of the objective, success criteria, and
        relevant context for both human operators and agents.
    difficulty:
        Difficulty label (e.g. ``"easy"``, ``"medium"``, ``"hard"``,
        ``"expert"``).
    max_ticks:
        Maximum number of ticks before the episode is force-terminated.
    """

    task_id: str = Field(..., min_length=1, description="Unique task identifier")
    name: str = Field(..., min_length=1, description="Short display name for the task")
    description: str = Field(..., min_length=1, description="Detailed task objective")
    difficulty: str = Field(
        ...,
        min_length=1,
        description="Difficulty label (easy / medium / hard / expert)",
    )
    max_ticks: int = Field(..., gt=0, description="Episode tick limit (must be > 0)")

    @field_validator("difficulty")
    @classmethod
    def difficulty_is_valid(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard", "expert"}
        if v.lower() not in allowed:
            raise ValueError(
                f"difficulty must be one of {sorted(allowed)}, got {v!r}"
            )
        return v.lower()

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class WorkflowKind(StrEnum):
    advisory = "advisory"
    code_audit = "code_audit"


class Severity(StrEnum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    informational = "informational"


class FindingStatus(StrEnum):
    confirmed_high = "confirmed_high"
    confirmed_low = "confirmed_low"
    unconfirmed = "unconfirmed"
    error = "error"
    false_positive = "false_positive"
    accepted_risk = "accepted_risk"


class AdvisoryWorkflowState(StrEnum):
    received = "received"
    triaging = "triaging"
    triage_complete = "triage_complete"
    pre_flight = "pre_flight"
    pre_flight_suspicious = "pre_flight_suspicious"
    awaiting_preflight_decision = "awaiting_preflight_decision"
    pre_flight_blocked = "pre_flight_blocked"
    building_env = "building_env"
    executing_sandbox = "executing_sandbox"
    sandbox_complete = "sandbox_complete"
    reporting = "reporting"
    awaiting_approval = "awaiting_approval"
    done = "done"
    error_triage = "error_triage"
    error_sandbox = "error_sandbox"
    error_reporting = "error_reporting"
    error_unrecoverable = "error_unrecoverable"


class SSVCAction(StrEnum):
    immediate = "immediate"
    act = "act"
    attend = "attend"
    track = "track"


class KnownStatus(StrEnum):
    duplicate = "duplicate"
    known_resolved = "known_resolved"
    known_accepted_risk = "known_accepted_risk"
    new_instance = "new_instance"


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    workflow: Mapped[WorkflowKind] = mapped_column(
        Enum(WorkflowKind, native_enum=False, length=32),
        nullable=False,
    )
    repo_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_ref: Mapped[str] = mapped_column(String(2048), nullable=False)
    severity: Mapped[Severity] = mapped_column(Enum(Severity, native_enum=False, length=32), nullable=False)
    ssvc_action: Mapped[SSVCAction | None] = mapped_column(
        Enum(SSVCAction, native_enum=False, length=32),
        nullable=True,
    )
    status: Mapped[FindingStatus] = mapped_column(
        Enum(FindingStatus, native_enum=False, length=32),
        nullable=False,
        default=FindingStatus.unconfirmed,
    )
    triage_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(512), nullable=True)
    duplicate_tracker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duplicate_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    known_status: Mapped[KnownStatus | None] = mapped_column(
        Enum(KnownStatus, native_enum=False, length=64),
        nullable=True,
    )
    cvss_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvss_vector: Mapped[str | None] = mapped_column(String(256), nullable=True)
    cve_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    cwe_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    reproduction: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    patch_available: Mapped[bool | None] = mapped_column(nullable=True)
    poc_executed: Mapped[bool | None] = mapped_column(nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workflow_runs: Mapped[list[WorkflowRun]] = relationship(back_populates="finding")


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("findings.id", ondelete="SET NULL"),
        nullable=True,
    )
    workflow_type: Mapped[WorkflowKind] = mapped_column(
        Enum(WorkflowKind, native_enum=False, length=32),
        nullable=False,
    )
    repo_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    advisory_ghsa_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(128), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    finding: Mapped[Finding | None] = relationship(back_populates="workflow_runs")
    action_logs: Mapped[list[AgentActionLog]] = relationship(back_populates="workflow_run")


class TriageDecision(StrEnum):
    approved = "approved"
    rejected = "rejected"
    escalated = "escalated"


class TriageAccuracy(Base):
    """Records the agent's triage prediction vs. the human's approval decision.

    ``outcome_signal`` is a coarse directional indicator: +1.0 (approved, agent
    prediction accepted), -1.0 (rejected, agent prediction overruled), 0.0
    (escalated, decision deferred).  It is **not** a computed distance between
    prediction and decision — richer retrospective analysis should use the raw
    ``predicted_ssvc_action`` and ``predicted_confidence`` columns alongside the
    ``human_decision`` enum.
    """

    __tablename__ = "triage_accuracy"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    finding_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("findings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workflow_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_ssvc_action: Mapped[SSVCAction | None] = mapped_column(
        Enum(SSVCAction, native_enum=False, length=32),
        nullable=True,
    )
    predicted_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    human_decision: Mapped[TriageDecision] = mapped_column(
        Enum(TriageDecision, native_enum=False, length=32),
        nullable=False,
    )
    outcome_signal: Mapped[float] = mapped_column(Float, nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(64), nullable=False)


class AgentActionLog(Base):
    __tablename__ = "agent_action_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    agent: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(256), nullable=False)
    tool_inputs: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tool_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    workflow_run: Mapped[WorkflowRun | None] = relationship(back_populates="action_logs")

# SPDX-License-Identifier: Apache-2.0
"""HTTP-side review handler.

Mirror of :mod:`agents.approval` for review decisions arriving via the
authenticated HTTP API instead of Slack interactive components. The
decision must clear the same governance gate the orchestrator consults
(:func:`agents.governance.decide_governance_tier`) and is only applied
when the most recent :class:`WorkflowRun` for the finding is still in
``awaiting_approval``.

The reviewer identity recorded in :class:`AgentActionLog`,
:class:`TriageAccuracy`, and ``Finding.approved_by`` is the opaque token
identifier (``api:<token_uuid>``) — never the raw token, owner Slack id,
or HTTP headers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.governance import GovernanceTier, decide_governance_tier
from config import AppConfig, RepoConfig
from models import (
    AdvisoryWorkflowState,
    AgentActionLog,
    Finding,
    FindingStatus,
    TriageAccuracy,
    TriageDecision,
    WorkflowRun,
)
from tools.input_sanitiser import sanitize_text

_LOG = structlog.get_logger(__name__)

_AGENT_NAME = "api_review"
_REVIEWER_PREFIX = "api:"
_CONTEXT_MAX_CHARS = 2000


class ReviewDecision(StrEnum):
    approve = "approve"
    reject = "reject"
    escalate = "escalate"


class ReviewRejectionReason(StrEnum):
    finding_not_found = "finding_not_found"
    no_pending_run = "no_pending_run"
    governance_does_not_require_approval = "governance_does_not_require_approval"


class ReviewError(Exception):
    """Raised when a review decision cannot be applied."""

    def __init__(self, reason: ReviewRejectionReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ReviewResult:
    decision: ReviewDecision
    finding_id: uuid.UUID
    workflow_run_id: uuid.UUID
    reviewer_id: str
    finding_status: FindingStatus
    workflow_state: AdvisoryWorkflowState


def reviewer_id_for_token(token_id: uuid.UUID) -> str:
    return f"{_REVIEWER_PREFIX}{token_id}"


def _find_repo(app_config: AppConfig, finding: Finding) -> RepoConfig | None:
    target = finding.repo_name.lower()
    for repo in app_config.repos.repos:
        if f"{repo.github_org}/{repo.github_repo}".lower() == target:
            return repo
    return None


async def _latest_workflow_run(session: AsyncSession, finding_id: uuid.UUID) -> WorkflowRun | None:
    stmt = (
        select(WorkflowRun).where(WorkflowRun.finding_id == finding_id).order_by(WorkflowRun.started_at.desc()).limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


def _decision_signal(decision: ReviewDecision) -> tuple[TriageDecision, float]:
    match decision:
        case ReviewDecision.approve:
            return TriageDecision.approved, 1.0
        case ReviewDecision.reject:
            return TriageDecision.rejected, -1.0
        case ReviewDecision.escalate:
            return TriageDecision.escalated, 0.0


async def apply_api_review_decision(
    session: AsyncSession,
    app_config: AppConfig,
    *,
    finding_id: uuid.UUID,
    decision: ReviewDecision,
    token_id: uuid.UUID,
    context: str | None = None,
) -> ReviewResult:
    """Apply *decision* on behalf of the API token holder.

    Mutates the finding / workflow run, writes :class:`TriageAccuracy` and
    :class:`AgentActionLog` rows, and commits. Raises :class:`ReviewError`
    if the finding does not exist, has no pending run, or its governance
    tier does not require human approval.
    """
    finding = await session.get(Finding, finding_id)
    if finding is None:
        msg = f"finding not found: {finding_id}"
        raise ReviewError(ReviewRejectionReason.finding_not_found, msg)

    repo = _find_repo(app_config, finding)
    governance = repo.governance if repo is not None else None
    tier = decide_governance_tier(finding, governance)
    if tier != GovernanceTier.approve:
        msg = f"governance tier {tier.value} does not require approval"
        raise ReviewError(ReviewRejectionReason.governance_does_not_require_approval, msg)

    run = await _latest_workflow_run(session, finding_id)
    if run is None or run.state != AdvisoryWorkflowState.awaiting_approval.value:
        msg = "finding has no workflow run awaiting approval"
        raise ReviewError(ReviewRejectionReason.no_pending_run, msg)

    reviewer_id = reviewer_id_for_token(token_id)
    sanitised_context = sanitize_text(context, max_chars=_CONTEXT_MAX_CHARS) if context else None
    now = datetime.now(UTC)
    decision_enum, outcome_signal = _decision_signal(decision)

    match decision:
        case ReviewDecision.approve:
            finding.approved_by = reviewer_id
            finding.approved_at = now
            run.state = AdvisoryWorkflowState.done.value
            run.completed_at = now
        case ReviewDecision.reject:
            finding.approved_by = reviewer_id
            finding.approved_at = now
            finding.status = FindingStatus.false_positive
            run.state = AdvisoryWorkflowState.done.value
            run.completed_at = now
        case ReviewDecision.escalate:
            pass

    session.add(
        TriageAccuracy(
            finding_id=finding.id,
            workflow_run_id=run.id,
            predicted_ssvc_action=finding.ssvc_action,
            predicted_confidence=finding.triage_confidence,
            human_decision=decision_enum,
            outcome_signal=outcome_signal,
            slack_user_id=reviewer_id,
        )
    )

    inputs: dict[str, object] = {
        "token_id": str(token_id),
        "finding_id": str(finding_id),
        "decision": decision.value,
        "repo_name": finding.repo_name,
        "governance_tier": tier.value,
    }
    if sanitised_context is not None:
        inputs["context"] = sanitised_context

    session.add(
        AgentActionLog(
            agent=_AGENT_NAME,
            tool_name=f"review.{decision.value}",
            tool_inputs=inputs,
            tool_output=decision.value,
            workflow_run_id=run.id,
        )
    )
    await session.commit()

    _LOG.info(
        "api_review_decision",
        metric_name="api_review_decision_total",
        decision=decision.value,
        finding_id=str(finding_id),
        workflow_run_id=str(run.id),
        token_id=str(token_id),
    )

    return ReviewResult(
        decision=decision,
        finding_id=finding.id,
        workflow_run_id=run.id,
        reviewer_id=reviewer_id,
        finding_status=finding.status,
        workflow_state=AdvisoryWorkflowState(run.state),
    )


__all__ = [
    "ReviewDecision",
    "ReviewError",
    "ReviewRejectionReason",
    "ReviewResult",
    "apply_api_review_decision",
    "reviewer_id_for_token",
]

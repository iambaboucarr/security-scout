"""Interactive deduplication and accepted-risk decision handler.

Owns the human resolution of dedup matches surfaced in the Slack finding message.
The four dedup actions (Confirm Duplicate / New Instance / Reopen / Confirm Resolved)
update :attr:`Finding.known_status` and may clear or reset
:attr:`Finding.duplicate_of`. The two accepted-risk actions (Still Accepted /
Re-evaluate) handle re-detection of previously accepted findings: "Still Accepted"
keeps the existing acceptance and closes the workflow run; "Re-evaluate" treats
the new finding as a distinct instance and routes it back through the standard
approval gate.

A dedup decision is independent of the standard approve/reject/escalate flow on
non-accepted-risk findings: humans can record the duplicate status without yet
making a final approval decision.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import AdvisoryWorkflowState
from models import AgentActionLog, Finding, FindingStatus, KnownStatus, WorkflowRun
from tools.slack import (
    ApprovalButtonContext,
    SlackAPIError,
    SlackClient,
    SlackMalformedResponseError,
)

if TYPE_CHECKING:
    from webhooks.slack import SlackActionId

_LOG = structlog.get_logger(__name__)


class DedupAction(StrEnum):
    confirm_duplicate = "confirm_duplicate"
    new_instance = "new_instance"
    reopen = "reopen"
    confirm_resolved = "confirm_resolved"
    risk_still_accepted = "risk_still_accepted"
    risk_reevaluate = "risk_reevaluate"


class DedupOutcome(StrEnum):
    recorded = "recorded"
    risk_still_accepted = "risk_still_accepted"
    risk_reevaluating = "risk_reevaluating"
    unknown_finding = "unknown_finding"
    unknown_run = "unknown_run"
    mismatched_finding = "mismatched_finding"


@dataclass(frozen=True, slots=True)
class DedupContext:
    finding_id: uuid.UUID
    workflow_run_id: uuid.UUID
    repo_name: str

    @classmethod
    def from_button_value(cls, encoded: str) -> DedupContext:
        decoded = ApprovalButtonContext.decode(encoded)
        return cls(
            finding_id=decoded.finding_id,
            workflow_run_id=decoded.workflow_run_id,
            repo_name=decoded.repo_name,
        )


def _now_utc() -> datetime:
    return datetime.now(UTC)


_DEDUP_ACTION_BY_ID: dict[str, DedupAction] = {
    "security_scout:dedup_confirm": DedupAction.confirm_duplicate,
    "security_scout:dedup_new_instance": DedupAction.new_instance,
    "security_scout:dedup_reopen": DedupAction.reopen,
    "security_scout:dedup_resolved": DedupAction.confirm_resolved,
    "security_scout:risk_still_accepted": DedupAction.risk_still_accepted,
    "security_scout:risk_reevaluate": DedupAction.risk_reevaluate,
}


def is_dedup_action_id(action_id: str) -> bool:
    return action_id in _DEDUP_ACTION_BY_ID


def dedup_action_from_action_id(action_id: str) -> DedupAction:
    try:
        return _DEDUP_ACTION_BY_ID[action_id]
    except KeyError as exc:
        msg = f"unknown dedup action_id: {action_id!r}"
        raise ValueError(msg) from exc


def _dedup_known_status(action: DedupAction) -> KnownStatus | None:
    match action:
        case DedupAction.confirm_duplicate:
            return KnownStatus.duplicate
        case DedupAction.new_instance:
            return KnownStatus.new_instance
        case DedupAction.confirm_resolved:
            return KnownStatus.known_resolved
        case DedupAction.risk_still_accepted:
            return KnownStatus.known_accepted_risk
        case DedupAction.reopen | DedupAction.risk_reevaluate:
            return None


def _confirmation_text(action: DedupAction, user_id: str) -> str:
    match action:
        case DedupAction.confirm_duplicate:
            return f"Marked as duplicate by <@{user_id}>."
        case DedupAction.new_instance:
            return f"Confirmed as a new instance by <@{user_id}> — proceeding."
        case DedupAction.reopen:
            return f"Reopened by <@{user_id}> — pipeline will continue."
        case DedupAction.confirm_resolved:
            return f"Marked as already resolved by <@{user_id}>."
        case DedupAction.risk_still_accepted:
            return f"Risk still accepted by <@{user_id}>; no further action."
        case DedupAction.risk_reevaluate:
            return f"Re-evaluating by <@{user_id}> — routed back through approval gate."


async def _post_thread_reply_best_effort(
    slack: SlackClient,
    *,
    channel: str,
    thread_ts: str,
    text: str,
    finding_id: str,
    workflow_run_id: uuid.UUID,
) -> None:
    try:
        await slack.post_thread_reply(
            channel,
            thread_ts=thread_ts,
            text=text,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    except (SlackAPIError, SlackMalformedResponseError) as exc:
        _LOG.warning(
            "slack_thread_reply_failed",
            metric_name="slack_thread_reply_failed_total",
            finding_id=finding_id,
            workflow_run_id=str(workflow_run_id),
            err=str(exc),
        )


async def _append_action_log(
    session: AsyncSession,
    *,
    workflow_run_id: uuid.UUID | None,
    agent: str,
    tool_name: str,
    tool_inputs: dict[str, str] | None,
    tool_output: str | None,
) -> None:
    row = AgentActionLog(
        agent=agent,
        tool_name=tool_name,
        tool_inputs=tool_inputs,
        tool_output=tool_output,
        workflow_run_id=workflow_run_id,
    )
    session.add(row)
    await session.flush()


def _emit_dedup_metrics(
    log: structlog.stdlib.BoundLogger,
    *,
    action: DedupAction,
    had_prior_match: bool,
) -> None:
    log.info(
        "dedup_human_decision",
        metric_name="dedup_human_decision",
        action=action.value,
    )
    if action == DedupAction.new_instance and had_prior_match:
        log.info(
            "dedup_false_duplicate_rate",
            metric_name="dedup_false_duplicate_rate",
            action=action.value,
        )


async def handle_slack_dedup_decision(
    session: AsyncSession,
    slack: SlackClient,
    *,
    ctx: DedupContext,
    action_id: SlackActionId,
    user_id: str,
    channel_id: str,
    message_ts: str,
) -> DedupOutcome:
    """Apply a dedup or accepted-risk decision to the finding and workflow run."""
    action = dedup_action_from_action_id(action_id.value)

    log = _LOG.bind(
        agent="dedup",
        finding_id=str(ctx.finding_id),
        workflow_run_id=str(ctx.workflow_run_id),
        slack_user=user_id,
        action=action.value,
    )

    run = await session.get(WorkflowRun, ctx.workflow_run_id)
    if run is None:
        log.warning("dedup_unknown_run", metric_name="dedup_unknown_run_total")
        await _post_thread_reply_best_effort(
            slack,
            channel=channel_id,
            thread_ts=message_ts,
            text="Could not find a matching workflow run for this finding.",
            finding_id=str(ctx.finding_id),
            workflow_run_id=ctx.workflow_run_id,
        )
        return DedupOutcome.unknown_run

    if run.finding_id != ctx.finding_id:
        log.warning("dedup_mismatched_finding", metric_name="dedup_mismatched_finding_total")
        return DedupOutcome.mismatched_finding

    finding = await session.get(Finding, ctx.finding_id)
    if finding is None:
        log.warning("dedup_unknown_finding", metric_name="dedup_unknown_finding_total")
        return DedupOutcome.unknown_finding

    had_prior_match = bool(finding.duplicate_of)

    new_status = _dedup_known_status(action)
    if new_status is not None:
        finding.known_status = new_status
    elif action == DedupAction.reopen:
        finding.known_status = None
    elif action == DedupAction.risk_reevaluate:
        finding.known_status = KnownStatus.new_instance

    outcome = DedupOutcome.recorded
    now = _now_utc()

    match action:
        case DedupAction.confirm_duplicate | DedupAction.confirm_resolved:
            # Duplicate / resolved findings are closed: no further pipeline work, no approval needed.
            if run.completed_at is None:
                run.state = AdvisoryWorkflowState.done.value
                run.completed_at = now
            finding.approved_by = user_id
            finding.approved_at = now
        case DedupAction.risk_still_accepted:
            # Re-detection of a still-accepted risk: close the run, keep the finding accepted.
            finding.status = FindingStatus.accepted_risk
            finding.approved_by = user_id
            finding.approved_at = now
            if run.completed_at is None:
                run.state = AdvisoryWorkflowState.done.value
                run.completed_at = now
            outcome = DedupOutcome.risk_still_accepted
        case DedupAction.risk_reevaluate:
            # Surface the finding through the standard approval flow on the next interaction.
            if run.state == AdvisoryWorkflowState.done.value:
                run.state = AdvisoryWorkflowState.awaiting_approval.value
                run.completed_at = None
            outcome = DedupOutcome.risk_reevaluating
        case DedupAction.new_instance | DedupAction.reopen:
            # Non-terminal: leave run state for the standard approval gate to resolve.
            pass

    await _append_action_log(
        session,
        workflow_run_id=ctx.workflow_run_id,
        agent="dedup",
        tool_name=f"dedup.{action.value}",
        tool_inputs={
            "slack_user_id": user_id,
            "finding_id": str(ctx.finding_id),
            "repo_name": ctx.repo_name,
        },
        tool_output=action.value,
    )
    await session.commit()

    _emit_dedup_metrics(log, action=action, had_prior_match=had_prior_match)
    log.info(
        "dedup_decision",
        metric_name="dedup_decision_total",
        outcome=outcome.value,
    )

    await _post_thread_reply_best_effort(
        slack,
        channel=channel_id,
        thread_ts=message_ts,
        text=_confirmation_text(action, user_id),
        finding_id=str(ctx.finding_id),
        workflow_run_id=ctx.workflow_run_id,
    )

    return outcome


__all__ = [
    "DedupAction",
    "DedupContext",
    "DedupOutcome",
    "dedup_action_from_action_id",
    "handle_slack_dedup_decision",
    "is_dedup_action_id",
]

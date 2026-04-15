# SPDX-License-Identifier: Apache-2.0
"""Interactive approval handler.

Owns the ``awaiting_approval → done`` transition of the advisory workflow. Invoked
from the Slack webhook after signature/freshness verification and payload parsing.
All three outcomes (approve, reject, escalate) append to :class:`AgentActionLog`
under the authenticated Slack user id and post a confirmation reply in the
original Slack thread. Escalation additionally DMs every configured approver.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import AdvisoryWorkflowState
from config import AppConfig, RepoConfig
from models import AgentActionLog, Finding, FindingStatus, TriageAccuracy, TriageDecision, WorkflowRun
from tools.slack import (
    ApprovalButtonContext,
    SlackAPIError,
    SlackClient,
    SlackMalformedResponseError,
)

if TYPE_CHECKING:
    from webhooks.slack import SlackActionId

_LOG = structlog.get_logger(__name__)


class ApprovalOutcome(StrEnum):
    approved = "approved"
    rejected = "rejected"
    escalated = "escalated"
    already_resolved = "already_resolved"


@dataclass(frozen=True, slots=True)
class ApprovalContext:
    finding_id: uuid.UUID
    workflow_run_id: uuid.UUID
    repo_name: str

    @classmethod
    def from_button_value(cls, encoded: str) -> ApprovalContext:
        decoded = ApprovalButtonContext.decode(encoded)
        return cls(
            finding_id=decoded.finding_id,
            workflow_run_id=decoded.workflow_run_id,
            repo_name=decoded.repo_name,
        )


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _find_repo(app_config: AppConfig, repo_name: str) -> RepoConfig | None:
    return next((r for r in app_config.repos.repos if r.name == repo_name), None)


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


async def _dm_approvers_best_effort(
    slack: SlackClient,
    repo: RepoConfig,
    *,
    text: str,
    finding_id: str,
    workflow_run_id: uuid.UUID,
    triggering_user_id: str,
) -> list[str]:
    delivered: list[str] = []
    for approver in repo.approvers:
        if approver.slack_user == triggering_user_id:
            continue
        try:
            await slack.send_dm(
                approver.slack_user,
                text=text,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
            )
        except (SlackAPIError, SlackMalformedResponseError) as exc:
            _LOG.warning(
                "slack_escalation_dm_failed",
                metric_name="slack_escalation_dm_failed_total",
                approver=approver.slack_user,
                finding_id=finding_id,
                workflow_run_id=str(workflow_run_id),
                err=str(exc),
            )
            continue
        delivered.append(approver.slack_user)
    return delivered


async def handle_slack_approval(
    session: AsyncSession,
    app_config: AppConfig,
    slack: SlackClient,
    *,
    ctx: ApprovalContext,
    action: SlackActionId,
    user_id: str,
    channel_id: str,
    message_ts: str,
) -> ApprovalOutcome:
    """Apply an interactive approval decision to the workflow.

    Returns the realised :class:`ApprovalOutcome`.  If the workflow run is not
    in the ``awaiting_approval`` state (e.g. a second click on an already-closed
    message) the function posts a short thread reply explaining that and returns
    :attr:`ApprovalOutcome.already_resolved` without mutating DB state.
    """
    # Local import to avoid circular dependency at module load.
    from webhooks.slack import SlackActionId

    log = _LOG.bind(
        agent="approval",
        finding_id=str(ctx.finding_id),
        workflow_run_id=str(ctx.workflow_run_id),
        slack_user=user_id,
        action=action.value,
    )

    run = await session.get(WorkflowRun, ctx.workflow_run_id)
    if run is None:
        log.warning("approval_unknown_run", metric_name="approval_unknown_run_total", error_kind="unknown_run")
        await _post_thread_reply_best_effort(
            slack,
            channel=channel_id,
            thread_ts=message_ts,
            text="Could not find a matching workflow run for this finding.",
            finding_id=str(ctx.finding_id),
            workflow_run_id=ctx.workflow_run_id,
        )
        return ApprovalOutcome.already_resolved

    if run.finding_id != ctx.finding_id:
        log.warning("approval_mismatched_finding", metric_name="approval_mismatched_finding_total")
        return ApprovalOutcome.already_resolved

    finding = await session.get(Finding, ctx.finding_id)
    if finding is None:
        log.warning("approval_unknown_finding", metric_name="approval_unknown_finding_total")
        return ApprovalOutcome.already_resolved

    repo = _find_repo(app_config, ctx.repo_name)
    if repo is None:
        log.warning("approval_unknown_repo", repo_name=ctx.repo_name)
        return ApprovalOutcome.already_resolved

    if run.state != AdvisoryWorkflowState.awaiting_approval.value:
        who = finding.approved_by or "a prior click"
        await _post_thread_reply_best_effort(
            slack,
            channel=channel_id,
            thread_ts=message_ts,
            text=f"This finding is already resolved by {who}; no change applied.",
            finding_id=str(ctx.finding_id),
            workflow_run_id=ctx.workflow_run_id,
        )
        log.info("approval_already_resolved", metric_name="approval_already_resolved_total")
        return ApprovalOutcome.already_resolved

    now = _now_utc()

    match action:
        case SlackActionId.approve:
            finding.approved_by = user_id
            finding.approved_at = now
            run.state = AdvisoryWorkflowState.done.value
            run.completed_at = now
            outcome = ApprovalOutcome.approved
            decision = TriageDecision.approved
            outcome_signal = 1.0
            confirmation = f"Approved by <@{user_id}>."
        case SlackActionId.reject:
            finding.approved_by = user_id
            finding.approved_at = now
            finding.status = FindingStatus.false_positive
            run.state = AdvisoryWorkflowState.done.value
            run.completed_at = now
            outcome = ApprovalOutcome.rejected
            decision = TriageDecision.rejected
            outcome_signal = -1.0
            confirmation = f"Rejected by <@{user_id}> (marked false positive)."
        case SlackActionId.escalate:
            outcome = ApprovalOutcome.escalated
            decision = TriageDecision.escalated
            outcome_signal = 0.0
            confirmation = f"Escalated by <@{user_id}>."
        case _:
            msg = f"unhandled approval action: {action}"
            raise ValueError(msg)

    if decision == TriageDecision.escalated:
        existing = (
            await session.execute(
                select(TriageAccuracy.id).where(
                    TriageAccuracy.finding_id == finding.id,
                    TriageAccuracy.slack_user_id == user_id,
                    TriageAccuracy.human_decision == TriageDecision.escalated,
                )
            )
        ).first()
        is_duplicate_escalation = existing is not None
    else:
        is_duplicate_escalation = False

    if not is_duplicate_escalation:
        session.add(
            TriageAccuracy(
                finding_id=finding.id,
                workflow_run_id=run.id,
                predicted_ssvc_action=finding.ssvc_action,
                predicted_confidence=finding.triage_confidence,
                human_decision=decision,
                outcome_signal=outcome_signal,
                slack_user_id=user_id,
            )
        )
        await session.flush()

        log.info(
            "triage_accuracy_recorded",
            metric_name="triage_accuracy_delta",
            value=outcome_signal,
            predicted_ssvc_action=finding.ssvc_action.value if finding.ssvc_action is not None else None,
            predicted_confidence=finding.triage_confidence,
            human_decision=decision.value,
        )

    await _append_action_log(
        session,
        workflow_run_id=ctx.workflow_run_id,
        agent="approval",
        tool_name=f"approval.{outcome.value}",
        tool_inputs={
            "slack_user_id": user_id,
            "finding_id": str(ctx.finding_id),
            "repo_name": ctx.repo_name,
        },
        tool_output=outcome.value,
    )
    await session.commit()

    log.info(
        "approval_decision",
        metric_name="approval_decision_total",
        outcome=outcome.value,
    )

    if outcome == ApprovalOutcome.escalated:
        dm_text = (
            f"Security Scout escalation from <@{user_id}>.\n"
            f"*Finding:* `{ctx.finding_id}` — {finding.title}\n"
            f"*Severity:* {finding.severity.value.upper()} — {finding.source_ref}"
        )
        delivered = await _dm_approvers_best_effort(
            slack,
            repo,
            text=dm_text,
            finding_id=str(ctx.finding_id),
            workflow_run_id=ctx.workflow_run_id,
            triggering_user_id=user_id,
        )
        if delivered:
            mentions = ", ".join(f"<@{u}>" for u in delivered)
            confirmation = f"{confirmation} Notified: {mentions}."
        else:
            confirmation = (
                f"{confirmation} No approvers are configured for `{ctx.repo_name}` — "
                "add `approvers:` to the repo's governance block."
            )

    await _post_thread_reply_best_effort(
        slack,
        channel=channel_id,
        thread_ts=message_ts,
        text=confirmation,
        finding_id=str(ctx.finding_id),
        workflow_run_id=ctx.workflow_run_id,
    )

    return outcome


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


__all__ = [
    "ApprovalContext",
    "ApprovalOutcome",
    "handle_slack_approval",
]

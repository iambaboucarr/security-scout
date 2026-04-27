# SPDX-License-Identifier: Apache-2.0
"""Slack-driven workflow actions beyond the main approval triad (preflight, patch oracle)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agents.approval import ApprovalContext
from config import AppConfig, RepoConfig
from models import AdvisoryWorkflowState, Finding, FindingStatus, WorkflowRun
from tools.slack import SlackAPIError, SlackClient, SlackMalformedResponseError

if TYPE_CHECKING:
    from webhooks.slack import SlackActionId

_LOG = structlog.get_logger(__name__)


def _find_repo(app_config: AppConfig, repo_name: str) -> RepoConfig | None:
    return next((r for r in app_config.repos.repos if r.name == repo_name), None)


async def _thread_reply(
    slack: SlackClient,
    *,
    channel_id: str,
    message_ts: str,
    text: str,
    finding_id: uuid.UUID,
    workflow_run_id: uuid.UUID,
) -> None:
    try:
        await slack.post_thread_reply(
            channel_id,
            thread_ts=message_ts,
            text=text,
            finding_id=str(finding_id),
            workflow_run_id=workflow_run_id,
        )
    except (SlackAPIError, SlackMalformedResponseError) as exc:
        _LOG.warning(
            "slack_thread_reply_failed",
            metric_name="slack_thread_reply_failed_total",
            err=str(exc),
            finding_id=str(finding_id),
        )


async def handle_preflight_review_decision(
    session: AsyncSession,
    app_config: AppConfig,
    slack: SlackClient,
    *,
    ctx: ApprovalContext,
    action_id: SlackActionId,
    user_id: str,
    channel_id: str,
    message_ts: str,
    enqueue_advisory: Callable[..., Awaitable[Any]] | None,
) -> None:
    from webhooks.slack import SlackActionId

    log = _LOG.bind(
        agent="preflight_review",
        finding_id=str(ctx.finding_id),
        workflow_run_id=str(ctx.workflow_run_id),
    )

    run = await session.get(WorkflowRun, ctx.workflow_run_id)
    if run is None or run.finding_id != ctx.finding_id:
        log.warning("preflight_review_unknown_run")
        return

    if run.state != AdvisoryWorkflowState.awaiting_preflight_decision.value:
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="This pre-flight review is already closed; no change applied.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return

    finding = await session.get(Finding, ctx.finding_id)
    if finding is None:
        return

    ev = finding.evidence or {}
    ghsa_raw = ev.get("ghsa_id")
    if not isinstance(ghsa_raw, str) or not ghsa_raw.strip():
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="Cannot resume: finding is missing GHSA id metadata.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return

    src_raw = ev.get("advisory_source")
    advisory_source: Literal["repository", "global"] = "global" if src_raw == "global" else "repository"

    repo = _find_repo(app_config, ctx.repo_name)
    if repo is None:
        log.warning("preflight_review_unknown_repo", repo_name=ctx.repo_name)
        return

    match action_id:
        case SlackActionId.preflight_proceed:
            if enqueue_advisory is None:
                await _thread_reply(
                    slack,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    text="Cannot queue sandbox run: background worker queue is not configured.",
                    finding_id=ctx.finding_id,
                    workflow_run_id=ctx.workflow_run_id,
                )
                return
            run.state = AdvisoryWorkflowState.building_env.value
            await session.commit()
            await enqueue_advisory(
                repo_name=ctx.repo_name,
                ghsa_id=ghsa_raw.strip(),
                advisory_source=advisory_source,
                resume_workflow_run_id=str(run.id),
            )
            await _thread_reply(
                slack,
                channel_id=channel_id,
                message_ts=message_ts,
                text=f"<@{user_id}> chose to proceed — sandbox build queued.",
                finding_id=ctx.finding_id,
                workflow_run_id=ctx.workflow_run_id,
            )
            log.info("preflight_review_proceed", slack_user=user_id)
        case SlackActionId.preflight_cancel:
            run.state = AdvisoryWorkflowState.pre_flight_blocked.value
            run.completed_at = datetime.now(UTC)
            await session.commit()
            await _thread_reply(
                slack,
                channel_id=channel_id,
                message_ts=message_ts,
                text=f"<@{user_id}> cancelled PoC execution after pre-flight review.",
                finding_id=ctx.finding_id,
                workflow_run_id=ctx.workflow_run_id,
            )
            log.info("preflight_review_cancel", slack_user=user_id)
        case _:
            log.warning("preflight_review_unknown_action", action_id=str(action_id))
            await _thread_reply(
                slack,
                channel_id=channel_id,
                message_ts=message_ts,
                text="Unrecognized pre-flight action — no change applied.",
                finding_id=ctx.finding_id,
                workflow_run_id=ctx.workflow_run_id,
            )


async def handle_patch_oracle_request(
    session: AsyncSession,
    app_config: AppConfig,
    slack: SlackClient,
    *,
    ctx: ApprovalContext,
    user_id: str,
    channel_id: str,
    message_ts: str,
    enqueue_patch_oracle: Callable[..., Awaitable[Any]] | None,
) -> None:
    log = _LOG.bind(
        agent="patch_oracle_slack",
        finding_id=str(ctx.finding_id),
        workflow_run_id=str(ctx.workflow_run_id),
    )

    finding = await session.get(Finding, ctx.finding_id)
    if finding is None:
        return

    if finding.status != FindingStatus.confirmed_low:
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="Patch oracle is only available after a CONFIRMED_LOW execution result.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return

    if not finding.patch_available:
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="Advisory has no recorded patched version — cannot run patch oracle.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return

    oracle = (finding.evidence or {}).get("oracle")
    if not isinstance(oracle, dict):
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="Patch oracle metadata missing on finding.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return
    cands = oracle.get("patched_ref_candidates")
    if not isinstance(cands, list) or not any(isinstance(x, str) and x.strip() for x in cands):
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="No patched git refs to check out — cannot run patch oracle.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return

    if _find_repo(app_config, ctx.repo_name) is None:
        log.warning("patch_oracle_unknown_repo", repo_name=ctx.repo_name)
        return

    if enqueue_patch_oracle is None:
        await _thread_reply(
            slack,
            channel_id=channel_id,
            message_ts=message_ts,
            text="Cannot queue patch oracle: background worker queue is not configured.",
            finding_id=ctx.finding_id,
            workflow_run_id=ctx.workflow_run_id,
        )
        return

    await enqueue_patch_oracle(
        repo_name=ctx.repo_name,
        finding_id=str(ctx.finding_id),
        workflow_run_id=str(ctx.workflow_run_id),
        slack_channel_id=channel_id,
        slack_message_ts=message_ts,
    )
    await _thread_reply(
        slack,
        channel_id=channel_id,
        message_ts=message_ts,
        text=f"<@{user_id}> queued patch oracle — results will post in this thread.",
        finding_id=ctx.finding_id,
        workflow_run_id=ctx.workflow_run_id,
    )
    log.info("patch_oracle_queued", slack_user=user_id)


__all__ = [
    "handle_patch_oracle_request",
    "handle_preflight_review_decision",
]

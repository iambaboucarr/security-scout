"""Orchestrator agent: event routing + advisory workflow state machine.

Tool access (least-privilege):
    ALLOWED:
        - poc_preflight.validate

    NOT ALLOWED:
        - scm.fetch_advisory, scm.read_code, scm.fetch_pr_diff, etc.
        - sast_adapter.scan
        - docker_sandbox.build / run / destroy
        - nuclei.run
        - slack.send_finding, slack.request_approval   (Slack Handler agent)

    Temporary: the orchestrator calls SlackClient directly for finding reports
    and error notifications until the Slack Handler agent is separated. Once
    interactive Slack workflows exist, Slack tool access moves to the dedicated
    handler and the orchestrator delegates via state transitions.

    Receives ``SCMProvider``, not ``GitHubClient`` directly.
    The provider is passed through to subordinate agents (triage).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agents.governance import GovernanceTier, decide_governance_tier
from agents.triage import run_advisory_triage
from ai.provider import LLMProvider
from config import RepoConfig
from exceptions import SecurityScoutError
from models import AgentActionLog, Finding, WorkflowKind, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.issue_tracker import IssueTrackerCredentials
from tools.rate_limiter import RateLimiterCircuitOpen, RateLimitExceeded, SlidingWindowRateLimiter
from tools.scm.protocol import SCMProvider
from tools.slack import (
    ApprovalButtonContext,
    SlackAPIError,
    SlackClient,
    SlackMalformedResponseError,
    finding_to_report_payload,
)

_LOG = structlog.get_logger(__name__)

_MAX_LOG_OUTPUT = 500

_RATE_LIMIT_RETRY_SECONDS = 120


class AdvisoryWorkflowState(StrEnum):
    received = "received"
    triaging = "triaging"
    triage_complete = "triage_complete"
    reporting = "reporting"
    awaiting_approval = "awaiting_approval"
    done = "done"
    error_triage = "error_triage"
    error_reporting = "error_reporting"
    error_unrecoverable = "error_unrecoverable"


@dataclass(frozen=True, slots=True)
class ScheduleRetryParams:
    """Arguments for ARQ ``enqueue_job`` with ``_defer_by`` (or equivalent delayed execution)."""

    workflow_run_id: uuid.UUID
    delay_seconds: int
    state: str
    reason: str


def _truncate_log(text: str | None, max_chars: int = _MAX_LOG_OUTPUT) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


async def _append_action_log(
    session: AsyncSession,
    *,
    workflow_run_id: uuid.UUID | None,
    agent: str,
    tool_name: str,
    tool_inputs: dict[str, Any] | None,
    tool_output: str | None,
) -> None:
    row = AgentActionLog(
        agent=agent,
        tool_name=tool_name,
        tool_inputs=tool_inputs,
        tool_output=_truncate_log(tool_output),
        workflow_run_id=workflow_run_id,
    )
    session.add(row)
    await session.flush()


def _now_utc() -> datetime:
    return datetime.now(UTC)


async def _best_effort_error_slack(
    slack: SlackClient,
    channel: str,
    *,
    title: str,
    detail: str,
    workflow_run_id: uuid.UUID | None,
    finding_id: str | None,
) -> None:
    try:
        await slack.notify_workflow_error(
            channel,
            title=title,
            detail=detail,
            workflow_run_id=workflow_run_id,
            finding_id=finding_id,
        )
    except SlackAPIError as e:
        _LOG.error(
            "workflow_error_slack_notify_failed",
            metric_name="workflow_error_total",
            workflow_run_id=str(workflow_run_id) if workflow_run_id else None,
            finding_id=finding_id,
            err=str(e),
        )


async def run_advisory_workflow(
    session: AsyncSession,
    repo: RepoConfig,
    scm: SCMProvider,
    http: httpx.AsyncClient,
    slack: SlackClient,
    *,
    ghsa_id: str,
    advisory_source: Literal["repository", "global"] = "repository",
    run_id: uuid.UUID | None = None,
    llm: LLMProvider | None = None,
    reasoning_model: str = "claude-sonnet-4-6",
    circuit_breaker: ExternalApiCircuitBreaker | None = None,
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None = None,
    resume_workflow_run_id: uuid.UUID | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    tracker_credentials: IssueTrackerCredentials | None = None,
) -> WorkflowRun:
    """Run or resume the advisory triage → Slack report workflow.

    When *resume_workflow_run_id* is ``None`` a fresh ``WorkflowRun`` is created.
    Pass an existing run's UUID to resume from where it left off (must be in
    ``triaging``, ``triage_complete``, or ``reporting`` state).  The resumed run
    keeps its original ``id``, ``started_at``, and ``retry_count``.
    """
    log = _LOG.bind(agent="orchestrator", run_id=str(run_id) if run_id else None)
    breaker = circuit_breaker or ExternalApiCircuitBreaker()

    needs_triage: bool
    run_stable_id: uuid.UUID

    if resume_workflow_run_id is None:
        workflow_started_at = _now_utc()
        run = WorkflowRun(
            workflow_type=WorkflowKind.advisory,
            state=AdvisoryWorkflowState.received.value,
            retry_count=0,
            finding_id=None,
            started_at=workflow_started_at,
        )
        session.add(run)
        await session.flush()
        run_stable_id = run.id

        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=None,
            to_state=AdvisoryWorkflowState.received.value,
            workflow_run_id=str(run_stable_id),
        )

        run.state = AdvisoryWorkflowState.triaging.value
        await session.commit()

        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.received.value,
            to_state=AdvisoryWorkflowState.triaging.value,
            workflow_run_id=str(run_stable_id),
        )
        needs_triage = True
    else:
        loaded = await _require_run(
            session,
            resume_workflow_run_id,
            missing_message="workflow run missing for resume",
        )
        if loaded.completed_at is not None:
            msg = "cannot resume a completed workflow run"
            raise RuntimeError(msg)
        if loaded.workflow_type != WorkflowKind.advisory:
            msg = "resume only supported for advisory workflows"
            raise RuntimeError(msg)
        if loaded.state not in (
            AdvisoryWorkflowState.triaging.value,
            AdvisoryWorkflowState.triage_complete.value,
            AdvisoryWorkflowState.reporting.value,
        ):
            msg = f"cannot resume from state {loaded.state!r}"
            raise RuntimeError(msg)
        run_stable_id = loaded.id
        needs_triage = loaded.state == AdvisoryWorkflowState.triaging.value

    for api in ("github", "slack"):
        if breaker.take_resume_log_event(api):
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="circuit_breaker",
                tool_inputs={"api": api, "event": "resumed"},
                tool_output=None,
            )
            await session.commit()

    finding: Finding | None = None

    if needs_triage:
        blocked_gh = breaker.blocked_seconds_remaining("github")
        if blocked_gh > 0:
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="circuit_breaker",
                tool_inputs={"api": "github", "blocked_seconds": blocked_gh},
                tool_output="API paused; deferring workflow",
            )
            await session.commit()
            if schedule_retry is None:
                msg = "schedule_retry is required when the GitHub API circuit is open"
                raise RuntimeError(msg)
            await schedule_retry(
                ScheduleRetryParams(
                    workflow_run_id=run_stable_id,
                    delay_seconds=blocked_gh,
                    state=AdvisoryWorkflowState.triaging.value,
                    reason="github_circuit_blocked",
                ),
            )
            return await _require_run(
                session,
                run_stable_id,
                missing_message="workflow run missing after GitHub circuit deferral",
            )

        try:
            finding = await run_advisory_triage(
                session,
                repo,
                scm,
                http,
                ghsa_id=ghsa_id,
                advisory_source=advisory_source,
                run_id=run_id,
                workflow_run_id=run_stable_id,
                llm=llm,
                reasoning_model=reasoning_model,
                tracker_credentials=tracker_credentials,
            )
        except SecurityScoutError as e:
            await session.rollback()
            opened = breaker.record_failure("github")
            if opened:
                await _append_action_log(
                    session,
                    workflow_run_id=run_stable_id,
                    agent="orchestrator",
                    tool_name="circuit_breaker",
                    tool_inputs={"api": "github", "event": "opened"},
                    tool_output=f"pause_seconds={breaker.PAUSE_SEC}",
                )
            run = await _require_run(
                session,
                run_stable_id,
                missing_message="workflow run missing after triage failure",
            )
            if e.is_transient and run.retry_count < 3 and schedule_retry is not None:
                delay = max(1, 2**run.retry_count)
                run.retry_count += 1
                await session.commit()
                log.warning(
                    "workflow_transient_retry",
                    metric_name="workflow_error_total",
                    phase="triage",
                    workflow_run_id=str(run_stable_id),
                    retry_count=run.retry_count,
                    delay_seconds=delay,
                )
                await schedule_retry(
                    ScheduleRetryParams(
                        workflow_run_id=run_stable_id,
                        delay_seconds=delay,
                        state=AdvisoryWorkflowState.triaging.value,
                        reason="triage_transient",
                    ),
                )
                return run
            run.state = AdvisoryWorkflowState.error_triage.value
            run.error_message = _truncate_log(str(e), 4000)
            run.completed_at = _now_utc()
            await session.commit()
            log.warning(
                "workflow_error",
                metric_name="workflow_error_total",
                phase="triage",
                workflow_run_id=str(run_stable_id),
                err=str(e),
            )
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="triage",
                tool_inputs={"ghsa_id": ghsa_id},
                tool_output=str(e),
            )
            await session.commit()
            await _best_effort_error_slack(
                slack,
                repo.slack_channel,
                title="Advisory triage failed",
                detail=str(e),
                workflow_run_id=run_stable_id,
                finding_id=None,
            )
            return run
        except Exception as e:
            await session.rollback()
            run = await _require_run(
                session,
                run_stable_id,
                missing_message="workflow run missing after triage failure",
            )
            run.state = AdvisoryWorkflowState.error_unrecoverable.value
            run.error_message = _truncate_log(str(e), 4000)
            run.completed_at = _now_utc()
            await session.commit()
            log.exception(
                "workflow_unrecoverable",
                metric_name="workflow_error_total",
                workflow_run_id=str(run_stable_id),
            )
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="triage",
                tool_inputs={"ghsa_id": ghsa_id},
                tool_output=str(e),
            )
            await session.commit()
            await _best_effort_error_slack(
                slack,
                repo.slack_channel,
                title="Advisory workflow failed (unrecoverable)",
                detail=str(e),
                workflow_run_id=run_stable_id,
                finding_id=None,
            )
            return run

        if finding is None:
            msg = "triage produced no finding"
            raise RuntimeError(msg)

        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing after triage",
        )
        run.finding_id = finding.id
        run.state = AdvisoryWorkflowState.triage_complete.value
        await session.commit()

        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.triaging.value,
            to_state=AdvisoryWorkflowState.triage_complete.value,
            workflow_run_id=str(run_stable_id),
        )
    else:
        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing before reporting resume",
        )
        if run.finding_id is None:
            msg = "resume run has no finding_id"
            raise RuntimeError(msg)
        finding = await session.get(Finding, run.finding_id)
        if finding is None:
            msg = "finding row missing for workflow resume"
            raise RuntimeError(msg)

    run = await _require_run(
        session,
        run_stable_id,
        missing_message="workflow run missing before reporting",
    )
    if run.state not in (
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.reporting.value,
    ):
        msg = f"unexpected workflow state before reporting: {run.state!r}"
        raise RuntimeError(msg)

    tier = decide_governance_tier(finding, repo.governance)
    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="orchestrator",
        tool_name="governance.decide",
        tool_inputs={
            "severity": finding.severity.value,
            "ssvc_action": finding.ssvc_action.value if finding.ssvc_action else None,
            "known_status": finding.known_status.value if finding.known_status else None,
            "has_governance_config": repo.governance is not None,
        },
        tool_output=tier.value,
    )
    await session.commit()

    if tier == GovernanceTier.auto_resolve and run.state == AdvisoryWorkflowState.triage_complete.value:
        done_at = _now_utc()
        run.state = AdvisoryWorkflowState.done.value
        run.completed_at = done_at
        await session.commit()
        log.info(
            "advisory_auto_resolved",
            metric_name="advisory_auto_resolved_total",
            finding_id=str(finding.id),
            workflow_run_id=str(run_stable_id),
        )
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.triage_complete.value,
            to_state=AdvisoryWorkflowState.done.value,
            workflow_run_id=str(run_stable_id),
        )
        return run

    if breaker.take_resume_log_event("slack"):
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="circuit_breaker",
            tool_inputs={"api": "slack", "event": "resumed"},
            tool_output=None,
        )
        await session.commit()

    blocked_sl = breaker.blocked_seconds_remaining("slack")
    if blocked_sl > 0:
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="circuit_breaker",
            tool_inputs={"api": "slack", "blocked_seconds": blocked_sl},
            tool_output="API paused; deferring workflow",
        )
        await session.commit()
        if schedule_retry is None:
            msg = "schedule_retry is required when the Slack API circuit is open"
            raise RuntimeError(msg)
        await schedule_retry(
            ScheduleRetryParams(
                workflow_run_id=run_stable_id,
                delay_seconds=blocked_sl,
                state=AdvisoryWorkflowState.reporting.value,
                reason="slack_circuit_blocked",
            ),
        )
        return await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing after Slack circuit deferral",
        )

    if rate_limiter is not None and run.state == AdvisoryWorkflowState.triage_complete.value:
        try:
            slack_limit = repo.rate_limits.slack_findings_per_hour if repo.rate_limits else 30
            await rate_limiter.check_and_increment(
                operation="slack_finding",
                scope=repo.slack_channel,
                limit=slack_limit,
                window_seconds=3600,
                circuit_scope=repo.name,
            )
        except RateLimiterCircuitOpen as e:
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="rate_limiter",
                tool_inputs={"operation": "slack_finding", "scope": repo.slack_channel},
                tool_output=f"circuit open; {e.remaining_seconds}s remaining",
            )
            await session.commit()
            if schedule_retry is None:
                msg = "schedule_retry required when rate limiter circuit is open"
                raise RuntimeError(msg) from None
            await schedule_retry(
                ScheduleRetryParams(
                    workflow_run_id=run_stable_id,
                    delay_seconds=e.remaining_seconds,
                    state=run.state,
                    reason="rate_limiter_circuit_open",
                ),
            )
            return await _require_run(
                session,
                run_stable_id,
                missing_message="workflow run missing after rate limiter circuit deferral",
            )
        except RateLimitExceeded as e:
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="rate_limiter",
                tool_inputs={
                    "operation": e.operation,
                    "scope": e.scope,
                    "limit": e.limit,
                },
                tool_output=(f"rate limit exceeded; should_alert={e.should_alert}, circuit_opened={e.circuit_opened}"),
            )
            await session.commit()
            if e.should_alert:
                await _best_effort_error_slack(
                    slack,
                    repo.slack_channel,
                    title=f"Rate limit reached: {e.operation}",
                    detail=f"Limit {e.limit}/hour for {e.scope}",
                    workflow_run_id=run_stable_id,
                    finding_id=str(finding.id),
                )
            if schedule_retry is not None:
                await schedule_retry(
                    ScheduleRetryParams(
                        workflow_run_id=run_stable_id,
                        delay_seconds=_RATE_LIMIT_RETRY_SECONDS,
                        state=run.state,
                        reason="slack_rate_limited",
                    ),
                )
                return await _require_run(
                    session,
                    run_stable_id,
                    missing_message="workflow run missing after rate limit deferral",
                )
            run.state = AdvisoryWorkflowState.error_reporting.value
            run.error_message = _truncate_log(str(e), 4000)
            run.completed_at = _now_utc()
            await session.commit()
            log.warning(
                "workflow_error",
                metric_name="workflow_error_total",
                phase="reporting",
                workflow_run_id=str(run_stable_id),
                err=str(e),
            )
            return run

    if run.state == AdvisoryWorkflowState.triage_complete.value:
        run.state = AdvisoryWorkflowState.reporting.value
        await session.commit()

        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.triage_complete.value,
            to_state=AdvisoryWorkflowState.reporting.value,
            workflow_run_id=str(run_stable_id),
        )

    report = finding_to_report_payload(finding)
    try:
        if tier == GovernanceTier.approve:
            await slack.send_finding_for_approval(
                repo.slack_channel,
                report,
                workflow_run_id=run_stable_id,
                approval_context=ApprovalButtonContext(
                    finding_id=finding.id,
                    workflow_run_id=run_stable_id,
                    repo_name=repo.name,
                ),
            )
        else:
            await slack.send_finding(
                repo.slack_channel,
                report,
                workflow_run_id=run_stable_id,
                informational=tier == GovernanceTier.notify,
            )
    except SlackAPIError as e:
        await session.rollback()
        opened = breaker.record_failure("slack")
        if opened:
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="circuit_breaker",
                tool_inputs={"api": "slack", "event": "opened"},
                tool_output=f"pause_seconds={breaker.PAUSE_SEC}",
            )
        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing after Slack failure",
        )
        if e.is_transient and run.retry_count < 3 and schedule_retry is not None:
            delay = max(1, 2**run.retry_count)
            run.retry_count += 1
            await session.commit()
            await schedule_retry(
                ScheduleRetryParams(
                    workflow_run_id=run_stable_id,
                    delay_seconds=delay,
                    state=AdvisoryWorkflowState.reporting.value,
                    reason="slack_transient",
                ),
            )
            return run
        run.state = AdvisoryWorkflowState.error_reporting.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="reporting",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="slack.send_finding",
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Slack finding report failed",
            detail=str(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run
    except SlackMalformedResponseError as e:
        await session.rollback()
        breaker.record_failure("slack")
        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing after Slack failure",
        )
        run.state = AdvisoryWorkflowState.error_reporting.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="slack.send_finding",
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Slack finding report failed",
            detail=str(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run
    except Exception as e:
        await session.rollback()
        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing after Slack failure",
        )
        run.state = AdvisoryWorkflowState.error_unrecoverable.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.exception(
            "workflow_unrecoverable",
            metric_name="workflow_error_total",
            workflow_run_id=str(run_stable_id),
        )
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="slack.send_finding",
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Reporting step failed (unrecoverable)",
            detail=str(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run

    slack_delivered_at = _now_utc()
    if tier == GovernanceTier.approve:
        run.state = AdvisoryWorkflowState.awaiting_approval.value
        # completed_at stays None — interactive approval handler will finalise the run.
        terminal_state = AdvisoryWorkflowState.awaiting_approval
    else:
        run.state = AdvisoryWorkflowState.done.value
        run.completed_at = slack_delivered_at
        terminal_state = AdvisoryWorkflowState.done
    await session.commit()

    started = run.started_at.replace(tzinfo=UTC) if run.started_at.tzinfo is None else run.started_at
    elapsed = (slack_delivered_at - started).total_seconds()
    log.info(
        "advisory_to_slack_delivered",
        metric_name="advisory_to_slack_seconds",
        duration_seconds=elapsed,
        workflow_run_id=str(run_stable_id),
    )

    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.reporting.value,
        to_state=terminal_state.value,
        workflow_run_id=str(run_stable_id),
    )

    return run


async def _get_run(session: AsyncSession, run_id: uuid.UUID) -> WorkflowRun | None:
    return await session.get(WorkflowRun, run_id)


async def _require_run(session: AsyncSession, run_id: uuid.UUID, *, missing_message: str) -> WorkflowRun:
    row = await _get_run(session, run_id)
    if row is None:
        raise RuntimeError(missing_message)
    return row


__all__ = [
    "AdvisoryWorkflowState",
    "ScheduleRetryParams",
    "run_advisory_workflow",
]

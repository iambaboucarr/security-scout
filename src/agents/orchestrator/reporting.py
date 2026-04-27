# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import UTC
from typing import Any

from agents.governance import GovernanceTier
from agents.orchestrator._constants import (
    _MSG_RUN_MISSING_AFTER_SLACK_FAILURE,
    _RATE_LIMIT_RETRY_SECONDS,
    _TOOL_NAME_SLACK_SEND_FINDING,
)
from agents.orchestrator._workflow_helpers import (
    _append_action_log,
    _best_effort_error_slack,
    _now_utc,
    _safe_exc_detail,
    _truncate_log,
)
from agents.orchestrator.deps import _AdvisoryDeps
from agents.orchestrator.params import ScheduleRetryParams
from agents.orchestrator.workflow_run import _require_run
from models import AdvisoryWorkflowState, Finding, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.rate_limiter import RateLimiterCircuitOpen, RateLimitExceeded
from tools.slack import (
    ApprovalButtonContext,
    SlackAPIError,
    SlackMalformedResponseError,
    finding_to_report_payload,
)


async def _advisory_slack_circuit_defer(
    d: _AdvisoryDeps, breaker: ExternalApiCircuitBreaker, *, run_stable_id: uuid.UUID
) -> WorkflowRun | None:
    if breaker.take_resume_log_event("slack"):
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="circuit_breaker",
            tool_inputs={"api": "slack", "event": "resumed"},
            tool_output=None,
        )
        await d.session.commit()

    blocked_sl = breaker.blocked_seconds_remaining("slack")
    if blocked_sl <= 0:
        return None
    await _append_action_log(
        d.session,
        workflow_run_id=run_stable_id,
        agent="orchestrator",
        tool_name="circuit_breaker",
        tool_inputs={"api": "slack", "blocked_seconds": blocked_sl},
        tool_output="API paused; deferring workflow",
    )
    await d.session.commit()
    if d.schedule_retry is None:
        msg = "schedule_retry is required when the Slack API circuit is open"
        raise RuntimeError(msg)
    await d.schedule_retry(
        ScheduleRetryParams(
            workflow_run_id=run_stable_id,
            delay_seconds=blocked_sl,
            state=AdvisoryWorkflowState.reporting.value,
            reason="slack_circuit_blocked",
        ),
    )
    return await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing after Slack circuit deferral",
    )


async def _advisory_rate_limit_gate(
    d: _AdvisoryDeps, log: Any, *, run_stable_id: uuid.UUID, run: WorkflowRun, finding: Finding
) -> WorkflowRun | None:
    if d.rate_limiter is None or run.state not in (
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.pre_flight.value,
        AdvisoryWorkflowState.sandbox_complete.value,
    ):
        return None

    try:
        slack_limit = d.repo.rate_limits.slack_findings_per_hour if d.repo.rate_limits else 30
        await d.rate_limiter.check_and_increment(
            operation="slack_finding",
            scope=d.repo.slack_channel,
            limit=slack_limit,
            window_seconds=3600,
            circuit_scope=d.repo.name,
        )
    except RateLimiterCircuitOpen as e:
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="rate_limiter",
            tool_inputs={"operation": "slack_finding", "scope": d.repo.slack_channel},
            tool_output=f"circuit open; {e.remaining_seconds}s remaining",
        )
        await d.session.commit()
        if d.schedule_retry is None:
            msg = "schedule_retry required when rate limiter circuit is open"
            raise RuntimeError(msg) from None
        await d.schedule_retry(
            ScheduleRetryParams(
                workflow_run_id=run_stable_id,
                delay_seconds=e.remaining_seconds,
                state=run.state,
                reason="rate_limiter_circuit_open",
            ),
        )
        return await _require_run(
            d.session,
            run_stable_id,
            missing_message="workflow run missing after rate limiter circuit deferral",
        )
    except RateLimitExceeded as e:
        await _append_action_log(
            d.session,
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
        await d.session.commit()
        if e.should_alert:
            await _best_effort_error_slack(
                d.slack,
                d.repo.slack_channel,
                title=f"Rate limit reached: {e.operation}",
                detail=f"Limit {e.limit}/hour for {e.scope}",
                workflow_run_id=run_stable_id,
                finding_id=str(finding.id),
            )
        if d.schedule_retry is not None:
            await d.schedule_retry(
                ScheduleRetryParams(
                    workflow_run_id=run_stable_id,
                    delay_seconds=_RATE_LIMIT_RETRY_SECONDS,
                    state=run.state,
                    reason="slack_rate_limited",
                ),
            )
            return await _require_run(
                d.session,
                run_stable_id,
                missing_message="workflow run missing after rate limit deferral",
            )
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message="workflow run missing after rate limit handling",
        )
        run.state = AdvisoryWorkflowState.error_reporting.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await d.session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="reporting",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        return run
    return None


async def _advisory_enter_reporting_state(
    d: _AdvisoryDeps, log: Any, *, run_stable_id: uuid.UUID, run: WorkflowRun
) -> WorkflowRun:
    if run.state not in (
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.pre_flight.value,
        AdvisoryWorkflowState.sandbox_complete.value,
    ):
        return run
    prev_reporting = run.state
    run.state = AdvisoryWorkflowState.reporting.value
    await d.session.commit()

    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=prev_reporting,
        to_state=AdvisoryWorkflowState.reporting.value,
        workflow_run_id=str(run_stable_id),
    )
    return run


async def _advisory_send_slack_report(
    d: _AdvisoryDeps,
    breaker: ExternalApiCircuitBreaker,
    log: Any,
    *,
    run_stable_id: uuid.UUID,
    finding: Finding,
    tier: GovernanceTier,
) -> WorkflowRun | None:
    report = finding_to_report_payload(finding)
    try:
        if tier == GovernanceTier.approve:
            await d.slack.send_finding_for_approval(
                d.repo.slack_channel,
                report,
                workflow_run_id=run_stable_id,
                approval_context=ApprovalButtonContext(
                    finding_id=finding.id,
                    workflow_run_id=run_stable_id,
                    repo_name=d.repo.name,
                ),
            )
        else:
            await d.slack.send_finding(
                d.repo.slack_channel,
                report,
                workflow_run_id=run_stable_id,
                informational=tier == GovernanceTier.notify,
            )
    except SlackAPIError as e:
        await d.session.rollback()
        opened = breaker.record_failure("slack")
        if opened:
            await _append_action_log(
                d.session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="circuit_breaker",
                tool_inputs={"api": "slack", "event": "opened"},
                tool_output=f"pause_seconds={breaker.PAUSE_SEC}",
            )
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message=_MSG_RUN_MISSING_AFTER_SLACK_FAILURE,
        )
        if e.is_transient and run.retry_count < 3 and d.schedule_retry is not None:
            delay = max(1, 2**run.retry_count)
            run.retry_count += 1
            await d.session.commit()
            await d.schedule_retry(
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
        await d.session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="reporting",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name=_TOOL_NAME_SLACK_SEND_FINDING,
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=str(e),
        )
        await d.session.commit()
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="Slack finding report failed",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run
    except SlackMalformedResponseError as e:
        await d.session.rollback()
        breaker.record_failure("slack")
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message=_MSG_RUN_MISSING_AFTER_SLACK_FAILURE,
        )
        run.state = AdvisoryWorkflowState.error_reporting.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await d.session.commit()
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name=_TOOL_NAME_SLACK_SEND_FINDING,
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=str(e),
        )
        await d.session.commit()
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="Slack finding report failed",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run
    except Exception as e:
        await d.session.rollback()
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message=_MSG_RUN_MISSING_AFTER_SLACK_FAILURE,
        )
        run.state = AdvisoryWorkflowState.error_unrecoverable.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await d.session.commit()
        log.exception(
            "workflow_unrecoverable",
            metric_name="workflow_error_total",
            workflow_run_id=str(run_stable_id),
        )
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name=_TOOL_NAME_SLACK_SEND_FINDING,
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=str(e),
        )
        await d.session.commit()
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="Reporting step failed (unrecoverable)",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run
    return None


async def _advisory_finalize_slack_success(
    d: _AdvisoryDeps, log: Any, *, run_stable_id: uuid.UUID, run: WorkflowRun, tier: GovernanceTier
) -> WorkflowRun:
    slack_delivered_at = _now_utc()
    if tier == GovernanceTier.approve:
        run.state = AdvisoryWorkflowState.awaiting_approval.value
        terminal_state = AdvisoryWorkflowState.awaiting_approval
    else:
        run.state = AdvisoryWorkflowState.done.value
        run.completed_at = slack_delivered_at
        terminal_state = AdvisoryWorkflowState.done
    await d.session.commit()

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

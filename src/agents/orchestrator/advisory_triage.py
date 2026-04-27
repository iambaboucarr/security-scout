# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from typing import Any

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
from agents.triage import run_advisory_triage
from exceptions import SecurityScoutError
from models import AdvisoryWorkflowState, Finding, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker


async def _advisory_triage_finding_for_reporting_resume(d: _AdvisoryDeps, run_stable_id: uuid.UUID) -> Finding:
    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing before reporting resume",
    )
    if run.finding_id is None:
        msg = "resume run has no finding_id"
        raise RuntimeError(msg)
    finding = await d.session.get(Finding, run.finding_id)
    if finding is None:
        msg = "finding row missing for workflow resume"
        raise RuntimeError(msg)
    return finding


async def _advisory_triage_maybe_defer_for_github_circuit(
    d: _AdvisoryDeps, breaker: ExternalApiCircuitBreaker, run_stable_id: uuid.UUID
) -> WorkflowRun | None:
    blocked_gh = breaker.blocked_seconds_remaining("github")
    if blocked_gh <= 0:
        return None
    await _append_action_log(
        d.session,
        workflow_run_id=run_stable_id,
        agent="orchestrator",
        tool_name="circuit_breaker",
        tool_inputs={"api": "github", "blocked_seconds": blocked_gh},
        tool_output="API paused; deferring workflow",
    )
    await d.session.commit()
    if d.schedule_retry is None:
        msg = "schedule_retry is required when the GitHub API circuit is open"
        raise RuntimeError(msg)
    await d.schedule_retry(
        ScheduleRetryParams(
            workflow_run_id=run_stable_id,
            delay_seconds=blocked_gh,
            state=AdvisoryWorkflowState.triaging.value,
            reason="github_circuit_blocked",
        ),
    )
    return await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing after GitHub circuit deferral",
    )


async def _advisory_triage_phase(
    d: _AdvisoryDeps,
    breaker: ExternalApiCircuitBreaker,
    log: Any,
    *,
    run_stable_id: uuid.UUID,
    needs_triage: bool,
) -> WorkflowRun | Finding:
    if not needs_triage:
        return await _advisory_triage_finding_for_reporting_resume(d, run_stable_id)

    maybe_deferred = await _advisory_triage_maybe_defer_for_github_circuit(d, breaker, run_stable_id)
    if maybe_deferred is not None:
        return maybe_deferred

    try:
        finding = await run_advisory_triage(
            d.session,
            d.repo,
            d.scm,
            d.http,
            ghsa_id=d.ghsa_id,
            advisory_source=d.advisory_source,
            run_id=d.run_id,
            llm=d.llm,
            reasoning_model=d.reasoning_model,
            tracker_credentials=d.tracker_credentials,
        )
    except SecurityScoutError as e:
        await d.session.rollback()
        opened = breaker.record_failure("github")
        if opened:
            await _append_action_log(
                d.session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="circuit_breaker",
                tool_inputs={"api": "github", "event": "opened"},
                tool_output=f"pause_seconds={breaker.PAUSE_SEC}",
            )
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message="workflow run missing after triage failure",
        )
        if e.is_transient and run.retry_count < 3 and d.schedule_retry is not None:
            delay = max(1, 2**run.retry_count)
            run.retry_count += 1
            await d.session.commit()
            log.warning(
                "workflow_transient_retry",
                metric_name="workflow_error_total",
                phase="triage",
                workflow_run_id=str(run_stable_id),
                retry_count=run.retry_count,
                delay_seconds=delay,
            )
            await d.schedule_retry(
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
        await d.session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="triage",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="triage",
            tool_inputs={"ghsa_id": d.ghsa_id},
            tool_output=str(e),
        )
        await d.session.commit()
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="Advisory triage failed",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=None,
        )
        return run
    except Exception as e:
        await d.session.rollback()
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message="workflow run missing after triage failure",
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
            tool_name="triage",
            tool_inputs={"ghsa_id": d.ghsa_id},
            tool_output=str(e),
        )
        await d.session.commit()
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="Advisory workflow failed (unrecoverable)",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=None,
        )
        return run

    if finding is None:
        msg = "triage produced no finding"
        raise RuntimeError(msg)

    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing after triage",
    )
    run.finding_id = finding.id
    run.state = AdvisoryWorkflowState.triage_complete.value
    await d.session.commit()

    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.triaging.value,
        to_state=AdvisoryWorkflowState.triage_complete.value,
        workflow_run_id=str(run_stable_id),
    )
    return finding

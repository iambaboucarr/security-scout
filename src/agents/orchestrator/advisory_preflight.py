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
from agents.orchestrator.workflow_run import _require_run
from models import AdvisoryWorkflowState, Finding, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.poc_preflight import PreflightResult, PreflightVerdict
from tools.poc_preflight import validate as run_preflight
from tools.slack import (
    ApprovalButtonContext,
    SlackAPIError,
    SlackMalformedResponseError,
    finding_to_report_payload,
)


async def _advisory_preflight_suspicious(
    d: _AdvisoryDeps,
    breaker: ExternalApiCircuitBreaker,
    log: Any,
    *,
    run_stable_id: uuid.UUID,
    finding: Finding,
    preflight_result: PreflightResult,
) -> WorkflowRun:
    ev_pf = dict(finding.evidence or {})
    ev_pf["preflight"] = {
        "score": preflight_result.score,
        "indicators": [
            {
                "category": ind.category,
                "pattern": ind.pattern,
                "severity_weight": ind.severity_weight,
                "detail": ind.detail,
            }
            for ind in preflight_result.indicators
        ],
    }
    finding.evidence = ev_pf
    await d.session.commit()

    log.info(
        "preflight_suspicious",
        metric_name="preflight_verdict_total",
        verdict="suspicious",
        score=preflight_result.score,
        workflow_run_id=str(run_stable_id),
        finding_id=str(finding.id),
    )

    report_pf = finding_to_report_payload(finding)
    review_ctx = ApprovalButtonContext(
        finding_id=finding.id,
        workflow_run_id=run_stable_id,
        repo_name=d.repo.name,
    )
    try:
        await d.slack.send_preflight_review(
            d.repo.slack_channel,
            report_pf,
            workflow_run_id=run_stable_id,
            review_context=review_ctx,
        )
    except (SlackAPIError, SlackMalformedResponseError) as e:
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
            missing_message="workflow run missing after preflight Slack failure",
        )
        run.state = AdvisoryWorkflowState.pre_flight_suspicious.value
        await d.session.commit()
        log.warning(
            "preflight_slack_failed",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="PoC flagged as suspicious — Slack delivery failed",
            detail=f"Score {preflight_result.score:.2f}: "
            + ", ".join(i.detail for i in preflight_result.indicators[:5]),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run

    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing after preflight review Slack",
    )
    run.state = AdvisoryWorkflowState.awaiting_preflight_decision.value
    await d.session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.pre_flight.value,
        to_state=AdvisoryWorkflowState.awaiting_preflight_decision.value,
        workflow_run_id=str(run_stable_id),
    )
    return run


async def _advisory_preflight_path(
    d: _AdvisoryDeps,
    breaker: ExternalApiCircuitBreaker,
    log: Any,
    *,
    run_stable_id: uuid.UUID,
    finding: Finding,
    run: WorkflowRun,
) -> WorkflowRun | None:
    poc_content = finding.reproduction
    if not (poc_content and run.state == AdvisoryWorkflowState.triage_complete.value):
        return None

    run.state = AdvisoryWorkflowState.pre_flight.value
    await d.session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.triage_complete.value,
        to_state=AdvisoryWorkflowState.pre_flight.value,
        workflow_run_id=str(run_stable_id),
    )

    try:
        preflight_result = await run_preflight(
            poc_content,
            cwe_ids=finding.cwe_ids,
        )
    except Exception as e:
        log.warning(
            "preflight_error_fail_closed",
            metric_name="preflight_verdict_total",
            verdict="suspicious",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            d.session,
            workflow_run_id=run_stable_id,
            agent="orchestrator",
            tool_name="poc_preflight.validate",
            tool_inputs={"finding_id": str(finding.id)},
            tool_output=f"error: {_truncate_log(str(e))}",
        )
        run.state = AdvisoryWorkflowState.pre_flight_suspicious.value
        await d.session.commit()
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="PoC pre-flight validation error — requires manual review",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.pre_flight.value,
            to_state=AdvisoryWorkflowState.pre_flight_suspicious.value,
            workflow_run_id=str(run_stable_id),
        )
        return run

    await _append_action_log(
        d.session,
        workflow_run_id=run_stable_id,
        agent="orchestrator",
        tool_name="poc_preflight.validate",
        tool_inputs={
            "finding_id": str(finding.id),
            "cwe_ids": finding.cwe_ids,
        },
        tool_output=_truncate_log(
            f"verdict={preflight_result.verdict.value} "
            f"score={preflight_result.score:.3f} "
            f"indicators={[i.detail for i in preflight_result.indicators]}"
        ),
    )
    await d.session.commit()

    if preflight_result.verdict == PreflightVerdict.MALICIOUS:
        run.state = AdvisoryWorkflowState.pre_flight_blocked.value
        run.completed_at = _now_utc()
        await d.session.commit()
        log.warning(
            "preflight_blocked",
            metric_name="preflight_verdict_total",
            verdict="malicious",
            score=preflight_result.score,
            workflow_run_id=str(run_stable_id),
            finding_id=str(finding.id),
        )
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.pre_flight.value,
            to_state=AdvisoryWorkflowState.pre_flight_blocked.value,
            workflow_run_id=str(run_stable_id),
        )
        await _best_effort_error_slack(
            d.slack,
            d.repo.slack_channel,
            title="PoC blocked — malicious indicators detected",
            detail=f"Score {preflight_result.score:.2f}: "
            + ", ".join(i.detail for i in preflight_result.indicators[:5]),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return run

    if preflight_result.verdict == PreflightVerdict.SUSPICIOUS:
        return await _advisory_preflight_suspicious(
            d, breaker, log, run_stable_id=run_stable_id, finding=finding, preflight_result=preflight_result
        )

    log.info(
        "preflight_clean",
        metric_name="preflight_verdict_total",
        verdict="clean",
        score=preflight_result.score,
        workflow_run_id=str(run_stable_id),
    )

    run.state = AdvisoryWorkflowState.building_env.value
    await d.session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.pre_flight.value,
        to_state=AdvisoryWorkflowState.building_env.value,
        workflow_run_id=str(run_stable_id),
    )
    return None

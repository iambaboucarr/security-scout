# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agents.env_builder import build_environment
from agents.governance import GovernanceTier, decide_governance_tier
from agents.sandbox_executor import ExecutionResult, PocType, execute_poc
from agents.triage import run_advisory_triage
from ai.provider import LLMProvider
from config import RepoConfig
from exceptions import SecurityScoutError
from models import (
    AdvisoryWorkflowState,
    AgentActionLog,
    Finding,
    WorkflowKind,
    WorkflowRun,
)
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.docker_sandbox import SandboxError
from tools.issue_tracker import IssueTrackerCredentials
from tools.poc_preflight import PreflightVerdict
from tools.poc_preflight import validate as run_preflight
from tools.rate_limiter import RateLimiterCircuitOpen, RateLimitExceeded, SlidingWindowRateLimiter
from tools.scm import normalise_ghsa_id
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


def _safe_exc_detail(exc: BaseException) -> str:
    # Exception messages may leak paths/tokens; full text persists in WorkflowRun.error_message.
    return type(exc).__name__


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
    work_dir: Path | None = None,
    container_socket: str = "unix:///var/run/docker.sock",
) -> WorkflowRun:
    """Run or resume the advisory triage → Slack report workflow.

    When *resume_workflow_run_id* is ``None`` a fresh ``WorkflowRun`` is created.
    Pass an existing run's UUID to resume from where it left off (must be in
    ``triaging``, ``triage_complete``, ``building_env``, ``executing_sandbox``,
    ``sandbox_complete``, or ``reporting`` state).  The resumed run keeps its
    original ``id``, ``started_at``, and ``retry_count``.
    """
    log = _LOG.bind(agent="orchestrator", run_id=str(run_id) if run_id else None)
    breaker = circuit_breaker or ExternalApiCircuitBreaker()

    needs_triage: bool
    run_stable_id: uuid.UUID

    repo_slug = f"{repo.github_org}/{repo.github_repo}".lower()

    if resume_workflow_run_id is None:
        workflow_started_at = _now_utc()
        run = WorkflowRun(
            workflow_type=WorkflowKind.advisory,
            repo_name=repo_slug,
            advisory_ghsa_id=normalise_ghsa_id(ghsa_id),
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
            AdvisoryWorkflowState.building_env.value,
            AdvisoryWorkflowState.executing_sandbox.value,
            AdvisoryWorkflowState.sandbox_complete.value,
            AdvisoryWorkflowState.reporting.value,
        ):
            msg = f"cannot resume from state {loaded.state!r}"
            raise RuntimeError(msg)
        if loaded.repo_name is None:
            loaded.repo_name = repo_slug
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
                detail=_safe_exc_detail(e),
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
                detail=_safe_exc_detail(e),
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
        AdvisoryWorkflowState.pre_flight.value,
        AdvisoryWorkflowState.building_env.value,
        AdvisoryWorkflowState.executing_sandbox.value,
        AdvisoryWorkflowState.sandbox_complete.value,
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

    # --- Pre-flight validation (mandatory gate before sandbox execution) ---
    poc_content = finding.reproduction
    if poc_content and run.state == AdvisoryWorkflowState.triage_complete.value:
        run.state = AdvisoryWorkflowState.pre_flight.value
        await session.commit()
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
            # Pre-flight errors → treat PoC as suspicious (fail-closed)
            log.warning(
                "preflight_error_fail_closed",
                metric_name="preflight_verdict_total",
                verdict="suspicious",
                workflow_run_id=str(run_stable_id),
                err=str(e),
            )
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="poc_preflight.validate",
                tool_inputs={"finding_id": str(finding.id)},
                tool_output=f"error: {_truncate_log(str(e))}",
            )
            run.state = AdvisoryWorkflowState.pre_flight_suspicious.value
            await session.commit()
            await _best_effort_error_slack(
                slack,
                repo.slack_channel,
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
            session,
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
        await session.commit()

        if preflight_result.verdict == PreflightVerdict.MALICIOUS:
            run.state = AdvisoryWorkflowState.pre_flight_blocked.value
            run.completed_at = _now_utc()
            await session.commit()
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
                slack,
                repo.slack_channel,
                title="PoC blocked — malicious indicators detected",
                detail=f"Score {preflight_result.score:.2f}: "
                + ", ".join(i.detail for i in preflight_result.indicators[:5]),
                workflow_run_id=run_stable_id,
                finding_id=str(finding.id),
            )
            return run

        if preflight_result.verdict == PreflightVerdict.SUSPICIOUS:
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
            await session.commit()

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
                repo_name=repo.name,
            )
            try:
                await slack.send_preflight_review(
                    repo.slack_channel,
                    report_pf,
                    workflow_run_id=run_stable_id,
                    review_context=review_ctx,
                )
            except (SlackAPIError, SlackMalformedResponseError) as e:
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
                    missing_message="workflow run missing after preflight Slack failure",
                )
                run.state = AdvisoryWorkflowState.pre_flight_suspicious.value
                await session.commit()
                log.warning(
                    "preflight_slack_failed",
                    workflow_run_id=str(run_stable_id),
                    err=str(e),
                )
                await _best_effort_error_slack(
                    slack,
                    repo.slack_channel,
                    title="PoC flagged as suspicious — Slack delivery failed",
                    detail=f"Score {preflight_result.score:.2f}: "
                    + ", ".join(i.detail for i in preflight_result.indicators[:5]),
                    workflow_run_id=run_stable_id,
                    finding_id=str(finding.id),
                )
                return run

            run = await _require_run(
                session,
                run_stable_id,
                missing_message="workflow run missing after preflight review Slack",
            )
            run.state = AdvisoryWorkflowState.awaiting_preflight_decision.value
            await session.commit()
            log.info(
                "workflow_state_transition",
                metric_name="workflow_state_current",
                from_state=AdvisoryWorkflowState.pre_flight.value,
                to_state=AdvisoryWorkflowState.awaiting_preflight_decision.value,
                workflow_run_id=str(run_stable_id),
            )
            return run

        # CLEAN — proceed to sandbox execution
        log.info(
            "preflight_clean",
            metric_name="preflight_verdict_total",
            verdict="clean",
            score=preflight_result.score,
            workflow_run_id=str(run_stable_id),
        )

        run.state = AdvisoryWorkflowState.building_env.value
        await session.commit()
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.pre_flight.value,
            to_state=AdvisoryWorkflowState.building_env.value,
            workflow_run_id=str(run_stable_id),
        )

    # --- Sandbox execution (building_env → executing_sandbox → sandbox_complete) ---
    if run.state == AdvisoryWorkflowState.executing_sandbox.value:
        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing before sandbox resume",
        )
        run.state = AdvisoryWorkflowState.building_env.value
        await session.commit()
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.executing_sandbox.value,
            to_state=AdvisoryWorkflowState.building_env.value,
            workflow_run_id=str(run_stable_id),
        )

    if run.state == AdvisoryWorkflowState.building_env.value:
        await _run_sandbox_phase(
            session=session,
            run_stable_id=run_stable_id,
            finding=finding,
            repo=repo,
            scm=scm,
            slack=slack,
            log=log,
            work_dir=work_dir,
            container_socket=container_socket,
            schedule_retry=schedule_retry,
        )
        run = await _require_run(
            session,
            run_stable_id,
            missing_message="workflow run missing after sandbox phase",
        )
        if run.state in (
            AdvisoryWorkflowState.error_sandbox.value,
            AdvisoryWorkflowState.error_unrecoverable.value,
        ):
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

    if rate_limiter is not None and run.state in (
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.pre_flight.value,
        AdvisoryWorkflowState.sandbox_complete.value,
    ):
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

    if run.state in (
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.pre_flight.value,
        AdvisoryWorkflowState.sandbox_complete.value,
    ):
        prev_reporting = run.state
        run.state = AdvisoryWorkflowState.reporting.value
        await session.commit()

        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=prev_reporting,
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
            detail=_safe_exc_detail(e),
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
            detail=_safe_exc_detail(e),
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
            detail=_safe_exc_detail(e),
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


async def _run_sandbox_phase(
    *,
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    scm: SCMProvider,
    slack: SlackClient,
    log: Any,
    work_dir: Path | None,
    container_socket: str,
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None,
) -> ExecutionResult | None:
    """Run env build → sandbox execute → sandbox_complete.

    Returns ``ExecutionResult`` on success, ``None`` on error (run state already
    updated to ``error_sandbox`` or ``error_unrecoverable``).
    """
    import tempfile

    owns_work_dir = work_dir is None
    effective_work_dir = work_dir or Path(tempfile.mkdtemp(prefix="scout-"))
    try:
        return await _run_sandbox_phase_inner(
            session=session,
            run_stable_id=run_stable_id,
            finding=finding,
            repo=repo,
            scm=scm,
            slack=slack,
            log=log,
            effective_work_dir=effective_work_dir,
            container_socket=container_socket,
            schedule_retry=schedule_retry,
        )
    finally:
        if owns_work_dir:
            shutil.rmtree(effective_work_dir, ignore_errors=True)


async def _run_sandbox_phase_inner(
    *,
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    scm: SCMProvider,
    slack: SlackClient,
    log: Any,
    effective_work_dir: Path,
    container_socket: str,
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None,
) -> ExecutionResult | None:
    repo_slug = f"{repo.github_org}/{repo.github_repo}".lower()

    run = await _require_run(session, run_stable_id, missing_message="run missing in sandbox phase")

    oracle_ev = (finding.evidence or {}).get("oracle")
    # Default from repos.yaml when triage did not set evidence.oracle.vulnerable_ref.
    clone_ref = repo.default_git_ref
    if isinstance(oracle_ev, dict):
        vr = oracle_ev.get("vulnerable_ref")
        if isinstance(vr, str) and vr.strip():
            clone_ref = vr.strip()

    # --- Build environment ---
    try:
        env_result = await build_environment(
            scm,
            repo_slug=repo_slug,
            ref=clone_ref,
            work_dir=effective_work_dir,
            container_socket=container_socket,
        )
    except SecurityScoutError as e:
        if e.is_transient:
            run = await _require_run(session, run_stable_id, missing_message="run missing after env build failure")
            if run.retry_count < 3 and schedule_retry is not None:
                delay = max(1, 2**run.retry_count)
                run.retry_count += 1
                await session.commit()
                log.warning(
                    "workflow_transient_retry",
                    metric_name="workflow_error_total",
                    phase="building_env",
                    workflow_run_id=str(run_stable_id),
                    retry_count=run.retry_count,
                    delay_seconds=delay,
                )
                await schedule_retry(
                    ScheduleRetryParams(
                        workflow_run_id=run_stable_id,
                        delay_seconds=delay,
                        state=AdvisoryWorkflowState.building_env.value,
                        reason="env_build_transient",
                    ),
                )
                return None
            run.state = AdvisoryWorkflowState.error_sandbox.value
            run.error_message = _truncate_log(str(e), 4000)
            run.completed_at = _now_utc()
            await session.commit()
            log.warning(
                "workflow_error",
                metric_name="workflow_error_total",
                phase="building_env",
                workflow_run_id=str(run_stable_id),
                err=str(e),
            )
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="env_builder",
                tool_name="build_environment",
                tool_inputs={"repo": repo_slug},
                tool_output=str(e),
            )
            await session.commit()
            await _best_effort_error_slack(
                slack,
                repo.slack_channel,
                title="Environment build failed",
                detail=_safe_exc_detail(e),
                workflow_run_id=run_stable_id,
                finding_id=str(finding.id),
            )
            return None
        run = await _require_run(session, run_stable_id, missing_message="run missing after env build failure")
        run.state = AdvisoryWorkflowState.error_sandbox.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="building_env",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="env_builder",
            tool_name="build_environment",
            tool_inputs={"repo": repo_slug},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Environment build failed (permanent)",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return None
    except Exception as e:
        run = await _require_run(session, run_stable_id, missing_message="run missing after env build failure")
        run.state = AdvisoryWorkflowState.error_unrecoverable.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.exception(
            "workflow_unrecoverable",
            metric_name="workflow_error_total",
            phase="building_env",
            workflow_run_id=str(run_stable_id),
        )
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Environment build failed (unrecoverable)",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return None

    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="env_builder",
        tool_name="build_environment",
        tool_inputs={"repo": repo_slug, "stack": env_result.detected_stack.value},
        tool_output=_truncate_log(f"image={env_result.image_tag}"),
    )
    await session.commit()

    run = await _require_run(session, run_stable_id, missing_message="run missing after env build")
    run.state = AdvisoryWorkflowState.executing_sandbox.value
    await session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.building_env.value,
        to_state=AdvisoryWorkflowState.executing_sandbox.value,
        workflow_run_id=str(run_stable_id),
    )

    # --- Execute PoC in sandbox ---
    poc_content = finding.reproduction or ""
    poc_command = ["python", "-c", poc_content] if poc_content else ["echo", "no PoC"]

    try:
        exec_result = await execute_poc(
            image=env_result.image_tag,
            poc_command=poc_command,
            poc_type=PocType.RESEARCHER_SUBMITTED,
            repo_path=env_result.repo_path,
            container_socket=container_socket,
        )
    except (NotImplementedError, SandboxError) as e:
        run = await _require_run(session, run_stable_id, missing_message="run missing after sandbox exec failure")
        run.state = AdvisoryWorkflowState.error_sandbox.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="executing_sandbox",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="sandbox_executor",
            tool_name="execute_poc",
            tool_inputs={"image": env_result.image_tag},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Sandbox execution failed",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return None
    except Exception as e:
        run = await _require_run(session, run_stable_id, missing_message="run missing after sandbox exec failure")
        run.state = AdvisoryWorkflowState.error_sandbox.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.exception(
            "workflow_unexpected_sandbox_error",
            metric_name="workflow_error_total",
            phase="executing_sandbox",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="sandbox_executor",
            tool_name="execute_poc",
            tool_inputs={"image": env_result.image_tag},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Sandbox execution failed",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return None

    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="sandbox_executor",
        tool_name="execute_poc",
        tool_inputs={
            "image": env_result.image_tag,
            "poc_type": exec_result.poc_type.value,
        },
        tool_output=_truncate_log(
            f"tier={exec_result.confidence_tier.value} "
            f"exit={exec_result.exit_code} "
            f"elapsed={exec_result.elapsed_seconds:.1f}s"
        ),
    )
    await session.commit()

    log.info(
        "sandbox_execution_complete",
        metric_name="sandbox_execution_seconds",
        duration_seconds=exec_result.elapsed_seconds,
        confidence_tier=exec_result.confidence_tier.value,
        workflow_run_id=str(run_stable_id),
    )

    finding.status = exec_result.confidence_tier
    finding.poc_executed = True
    merged_evidence = dict(finding.evidence or {})
    merged_evidence["execution"] = {
        "excerpt": exec_result.evidence_excerpt,
        "exit_code": exec_result.exit_code,
        "elapsed_seconds": exec_result.elapsed_seconds,
        "poc_type": exec_result.poc_type.value,
        "timed_out": exec_result.timed_out,
    }
    finding.evidence = merged_evidence
    await session.commit()

    run = await _require_run(session, run_stable_id, missing_message="run missing after sandbox exec")
    run.state = AdvisoryWorkflowState.sandbox_complete.value
    await session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.executing_sandbox.value,
        to_state=AdvisoryWorkflowState.sandbox_complete.value,
        workflow_run_id=str(run_stable_id),
    )

    return exec_result


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

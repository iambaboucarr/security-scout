# SPDX-License-Identifier: Apache-2.0
"""Orchestrator integration tests for the rate limiter code paths.

Exercises:
- RateLimiterCircuitOpen deferral (with and without schedule_retry)
- RateLimitExceeded with retry scheduling
- RateLimitExceeded without schedule_retry (error terminal)
- Rate limiter only checked when transitioning from triage_complete
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

from agents.orchestrator import (
    _RATE_LIMIT_RETRY_SECONDS,
    AdvisoryWorkflowState,
    ScheduleRetryParams,
    run_advisory_workflow,
)
from config import GovernanceConfig, GovernanceRule, RepoConfig
from models import AgentActionLog, Finding, FindingStatus, Severity, SSVCAction, WorkflowKind, WorkflowRun
from tools.github import GitHubClient
from tools.rate_limiter import RateLimiterCircuitOpen, RateLimitExceeded, SlidingWindowRateLimiter
from tools.scm.github import GitHubSCMProvider


def _make_scm(gh: object) -> GitHubSCMProvider:
    return GitHubSCMProvider.from_client(gh)  # type: ignore[arg-type]


def _repo() -> RepoConfig:
    return RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        issue_trackers=[],
        governance=GovernanceConfig(notify=[GovernanceRule(severity=[Severity.high])]),
    )


def _slack_transport_ok() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if "chat.postMessage" in str(request.url):
            return httpx.Response(200, json={"ok": True, "channel": "C123", "ts": "1234.5678"})
        return httpx.Response(404, json={"ok": False})

    return httpx.MockTransport(handler)


async def _make_finding(session: object, *_a: object, **_k: object) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-TEST",
        severity=Severity.high,
        ssvc_action=SSVCAction.act,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.9,
        title="Test advisory",
    )
    session.add(f)  # type: ignore[arg-type]
    await session.flush()  # type: ignore[attr-defined]
    return f


# ── RateLimiterCircuitOpen deferral ───────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limiter_circuit_open_defers_with_schedule_retry(db_session, mocker) -> None:
    """When the rate limiter circuit is open, the workflow defers via schedule_retry."""
    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

    rl = AsyncMock(spec=SlidingWindowRateLimiter)
    rl.check_and_increment = AsyncMock(
        side_effect=RateLimiterCircuitOpen("circuit open for demo", scope="demo", remaining_seconds=600),
    )
    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = MagicMock()
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            rate_limiter=rl,
            schedule_retry=schedule,
        )

    schedule.assert_awaited_once()
    params: ScheduleRetryParams = schedule.await_args[0][0]
    assert params.reason == "rate_limiter_circuit_open"
    assert params.delay_seconds == 600
    assert params.state == AdvisoryWorkflowState.triage_complete.value

    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "rate_limiter" and "circuit open" in (log.tool_output or "") for log in logs)


@pytest.mark.asyncio
async def test_rate_limiter_circuit_open_no_schedule_retry_raises(db_session, mocker) -> None:
    """RateLimiterCircuitOpen without schedule_retry raises RuntimeError."""
    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

    rl = AsyncMock(spec=SlidingWindowRateLimiter)
    rl.check_and_increment = AsyncMock(
        side_effect=RateLimiterCircuitOpen("circuit open", scope="demo", remaining_seconds=300),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = MagicMock()
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="schedule_retry required"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                ghsa_id="GHSA-TEST-ABCD-EFGH",
                rate_limiter=rl,
                schedule_retry=None,
            )


# ── RateLimitExceeded with retry ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_exceeded_schedules_retry(db_session, mocker) -> None:
    """When rate limit is exceeded with schedule_retry available, workflow defers."""
    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

    rl = AsyncMock(spec=SlidingWindowRateLimiter)
    rl.check_and_increment = AsyncMock(
        side_effect=RateLimitExceeded(
            "limit hit",
            operation="slack_finding",
            scope="#security",
            limit=30,
            window_seconds=3600,
            should_alert=True,
            circuit_opened=False,
        ),
    )
    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = MagicMock()
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            rate_limiter=rl,
            schedule_retry=schedule,
        )

    schedule.assert_awaited_once()
    params: ScheduleRetryParams = schedule.await_args[0][0]
    assert params.reason == "slack_rate_limited"
    assert params.delay_seconds == _RATE_LIMIT_RETRY_SECONDS

    slack.notify_workflow_error.assert_awaited_once()

    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "rate_limiter" and "rate limit exceeded" in (log.tool_output or "") for log in logs)


# ── RateLimitExceeded without schedule_retry → error terminal ─────────────


@pytest.mark.asyncio
async def test_rate_limit_exceeded_no_retry_goes_to_error_reporting(db_session, mocker) -> None:
    """Without schedule_retry, RateLimitExceeded marks the run as error_reporting."""
    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

    rl = AsyncMock(spec=SlidingWindowRateLimiter)
    rl.check_and_increment = AsyncMock(
        side_effect=RateLimitExceeded(
            "limit hit",
            operation="slack_finding",
            scope="#security",
            limit=30,
            window_seconds=3600,
            should_alert=False,
            circuit_opened=False,
        ),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = MagicMock()
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            rate_limiter=rl,
            schedule_retry=None,
        )

    assert run.state == AdvisoryWorkflowState.error_reporting.value
    assert run.completed_at is not None
    assert "limit hit" in (run.error_message or "")

    slack.notify_workflow_error.assert_not_awaited()


# ── Rate limiter skipped on resume at reporting state ─────────────────────


@pytest.mark.asyncio
async def test_rate_limiter_skipped_when_resuming_at_reporting(db_session, mocker) -> None:
    """Resuming a workflow at reporting state should NOT re-check the rate limiter."""
    f = await _make_finding(db_session)
    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.reporting.value,
        retry_count=1,
        finding_id=f.id,
    )
    db_session.add(wr)
    await db_session.commit()

    rl = AsyncMock(spec=SlidingWindowRateLimiter)
    rl.check_and_increment = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = MagicMock()
        slack.send_finding = AsyncMock()
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        mocker.patch("agents.orchestrator.finding_to_report_payload", return_value=MagicMock())
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            resume_workflow_run_id=wr.id,
            rate_limiter=rl,
        )

    rl.check_and_increment.assert_not_called()
    assert run.state == AdvisoryWorkflowState.done.value


# ── Rate limiter allowed → workflow proceeds to Slack ─────────────────────


@pytest.mark.asyncio
async def test_rate_limiter_allows_and_workflow_completes(db_session, mocker) -> None:
    """When rate limiter allows, the workflow proceeds normally to done."""
    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

    rl = AsyncMock(spec=SlidingWindowRateLimiter)
    rl.check_and_increment = AsyncMock(return_value=None)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = MagicMock()
        slack.send_finding = AsyncMock()
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        mocker.patch("agents.orchestrator.finding_to_report_payload", return_value=MagicMock())
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            rate_limiter=rl,
        )

    rl.check_and_increment.assert_awaited_once()
    assert run.state == AdvisoryWorkflowState.done.value

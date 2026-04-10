from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

from agents.orchestrator import AdvisoryWorkflowState, ScheduleRetryParams, run_advisory_workflow
from config import RepoConfig
from models import AgentActionLog, Finding, FindingStatus, Severity, SSVCAction, WorkflowKind
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.github import GitHubAPIError, GitHubClient
from tools.slack import SlackClient


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
    )


def _slack_transport_ok() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if "chat.postMessage" in str(request.url):
            return httpx.Response(
                200,
                json={"ok": True, "channel": "C123", "ts": "1234.5678"},
            )
        return httpx.Response(404, json={"ok": False})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_workflow_happy_path_completes(db_session, mocker) -> None:
    repo = _repo()

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = Finding(
            workflow=WorkflowKind.advisory,
            source_ref="https://github.com/advisories/GHSA-TEST",
            severity=Severity.high,
            ssvc_action=SSVCAction.act,
            status=FindingStatus.unconfirmed,
            triage_confidence=0.9,
            title="Test advisory",
        )
        session.add(f)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        run = await run_advisory_workflow(
            db_session,
            repo,
            gh,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None
    assert run.finding_id is not None


@pytest.mark.asyncio
async def test_github_transient_schedules_retry(db_session, mocker) -> None:
    repo = _repo()
    mocker.patch(
        "agents.orchestrator.run_advisory_triage",
        side_effect=GitHubAPIError("upstream", is_transient=True, http_status=503),
    )
    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        run = await run_advisory_workflow(
            db_session,
            repo,
            gh,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            schedule_retry=schedule,
        )

    assert run.retry_count == 1
    assert run.state == AdvisoryWorkflowState.triaging.value
    schedule.assert_awaited_once()
    args = schedule.await_args[0][0]
    assert isinstance(args, ScheduleRetryParams)
    assert args.delay_seconds == 1
    assert args.reason == "github_transient"


@pytest.mark.asyncio
async def test_github_permanent_error_triage_terminal(db_session, mocker) -> None:
    repo = _repo()
    mocker.patch(
        "agents.orchestrator.run_advisory_triage",
        side_effect=GitHubAPIError("bad request", is_transient=False, http_status=400),
    )
    notify = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = notify
        gh = MagicMock(spec=GitHubClient)
        run = await run_advisory_workflow(
            db_session,
            repo,
            gh,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.error_triage.value
    assert run.completed_at is not None
    notify.assert_awaited()
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "triage" for log in logs)


@pytest.mark.asyncio
async def test_circuit_github_blocks_before_triage(db_session, mocker) -> None:
    repo = _repo()
    triage = mocker.patch("agents.orchestrator.run_advisory_triage")
    t = [0.0]

    def clock() -> float:
        return t[0]

    breaker = ExternalApiCircuitBreaker(now_fn=clock)
    for _ in range(4):
        assert breaker.record_failure("github") is False
    assert breaker.record_failure("github") is True

    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        await run_advisory_workflow(
            db_session,
            repo,
            gh,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            circuit_breaker=breaker,
            schedule_retry=schedule,
        )

    triage.assert_not_called()
    schedule.assert_awaited_once()
    assert schedule.await_args[0][0].reason == "github_circuit_blocked"


@pytest.mark.asyncio
async def test_slack_transient_schedules_retry(db_session, mocker) -> None:
    repo = _repo()

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = Finding(
            workflow=WorkflowKind.advisory,
            source_ref="https://github.com/advisories/GHSA-TEST",
            severity=Severity.high,
            title="T",
        )
        session.add(f)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)

    post_calls = [0]

    def fail_then_ok(request: httpx.Request) -> httpx.Response:
        if "chat.postMessage" in str(request.url):
            post_calls[0] += 1
            if post_calls[0] == 1:
                return httpx.Response(503, json={"ok": False})
            return httpx.Response(200, json={"ok": True, "channel": "C", "ts": "1.0"})
        return httpx.Response(404)

    async with httpx.AsyncClient(
        base_url="https://slack.com/api",
        transport=httpx.MockTransport(fail_then_ok),
    ) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        schedule = AsyncMock()
        run = await run_advisory_workflow(
            db_session,
            repo,
            gh,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            schedule_retry=schedule,
        )

    assert run.state == AdvisoryWorkflowState.reporting.value
    assert run.retry_count == 1
    schedule.assert_awaited_once()
    assert schedule.await_args[0][0].reason == "slack_transient"

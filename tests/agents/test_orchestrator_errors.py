# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

from agents.orchestrator import (
    AdvisoryWorkflowParams,
    AdvisoryWorkflowState,
    ScheduleRetryParams,
    run_advisory_workflow,
)
from agents.orchestrator._workflow_helpers import (
    _best_effort_error_slack,
    _safe_exc_detail,
    _truncate_log,
)
from config import GovernanceConfig, GovernanceRule, RepoConfig
from exceptions import SecurityScoutError
from models import AgentActionLog, Finding, FindingStatus, Severity, SSVCAction, WorkflowKind, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.github import GitHubAPIError, GitHubClient
from tools.scm.github import GitHubSCMProvider
from tools.slack import SlackAPIError, SlackClient, SlackMalformedResponseError


def _make_scm(gh: object) -> GitHubSCMProvider:
    return GitHubSCMProvider.from_client(gh)  # type: ignore[arg-type]


def _repo() -> RepoConfig:
    # Routes severity=high to notify tier so happy reporting paths terminate in ``done``.
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


async def _make_finding(session: object) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        repo_name="acme/app",
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


# ── _truncate_log ──────────────────────────────────────────────────────────


def test_truncate_log_none() -> None:
    assert _truncate_log(None) is None


def test_truncate_log_short() -> None:
    assert _truncate_log("abc") == "abc"


def test_truncate_log_exact_boundary() -> None:
    text = "x" * 500
    assert _truncate_log(text) == text


def test_truncate_log_long() -> None:
    text = "x" * 600
    result = _truncate_log(text, max_chars=500)
    assert len(result) == 500
    assert result.endswith("…")


# ── Resume validation errors ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_completed_workflow_raises(db_session, mocker) -> None:
    from datetime import UTC, datetime

    f = await _make_finding(db_session)
    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.done.value,
        retry_count=0,
        finding_id=f.id,
        completed_at=datetime.now(UTC),
    )
    db_session.add(wr)
    await db_session.commit()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="cannot resume a completed"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", resume_workflow_run_id=wr.id),
            )


@pytest.mark.asyncio
async def test_resume_wrong_workflow_type_raises(db_session) -> None:
    wr = WorkflowRun(
        workflow_type=WorkflowKind.code_audit,
        state="triaging",
        retry_count=0,
    )
    db_session.add(wr)
    await db_session.commit()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="resume only supported for advisory"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", resume_workflow_run_id=wr.id),
            )


@pytest.mark.asyncio
async def test_resume_invalid_state_raises(db_session) -> None:
    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.error_triage.value,
        retry_count=0,
    )
    db_session.add(wr)
    await db_session.commit()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="cannot resume from state"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", resume_workflow_run_id=wr.id),
            )


@pytest.mark.asyncio
async def test_resume_triage_complete_with_no_finding_raises(db_session) -> None:
    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.triage_complete.value,
        retry_count=0,
        finding_id=None,
    )
    db_session.add(wr)
    await db_session.commit()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="resume run has no finding_id"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", resume_workflow_run_id=wr.id),
            )


# ── SecurityScoutError during triage ──────────────────────────────────────


@pytest.mark.asyncio
async def test_security_scout_error_transient_schedules_retry(db_session, mocker) -> None:
    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=SecurityScoutError("transient issue", is_transient=True),
    )
    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", schedule_retry=schedule),
        )

    assert run.retry_count == 1
    assert run.state == AdvisoryWorkflowState.triaging.value
    schedule.assert_awaited_once()
    args = schedule.await_args[0][0]
    assert isinstance(args, ScheduleRetryParams)
    assert args.reason == "triage_transient"


@pytest.mark.asyncio
async def test_security_scout_error_permanent_terminal(db_session, mocker) -> None:
    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=SecurityScoutError("bad data", is_transient=False),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session, _repo(), scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.error_triage.value
    assert run.completed_at is not None
    assert "bad data" in (run.error_message or "")
    slack.notify_workflow_error.assert_awaited()


# ── Unrecoverable Exception during triage ─────────────────────────────────


@pytest.mark.asyncio
async def test_unrecoverable_exception_during_triage(db_session, mocker) -> None:
    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=RuntimeError("unexpected kaboom"),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session, _repo(), scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.error_unrecoverable.value
    assert run.completed_at is not None
    assert "unexpected kaboom" in (run.error_message or "")

    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "triage" for log in logs)


# ── Slack reporting errors ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_permanent_error_terminal(db_session, mocker) -> None:
    async def fake_triage(session, *a, **kw):
        f = await _make_finding(session)
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    def fail_handler(request: httpx.Request) -> httpx.Response:
        if "chat.postMessage" in str(request.url):
            return httpx.Response(400, json={"ok": False, "error": "invalid_auth"})
        return httpx.Response(404)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(fail_handler)) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session, _repo(), scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.error_reporting.value
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_slack_malformed_response_terminal(db_session, mocker) -> None:
    async def fake_triage(session, *a, **kw):
        return await _make_finding(session)

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.reporting.finding_to_report_payload",
        return_value=MagicMock(),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.send_finding = AsyncMock(
            side_effect=SlackMalformedResponseError("bad shape"),
        )
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session, _repo(), scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.error_reporting.value
    assert "bad shape" in (run.error_message or "")


@pytest.mark.asyncio
async def test_unrecoverable_exception_during_reporting(db_session, mocker) -> None:
    async def fake_triage(session, *a, **kw):
        return await _make_finding(session)

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.reporting.finding_to_report_payload",
        return_value=MagicMock(),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.send_finding = AsyncMock(side_effect=RuntimeError("surprise"))
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session, _repo(), scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.error_unrecoverable.value
    assert "surprise" in (run.error_message or "")
    slack.notify_workflow_error.assert_awaited()


# ── Slack circuit breaker blocks reporting ────────────────────────────────


@pytest.mark.asyncio
async def test_slack_circuit_blocks_before_reporting(db_session, mocker) -> None:
    f = await _make_finding(db_session)
    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.triage_complete.value,
        retry_count=0,
        finding_id=f.id,
    )
    db_session.add(wr)
    await db_session.commit()

    t = [0.0]

    def clock() -> float:
        return t[0]

    breaker = ExternalApiCircuitBreaker(now_fn=clock)
    for _ in range(4):
        assert breaker.record_failure("slack") is False
    assert breaker.record_failure("slack") is True

    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(
                ghsa_id="GHSA-TEST-ABCD-EFGH",
                resume_workflow_run_id=wr.id,
                circuit_breaker=breaker,
                schedule_retry=schedule,
            ),
        )

    schedule.assert_awaited_once()
    assert schedule.await_args[0][0].reason == "slack_circuit_blocked"


@pytest.mark.asyncio
async def test_slack_circuit_blocks_no_schedule_retry_raises(db_session) -> None:
    f = await _make_finding(db_session)
    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.triage_complete.value,
        retry_count=0,
        finding_id=f.id,
    )
    db_session.add(wr)
    await db_session.commit()

    t = [0.0]
    breaker = ExternalApiCircuitBreaker(now_fn=lambda: t[0])
    for _ in range(4):
        breaker.record_failure("slack")
    breaker.record_failure("slack")

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="schedule_retry is required"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                AdvisoryWorkflowParams(
                    ghsa_id="GHSA-TEST-ABCD-EFGH",
                    resume_workflow_run_id=wr.id,
                    circuit_breaker=breaker,
                    schedule_retry=None,
                ),
            )


# ── GitHub circuit blocks + no schedule_retry → error ─────────────────────


@pytest.mark.asyncio
async def test_github_circuit_blocks_no_schedule_retry_raises(db_session, mocker) -> None:
    t = [0.0]
    breaker = ExternalApiCircuitBreaker(now_fn=lambda: t[0])
    for _ in range(4):
        breaker.record_failure("github")
    breaker.record_failure("github")

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        with pytest.raises(RuntimeError, match="schedule_retry is required"):
            await run_advisory_workflow(
                db_session,
                _repo(),
                scm,
                http,
                slack,
                AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", circuit_breaker=breaker, schedule_retry=None),
            )


# ── GitHub triage transient with circuit breaker opening ──────────────────


@pytest.mark.asyncio
async def test_github_transient_opens_circuit_breaker(db_session, mocker) -> None:
    t = [0.0]
    breaker = ExternalApiCircuitBreaker(now_fn=lambda: t[0])
    for _ in range(4):
        breaker.record_failure("github")

    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=GitHubAPIError("upstream fail", is_transient=True, http_status=503),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", circuit_breaker=breaker),
        )

    assert run.state == AdvisoryWorkflowState.error_triage.value
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "circuit_breaker" and log.tool_inputs.get("event") == "opened" for log in logs)


# ── Slack transient opens circuit breaker during reporting ────────────────


@pytest.mark.asyncio
async def test_slack_transient_opens_circuit_breaker(db_session, mocker) -> None:
    async def fake_triage(session, *a, **kw):
        return await _make_finding(session)

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    t = [0.0]
    breaker = ExternalApiCircuitBreaker(now_fn=lambda: t[0])
    for _ in range(4):
        breaker.record_failure("slack")

    def fail_handler(request: httpx.Request) -> httpx.Response:
        if "chat.postMessage" in str(request.url):
            return httpx.Response(503, json={"ok": False})
        return httpx.Response(404)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(fail_handler)) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", circuit_breaker=breaker),
        )

    assert run.state == AdvisoryWorkflowState.error_reporting.value
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "circuit_breaker" and log.tool_inputs.get("event") == "opened" for log in logs)


# ── _best_effort_error_slack swallows SlackAPIError ───────────────────────


def test_safe_exc_detail_returns_class_name_only() -> None:
    secret = "token=sk-abc123 path=/home/user/secrets/config.yaml"
    detail = _safe_exc_detail(RuntimeError(secret))
    assert detail == "RuntimeError"
    assert "token=" not in detail
    assert "/home/user" not in detail


def test_safe_exc_detail_preserves_custom_exception_type() -> None:
    class _CustomBoom(Exception):
        pass

    detail = _safe_exc_detail(_CustomBoom("oops with /etc/passwd and api_key=x"))
    assert detail == "_CustomBoom"
    assert "passwd" not in detail
    assert "api_key" not in detail


@pytest.mark.asyncio
async def test_slack_notify_on_triage_failure_excludes_exception_message(db_session, mocker) -> None:
    secret_msg = "db-password=hunter2 /home/ci/app/.env"
    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=SecurityScoutError(secret_msg, is_transient=False),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        slack.notify_workflow_error = AsyncMock()
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        await run_advisory_workflow(
            db_session, _repo(), scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    slack.notify_workflow_error.assert_awaited_once()
    call_kwargs = slack.notify_workflow_error.await_args.kwargs
    assert call_kwargs["detail"] == "SecurityScoutError"
    assert "hunter2" not in call_kwargs["detail"]
    assert "/home/ci" not in call_kwargs["detail"]


@pytest.mark.asyncio
async def test_best_effort_error_slack_swallows_exception() -> None:
    import uuid

    slack = MagicMock(spec=SlackClient)
    slack.notify_workflow_error = AsyncMock(
        side_effect=SlackAPIError("boom", is_transient=True),
    )
    await _best_effort_error_slack(
        slack,
        "#channel",
        title="test",
        detail="detail",
        workflow_run_id=uuid.uuid4(),
        finding_id="f-1",
    )
    slack.notify_workflow_error.assert_awaited_once()


# ── Circuit breaker resume log during workflow ────────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_resume_logged_on_workflow_start(db_session, mocker) -> None:
    async def fake_triage(session, *a, **kw):
        return await _make_finding(session)

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    t = [0.0]
    breaker = ExternalApiCircuitBreaker(now_fn=lambda: t[0])
    for _ in range(4):
        breaker.record_failure("github")
    breaker.record_failure("github")
    t[0] = breaker.PAUSE_SEC + 1

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            _repo(),
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", circuit_breaker=breaker),
        )

    assert run.state == AdvisoryWorkflowState.done.value
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "circuit_breaker" and log.tool_inputs.get("event") == "resumed" for log in logs)

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select

from agents.orchestrator import AdvisoryWorkflowState, ScheduleRetryParams, run_advisory_workflow
from config import GovernanceConfig, GovernanceRule, RepoConfig
from models import AgentActionLog, Finding, FindingStatus, KnownStatus, Severity, SSVCAction, WorkflowKind
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.github import GitHubAPIError, GitHubClient
from tools.scm.github import GitHubSCMProvider
from tools.slack import SlackClient


def _make_scm(gh: object) -> GitHubSCMProvider:
    return GitHubSCMProvider.from_client(gh)  # type: ignore[arg-type]


def _repo(governance: GovernanceConfig | None = None) -> RepoConfig:
    # Default governance sends severity=high to the notify tier so existing happy-path
    # tests continue to terminate in ``done``. Tests that want to exercise default strict
    # behaviour or other tiers pass an explicit ``governance=`` value.
    gov = (
        governance
        if governance is not None
        else GovernanceConfig(
            notify=[GovernanceRule(severity=[Severity.high])],
        )
    )
    return RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        issue_trackers=[],
        governance=gov,
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
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
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
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
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
    assert args.reason == "triage_transient"


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
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
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
        scm = _make_scm(gh)
        await run_advisory_workflow(
            db_session,
            repo,
            scm,
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
        scm = _make_scm(gh)
        schedule = AsyncMock()
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            schedule_retry=schedule,
        )

    assert run.state == AdvisoryWorkflowState.reporting.value
    assert run.retry_count == 1
    schedule.assert_awaited_once()
    assert schedule.await_args[0][0].reason == "slack_transient"


@pytest.mark.asyncio
async def test_advisory_to_slack_seconds_metric_emitted(db_session, mocker) -> None:
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
        scm = _make_scm(gh)
        with patch("agents.orchestrator._LOG") as mock_log:
            mock_log.bind.return_value = mock_log
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-TEST-ABCD-EFGH",
            )

    assert run.state == AdvisoryWorkflowState.done.value

    metric_calls = [
        c for c in mock_log.info.call_args_list if c.kwargs.get("metric_name") == "advisory_to_slack_seconds"
    ]
    assert len(metric_calls) == 1
    kw = metric_calls[0].kwargs
    assert kw["duration_seconds"] >= 0
    assert "workflow_run_id" in kw


# ── Governance tiering ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_resolve_tier_skips_slack_and_finishes(db_session, mocker) -> None:
    """auto_resolve tier: log decision, transition to done, never call Slack."""
    repo = _repo(
        governance=GovernanceConfig(auto_resolve=[GovernanceRule(duplicate=True)]),
    )

    async def fake_triage(session, *_a: object, **_k: object) -> Finding:
        f = Finding(
            workflow=WorkflowKind.advisory,
            source_ref="https://github.com/advisories/GHSA-DUP",
            severity=Severity.low,
            ssvc_action=SSVCAction.track,
            status=FindingStatus.unconfirmed,
            triage_confidence=0.6,
            known_status=KnownStatus.duplicate,
            title="Duplicate advisory",
        )
        session.add(f)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)

    slack_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        slack_calls.append(request)
        return httpx.Response(200, json={"ok": True, "channel": "C", "ts": "1.0"})

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(handler)) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None
    assert slack_calls == []  # no Slack post for auto_resolve

    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    gov_logs = [log for log in logs if log.tool_name == "governance.decide"]
    assert len(gov_logs) == 1
    assert gov_logs[0].tool_output == "auto_resolve"


@pytest.mark.asyncio
async def test_notify_tier_posts_slack_and_finishes(db_session, mocker) -> None:
    """notify tier: post to Slack, transition straight to done (non-blocking)."""

    async def fake_triage(session, *_a: object, **_k: object) -> Finding:
        f = Finding(
            workflow=WorkflowKind.advisory,
            source_ref="https://github.com/advisories/GHSA-NOTIFY",
            severity=Severity.medium,
            ssvc_action=SSVCAction.attend,
            status=FindingStatus.unconfirmed,
            triage_confidence=0.7,
            title="Medium advisory",
        )
        session.add(f)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)

    repo = _repo(
        governance=GovernanceConfig(notify=[GovernanceRule(severity=[Severity.medium])]),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    gov_logs = [log for log in logs if log.tool_name == "governance.decide"]
    assert len(gov_logs) == 1
    assert gov_logs[0].tool_output == "notify"


@pytest.mark.asyncio
async def test_approve_tier_parks_in_awaiting_approval(db_session, mocker) -> None:
    """approve tier: Slack posted, run parked awaiting human approval (no completed_at)."""

    async def fake_triage(session, *_a: object, **_k: object) -> Finding:
        f = Finding(
            workflow=WorkflowKind.advisory,
            source_ref="https://github.com/advisories/GHSA-CRIT",
            severity=Severity.critical,
            ssvc_action=SSVCAction.immediate,
            status=FindingStatus.unconfirmed,
            triage_confidence=0.95,
            title="Critical advisory",
        )
        session.add(f)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)

    repo = _repo(
        governance=GovernanceConfig(approve=[GovernanceRule(severity=[Severity.critical])]),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.awaiting_approval.value
    assert run.completed_at is None  # intentionally unfinished — interactive approval will finalise
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    gov_logs = [log for log in logs if log.tool_name == "governance.decide"]
    assert len(gov_logs) == 1
    assert gov_logs[0].tool_output == "approve"


@pytest.mark.asyncio
async def test_default_governance_parks_high_severity_in_awaiting_approval(db_session, mocker) -> None:
    """No governance block: non-informational severity defaults to approve tier (strict)."""

    async def fake_triage(session, *_a: object, **_k: object) -> Finding:
        f = Finding(
            workflow=WorkflowKind.advisory,
            source_ref="https://github.com/advisories/GHSA-DEFAULT",
            severity=Severity.high,
            ssvc_action=SSVCAction.act,
            status=FindingStatus.unconfirmed,
            triage_confidence=0.8,
            title="High advisory",
        )
        session.add(f)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)

    # Explicitly pass RepoConfig without governance to exercise the default.
    repo_no_gov = RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        issue_trackers=[],
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        run = await run_advisory_workflow(
            db_session,
            repo_no_gov,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.awaiting_approval.value
    assert run.completed_at is None

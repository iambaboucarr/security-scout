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
from config import GovernanceConfig, GovernanceRule, RepoConfig
from models import Finding, FindingStatus, Severity, SSVCAction, WorkflowKind, WorkflowRun
from tools.github import GitHubAPIError, GitHubClient
from tools.scm.github import GitHubSCMProvider
from tools.slack import SlackClient


def _make_scm(gh: object) -> GitHubSCMProvider:
    return GitHubSCMProvider.from_client(gh)  # type: ignore[arg-type]


def _repo() -> RepoConfig:
    # Routes severity=high to the notify tier so resume flows terminate in ``done``
    # (otherwise the default strict behaviour parks high-severity runs in ``awaiting_approval``).
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
            return httpx.Response(
                200,
                json={"ok": True, "channel": "C123", "ts": "1234.5678"},
            )
        return httpx.Response(404, json={"ok": False})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_resume_reporting_skips_triage(db_session, mocker) -> None:
    """Deferred Slack retry must not create a second WorkflowRun or re-run triage."""

    async def _triage_must_not_run(*_a: object, **_k: object) -> Finding:
        msg = "triage should not run"
        raise AssertionError(msg)

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=_triage_must_not_run)

    repo = _repo()
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
    db_session.add(f)
    await db_session.flush()

    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.reporting.value,
        retry_count=1,
        finding_id=f.id,
    )
    db_session.add(wr)
    await db_session.commit()

    run_id = wr.id

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        out = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", resume_workflow_run_id=run_id),
        )

    assert out.state == AdvisoryWorkflowState.done.value
    rows = (await db_session.execute(select(WorkflowRun))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_resume_triaging_github_transient_no_second_workflow_run(db_session, mocker) -> None:
    repo = _repo()
    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=GitHubAPIError("upstream", is_transient=True, http_status=503),
    )
    schedule = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        out = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", schedule_retry=schedule),
        )

    assert out.state == AdvisoryWorkflowState.triaging.value
    assert out.retry_count == 1
    assert len((await db_session.execute(select(WorkflowRun))).scalars().all()) == 1

    mocker.patch(
        "agents.orchestrator.advisory_triage.run_advisory_triage",
        side_effect=GitHubAPIError("upstream", is_transient=True, http_status=503),
    )
    schedule2 = AsyncMock()

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        out2 = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(
                ghsa_id="GHSA-TEST-ABCD-EFGH", resume_workflow_run_id=out.id, schedule_retry=schedule2
            ),
        )

    assert out2.retry_count == 2
    assert out2.state == AdvisoryWorkflowState.triaging.value
    schedule2.assert_awaited_once()
    args = schedule2.await_args[0][0]
    assert isinstance(args, ScheduleRetryParams)
    assert len((await db_session.execute(select(WorkflowRun))).scalars().all()) == 1

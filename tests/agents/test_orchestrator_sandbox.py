# SPDX-License-Identifier: Apache-2.0
"""Orchestrator sandbox execution integration tests.

Tests the building_env → executing_sandbox → sandbox_complete pipeline
with mocked env_builder and sandbox_executor.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

from agents.env_builder import DetectedStack, EnvBuildResult
from agents.orchestrator import AdvisoryWorkflowState, run_advisory_workflow
from agents.sandbox_executor import ExecutionResult, PocType
from config import GovernanceConfig, GovernanceRule, RepoConfig
from exceptions import PermanentError, TransientError
from models import AgentActionLog, Finding, FindingStatus, Severity, SSVCAction, WorkflowKind, WorkflowRun
from tools.github import GitHubClient
from tools.scm.github import CloneError, GitHubSCMProvider
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
    reproduction: str = "requests.get('http://localhost')",
    cwe_ids: list[str] | None = None,
) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
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


def _env_result(tmp_path: Path) -> EnvBuildResult:
    return EnvBuildResult(
        image_tag="securityscout/sandbox:latest",
        repo_path=tmp_path / "repo",
        detected_stack=DetectedStack.PYTHON,
        build_log="ok",
    )


def _exec_result_confirmed_low() -> ExecutionResult:
    return ExecutionResult(
        confidence_tier=FindingStatus.confirmed_low,
        evidence_excerpt="SQL injection successful",
        raw_stdout="SQL injection successful",
        raw_stderr="",
        elapsed_seconds=2.5,
        poc_type=PocType.RESEARCHER_SUBMITTED,
        exit_code=0,
    )


def _exec_result_error() -> ExecutionResult:
    return ExecutionResult(
        confidence_tier=FindingStatus.error,
        evidence_excerpt="OOMKilled",
        raw_stdout="",
        raw_stderr="OOMKilled",
        elapsed_seconds=5.0,
        poc_type=PocType.RESEARCHER_SUBMITTED,
        exit_code=137,
    )


# ---------------------------------------------------------------------------
# Happy path: pre-flight CLEAN → building_env → executing_sandbox → sandbox_complete → done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_happy_path_confirmed_low(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="requests.get('http://localhost/admin')")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        return_value=_env_result(tmp_path),
    )
    mocker.patch(
        "agents.orchestrator.execute_poc",
        new_callable=AsyncMock,
        return_value=_exec_result_confirmed_low(),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
        )

    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None

    # Verify sandbox action logs
    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    env_logs = [r for r in logs if r.agent == "env_builder"]
    assert len(env_logs) >= 1
    exec_logs = [r for r in logs if r.agent == "sandbox_executor"]
    assert len(exec_logs) >= 1
    assert "confirmed_low" in (exec_logs[0].tool_output or "")

    # Verify finding updated
    finding = await db_session.get(Finding, run.finding_id)
    assert finding is not None
    assert finding.status == FindingStatus.confirmed_low
    assert finding.poc_executed is True
    assert finding.evidence is not None


# ---------------------------------------------------------------------------
# Sandbox execution returns ERROR tier → still reaches reporting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_error_tier_proceeds_to_reporting(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="some exploit")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        return_value=_env_result(tmp_path),
    )
    mocker.patch(
        "agents.orchestrator.execute_poc",
        new_callable=AsyncMock,
        return_value=_exec_result_error(),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
        )

    assert run.state == AdvisoryWorkflowState.done.value

    finding = await db_session.get(Finding, run.finding_id)
    assert finding is not None
    assert finding.status == FindingStatus.error
    assert finding.poc_executed is True


# ---------------------------------------------------------------------------
# Environment build failure → error_sandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_build_permanent_error(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="requests.get('http://localhost')")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        side_effect=PermanentError("docker build failed"),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
        )

    assert run.state == AdvisoryWorkflowState.error_sandbox.value
    assert run.completed_at is not None
    assert "docker build failed" in (run.error_message or "")


# ---------------------------------------------------------------------------
# Environment build transient error → retry when schedule_retry available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_build_transient_error_retries(db_session, mocker, tmp_path) -> None:
    repo = _repo()
    retries: list = []

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="requests.get('http://localhost')")
        await session.flush()
        return f

    async def fake_retry(params):
        retries.append(params)

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        side_effect=TransientError("network flake"),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
            schedule_retry=fake_retry,
        )

    assert len(retries) == 1
    assert retries[0].state == AdvisoryWorkflowState.building_env.value
    assert retries[0].reason == "env_build_transient"


@pytest.mark.asyncio
async def test_env_build_transient_security_scout_error_retries(db_session, mocker, tmp_path) -> None:
    """Transient ``CloneError`` (``is_transient=True``) must retry like ``TransientError``."""
    repo = _repo()
    retries: list = []

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="requests.get('http://localhost')")
        await session.flush()
        return f

    async def fake_retry(params):
        retries.append(params)

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        side_effect=CloneError("git fetch timed out", is_transient=True),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
            schedule_retry=fake_retry,
        )

    assert len(retries) == 1
    assert retries[0].state == AdvisoryWorkflowState.building_env.value
    assert retries[0].reason == "env_build_transient"


# ---------------------------------------------------------------------------
# Resume from executing_sandbox → rebuild env → completed run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_executing_sandbox_rebuilds_env(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(*_a: object, **_k: object) -> Finding:
        msg = "triage must not run on resume"
        raise AssertionError(msg)

    f = _finding_with_poc(db_session, reproduction="print(1)")
    await db_session.flush()

    wr = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.executing_sandbox.value,
        retry_count=0,
        finding_id=f.id,
    )
    db_session.add(wr)
    await db_session.commit()

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    build_mock = mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        return_value=_env_result(tmp_path),
    )
    mocker.patch(
        "agents.orchestrator.execute_poc",
        new_callable=AsyncMock,
        return_value=_exec_result_confirmed_low(),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            resume_workflow_run_id=wr.id,
            work_dir=tmp_path,
        )

    assert run.state == AdvisoryWorkflowState.done.value
    build_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# Sandbox execution exception → error_sandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_execution_exception(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="some poc")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        return_value=_env_result(tmp_path),
    )
    mocker.patch(
        "agents.orchestrator.execute_poc",
        new_callable=AsyncMock,
        side_effect=RuntimeError("container crashed"),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
        )

    assert run.state == AdvisoryWorkflowState.error_sandbox.value
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_sandbox_execute_notimplemented_maps_to_error_sandbox(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="some poc")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        return_value=_env_result(tmp_path),
    )
    mocker.patch(
        "agents.orchestrator.execute_poc",
        new_callable=AsyncMock,
        side_effect=NotImplementedError("run_container is not implemented yet"),
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
        )

    assert run.state == AdvisoryWorkflowState.error_sandbox.value
    assert "not implemented" in (run.error_message or "").lower()


# ---------------------------------------------------------------------------
# No PoC → skip sandbox entirely → reporting → done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_poc_skips_sandbox(db_session, mocker) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction=None)
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    build_mock = mocker.patch("agents.orchestrator.build_environment", new_callable=AsyncMock)

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        run = await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
        )

    assert run.state == AdvisoryWorkflowState.done.value
    build_mock.assert_not_called()


# ---------------------------------------------------------------------------
# sandbox_execution_seconds metric emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_metric_emitted(db_session, mocker, tmp_path) -> None:
    repo = _repo()

    async def fake_triage(session, *args, **kwargs):
        f = _finding_with_poc(session, reproduction="requests.get('http://localhost')")
        await session.flush()
        return f

    mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=fake_triage)
    mocker.patch(
        "agents.orchestrator.build_environment",
        new_callable=AsyncMock,
        return_value=_env_result(tmp_path),
    )
    mocker.patch(
        "agents.orchestrator.execute_poc",
        new_callable=AsyncMock,
        return_value=_exec_result_confirmed_low(),
    )

    log_calls = []
    mocker.patch(
        "agents.orchestrator._LOG",
        **{
            "bind.return_value": MagicMock(
                info=lambda *a, **kw: log_calls.append(("info", kw)),
                warning=lambda *a, **kw: log_calls.append(("warning", kw)),
                exception=lambda *a, **kw: log_calls.append(("exception", kw)),
            ),
        },
    )

    async with httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok()) as http:
        slack = SlackClient("xoxb-test", client=http)
        scm = _make_scm(MagicMock(spec=GitHubClient))
        await run_advisory_workflow(
            db_session,
            repo,
            scm,
            http,
            slack,
            ghsa_id="GHSA-TEST-ABCD-EFGH",
            work_dir=tmp_path,
        )

    sandbox_metrics = [c for _, c in log_calls if c.get("metric_name") == "sandbox_execution_seconds"]
    assert len(sandbox_metrics) >= 1
    assert sandbox_metrics[0]["duration_seconds"] == 2.5

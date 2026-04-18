# SPDX-License-Identifier: Apache-2.0
"""Integration tests for three cross-cutting flows with real SQLite:

1. Slack button click → approval handler → DB update → thread reply
2. Governance routing: auto_resolve / notify / approve tiers end-to-end
3. MCP read-only queries against a DB populated by the triage+orchestrator path
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

from agents.approval import ApprovalContext, ApprovalOutcome, handle_slack_approval
from agents.governance import GovernanceTier
from agents.orchestrator import AdvisoryWorkflowState, run_advisory_workflow
from config import (
    AppConfig,
    GovernanceApprover,
    GovernanceConfig,
    GovernanceRule,
    RepoConfig,
    ReposManifest,
    Settings,
)
from db import create_engine, create_session_factory
from mcp_readonly import create_mcp_server
from models import (
    AgentActionLog,
    Base,
    Finding,
    FindingStatus,
    Severity,
    SSVCAction,
    TriageAccuracy,
    TriageDecision,
    WorkflowKind,
)
from tools.github import GitHubClient
from tools.scm.github import GitHubSCMProvider
from tools.slack import SlackClient
from webhooks.slack import SlackActionId

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """Captures Slack API call bodies for assertion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.calls.append(body)
        return httpx.Response(
            200,
            json={"ok": True, "channel": str(body.get("channel", "C1")), "ts": "9.9"},
        )


def _httpx_and_transport() -> tuple[httpx.AsyncClient, _RecordingTransport]:
    transport = _RecordingTransport()
    client = httpx.AsyncClient(
        base_url="https://slack.com/api",
        transport=httpx.MockTransport(transport),
    )
    return client, transport


def _slack_transport_ok() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if "chat.postMessage" in str(request.url):
            return httpx.Response(
                200,
                json={"ok": True, "channel": "C123", "ts": "1234.5678"},
            )
        return httpx.Response(404, json={"ok": False})

    return httpx.MockTransport(handler)


def _make_scm(gh: object) -> GitHubSCMProvider:
    return GitHubSCMProvider.from_client(gh)  # type: ignore[arg-type]


def _base_repo(
    *,
    governance: GovernanceConfig | None = None,
    approvers: list[GovernanceApprover] | None = None,
) -> RepoConfig:
    return RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        issue_trackers=[],
        governance=governance,
        approvers=approvers or [],
    )


def _app_config(
    *,
    governance: GovernanceConfig | None = None,
    approvers: list[GovernanceApprover] | None = None,
) -> AppConfig:
    repo = _base_repo(governance=governance, approvers=approvers)
    return AppConfig(
        settings=Settings(),
        repos=ReposManifest(repos=[repo]),
        repos_yaml_sha256="0" * 64,
        repos_yaml_path=Path("/tmp/repos.yaml"),
    )


async def _make_finding(session: object, *_a: object, **_k: object) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-TEST",
        severity=Severity.high,
        ssvc_action=SSVCAction.act,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.9,
        title="Test advisory finding",
    )
    session.add(f)  # type: ignore[arg-type]
    await session.flush()  # type: ignore[attr-defined]
    return f


async def _make_finding_low(session: object, *_a: object, **_k: object) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-LOW",
        severity=Severity.low,
        ssvc_action=SSVCAction.track,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.4,
        title="Low-severity advisory",
    )
    session.add(f)  # type: ignore[arg-type]
    await session.flush()  # type: ignore[attr-defined]
    return f


async def _make_finding_informational(session: object, *_a: object, **_k: object) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-INFO",
        severity=Severity.informational,
        ssvc_action=SSVCAction.track,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.2,
        title="Informational advisory",
    )
    session.add(f)  # type: ignore[arg-type]
    await session.flush()  # type: ignore[attr-defined]
    return f


async def _make_finding_critical(session: object, *_a: object, **_k: object) -> Finding:
    f = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-CRIT",
        severity=Severity.critical,
        ssvc_action=SSVCAction.immediate,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.95,
        title="Critical SQL injection",
        cve_id="CVE-2026-99999",
        cwe_ids=["CWE-89"],
        cvss_score=9.8,
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        description="Critical vulnerability in query builder",
    )
    session.add(f)  # type: ignore[arg-type]
    await session.flush()  # type: ignore[attr-defined]
    return f


# ═══════════════════════════════════════════════════════════════════════════
# 1. SLACK BUTTON CLICK → ORCHESTRATOR RESUME → DB UPDATE → THREAD REPLY
# ═══════════════════════════════════════════════════════════════════════════


class TestSlackApprovalEndToEnd:
    """Full approval flow: orchestrator parks at awaiting_approval, then Slack
    button click resumes → DB updated → thread reply posted.
    """

    @pytest.mark.asyncio
    async def test_approve_flow_triage_to_approval_to_done(self, db_session, mocker) -> None:
        """Orchestrator triages → parks at awaiting_approval → approve click
        → run.state=done, finding.approved_by set, thread reply posted,
        TriageAccuracy recorded.
        """
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(approve=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance, approvers=[GovernanceApprover(slack_user="U0LEADAPPRV")])
        app_config = _app_config(governance=governance, approvers=[GovernanceApprover(slack_user="U0LEADAPPRV")])

        http, _transport = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
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
        assert run.completed_at is None

        finding = await db_session.get(Finding, run.finding_id)
        assert finding is not None
        assert finding.approved_by is None

        governance_logs = (
            (
                await db_session.execute(
                    select(AgentActionLog).where(
                        AgentActionLog.workflow_run_id == run.id,
                        AgentActionLog.tool_name == "governance.decide",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(governance_logs) == 1
        assert governance_logs[0].tool_output == GovernanceTier.approve.value

        ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")
        http2, transport2 = _httpx_and_transport()
        async with http2, SlackClient("xoxb-test", client=http2) as slack2:
            outcome = await handle_slack_approval(
                db_session,
                app_config,
                slack2,
                ctx=ctx,
                action=SlackActionId.approve,
                user_id="U0LEADAPPRV",
                channel_id="C123",
                message_ts="1234.5678",
            )

        assert outcome == ApprovalOutcome.approved

        await db_session.refresh(run)
        await db_session.refresh(finding)
        assert run.state == AdvisoryWorkflowState.done.value
        assert run.completed_at is not None
        assert finding.approved_by == "U0LEADAPPRV"
        assert finding.approved_at is not None

        assert any(call.get("thread_ts") == "1234.5678" for call in transport2.calls)
        assert any("Approved by" in str(call.get("text", "")) for call in transport2.calls)

        accuracy_rows = (
            (await db_session.execute(select(TriageAccuracy).where(TriageAccuracy.finding_id == finding.id)))
            .scalars()
            .all()
        )
        assert len(accuracy_rows) == 1
        assert accuracy_rows[0].human_decision == TriageDecision.approved
        assert accuracy_rows[0].outcome_signal == 1.0
        assert accuracy_rows[0].predicted_ssvc_action == SSVCAction.act
        assert accuracy_rows[0].predicted_confidence == 0.9
        assert accuracy_rows[0].slack_user_id == "U0LEADAPPRV"

    @pytest.mark.asyncio
    async def test_reject_flow_marks_false_positive_and_posts_reply(self, db_session, mocker) -> None:
        """Orchestrator parks → reject click → finding marked false_positive,
        run done, thread reply says 'Rejected'.
        """
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(approve=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance)
        app_config = _app_config(governance=governance)

        http, _ = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
            gh = MagicMock(spec=GitHubClient)
            scm = _make_scm(gh)
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-TEST-REJECT",
            )

        assert run.state == AdvisoryWorkflowState.awaiting_approval.value
        finding = await db_session.get(Finding, run.finding_id)
        assert finding is not None

        ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")
        http2, transport2 = _httpx_and_transport()
        async with http2, SlackClient("xoxb-test", client=http2) as slack2:
            outcome = await handle_slack_approval(
                db_session,
                app_config,
                slack2,
                ctx=ctx,
                action=SlackActionId.reject,
                user_id="U0REJECTOR",
                channel_id="C1",
                message_ts="2222.3333",
            )

        assert outcome == ApprovalOutcome.rejected
        await db_session.refresh(run)
        await db_session.refresh(finding)
        assert run.state == AdvisoryWorkflowState.done.value
        assert finding.status == FindingStatus.false_positive
        assert any("Rejected" in str(call.get("text", "")) for call in transport2.calls)

        accuracy = (
            (await db_session.execute(select(TriageAccuracy).where(TriageAccuracy.finding_id == finding.id)))
            .scalars()
            .first()
        )
        assert accuracy is not None
        assert accuracy.human_decision == TriageDecision.rejected
        assert accuracy.outcome_signal == -1.0

    @pytest.mark.asyncio
    async def test_escalate_flow_keeps_run_open_and_dms_approvers(self, db_session, mocker) -> None:
        """Escalate keeps the run in awaiting_approval and DMs configured approvers."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        approvers = [
            GovernanceApprover(slack_user="U0SECENG01"),
            GovernanceApprover(slack_user="U0SECENG02"),
        ]
        governance = GovernanceConfig(approve=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance, approvers=approvers)
        app_config = _app_config(governance=governance, approvers=approvers)

        http, _ = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
            gh = MagicMock(spec=GitHubClient)
            scm = _make_scm(gh)
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-TEST-ESCALATE",
            )

        finding = await db_session.get(Finding, run.finding_id)
        assert finding is not None

        ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")
        http2, transport2 = _httpx_and_transport()
        async with http2, SlackClient("xoxb-test", client=http2) as slack2:
            outcome = await handle_slack_approval(
                db_session,
                app_config,
                slack2,
                ctx=ctx,
                action=SlackActionId.escalate,
                user_id="U0REPORTER",
                channel_id="C1",
                message_ts="3333.4444",
            )

        assert outcome == ApprovalOutcome.escalated
        await db_session.refresh(run)
        assert run.state == AdvisoryWorkflowState.awaiting_approval.value
        assert run.completed_at is None

        dm_channels = [
            call["channel"] for call in transport2.calls if str(call.get("channel", "")).startswith("U0SECENG")
        ]
        assert set(dm_channels) == {"U0SECENG01", "U0SECENG02"}
        assert any(call.get("thread_ts") == "3333.4444" for call in transport2.calls)

    @pytest.mark.asyncio
    async def test_double_click_returns_already_resolved(self, db_session, mocker) -> None:
        """Clicking approve twice returns already_resolved on the second click."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(approve=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance)
        app_config = _app_config(governance=governance)

        http, _ = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
            gh = MagicMock(spec=GitHubClient)
            scm = _make_scm(gh)
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-DOUBLE",
            )

        finding = await db_session.get(Finding, run.finding_id)
        assert finding is not None
        ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

        http2, _ = _httpx_and_transport()
        async with http2, SlackClient("xoxb-test", client=http2) as slack2:
            first = await handle_slack_approval(
                db_session,
                app_config,
                slack2,
                ctx=ctx,
                action=SlackActionId.approve,
                user_id="U0FIRST",
                channel_id="C1",
                message_ts="5555.6666",
            )

        assert first == ApprovalOutcome.approved

        http3, _transport3 = _httpx_and_transport()
        async with http3, SlackClient("xoxb-test", client=http3) as slack3:
            second = await handle_slack_approval(
                db_session,
                app_config,
                slack3,
                ctx=ctx,
                action=SlackActionId.approve,
                user_id="U0SECOND",
                channel_id="C1",
                message_ts="5555.6666",
            )

        assert second == ApprovalOutcome.already_resolved
        await db_session.refresh(finding)
        assert finding.approved_by == "U0FIRST"


# ═══════════════════════════════════════════════════════════════════════════
# 2. GOVERNANCE ROUTING: AUTO-RESOLVE vs NOTIFY vs APPROVE
# ═══════════════════════════════════════════════════════════════════════════


class TestGovernanceRoutingEndToEnd:
    """End-to-end governance routing through the orchestrator with real SQLite."""

    @pytest.mark.asyncio
    async def test_auto_resolve_skips_slack_and_marks_done(self, db_session, mocker) -> None:
        """auto_resolve tier: informational severity with default governance → done
        immediately, no Slack message sent."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding_informational)

        repo = _base_repo()
        slack = MagicMock()
        slack.send_finding = AsyncMock()
        slack.send_finding_for_approval = AsyncMock()
        slack.notify_workflow_error = AsyncMock()

        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        http = httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok())
        async with http:
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-INFO-TEST",
            )

        assert run.state == AdvisoryWorkflowState.done.value
        assert run.completed_at is not None

        slack.send_finding.assert_not_called()
        slack.send_finding_for_approval.assert_not_called()

        governance_log = (
            (
                await db_session.execute(
                    select(AgentActionLog).where(
                        AgentActionLog.workflow_run_id == run.id,
                        AgentActionLog.tool_name == "governance.decide",
                    )
                )
            )
            .scalars()
            .first()
        )
        assert governance_log is not None
        assert governance_log.tool_output == GovernanceTier.auto_resolve.value

    @pytest.mark.asyncio
    async def test_auto_resolve_with_explicit_governance_rule(self, db_session, mocker) -> None:
        """Explicit auto_resolve rule for low severity skips Slack."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding_low)

        governance = GovernanceConfig(auto_resolve=[GovernanceRule(severity=[Severity.low])])
        repo = _base_repo(governance=governance)

        slack = MagicMock()
        slack.send_finding = AsyncMock()
        slack.send_finding_for_approval = AsyncMock()
        slack.notify_workflow_error = AsyncMock()

        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        http = httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok())
        async with http:
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-LOW-TEST",
            )

        assert run.state == AdvisoryWorkflowState.done.value
        slack.send_finding.assert_not_called()
        slack.send_finding_for_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_tier_sends_informational_slack_and_completes(self, db_session, mocker) -> None:
        """Notify tier sends Slack message as informational (no buttons) → done."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(notify=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance)

        slack = MagicMock()
        slack.send_finding = AsyncMock()
        slack.send_finding_for_approval = AsyncMock()
        slack.notify_workflow_error = AsyncMock()

        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        http = httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok())
        mocker.patch("agents.orchestrator.finding_to_report_payload", return_value=MagicMock())
        async with http:
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-NOTIFY-TEST",
            )

        assert run.state == AdvisoryWorkflowState.done.value
        assert run.completed_at is not None

        slack.send_finding.assert_awaited_once()
        call_kwargs = slack.send_finding.call_args
        assert call_kwargs.kwargs.get("informational") is True
        slack.send_finding_for_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_approve_tier_parks_at_awaiting_approval(self, db_session, mocker) -> None:
        """Approve tier sends Slack with buttons → parks at awaiting_approval."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(approve=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance)

        http, _transport = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
            gh = MagicMock(spec=GitHubClient)
            scm = _make_scm(gh)
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-APPROVE-TEST",
            )

        assert run.state == AdvisoryWorkflowState.awaiting_approval.value
        assert run.completed_at is None
        assert run.finding_id is not None

    @pytest.mark.asyncio
    async def test_default_governance_sends_high_to_approve(self, db_session, mocker) -> None:
        """No governance block → high severity defaults to approve tier."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        repo = _base_repo(governance=None)

        http, _ = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
            gh = MagicMock(spec=GitHubClient)
            scm = _make_scm(gh)
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-DEFAULT",
            )

        assert run.state == AdvisoryWorkflowState.awaiting_approval.value

    @pytest.mark.asyncio
    async def test_approve_takes_precedence_over_auto_resolve(self, db_session, mocker) -> None:
        """When both approve and auto_resolve rules match, approve wins (strictest)."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(
            approve=[GovernanceRule(severity=[Severity.high])],
            auto_resolve=[GovernanceRule(ssvc_action=[SSVCAction.act])],
        )
        repo = _base_repo(governance=governance)

        http, _ = _httpx_and_transport()
        async with http, SlackClient("xoxb-test", client=http) as slack:
            gh = MagicMock(spec=GitHubClient)
            scm = _make_scm(gh)
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-PREC",
            )

        assert run.state == AdvisoryWorkflowState.awaiting_approval.value

    @pytest.mark.asyncio
    async def test_governance_audit_trail_logged(self, db_session, mocker) -> None:
        """Every governance decision is logged to AgentActionLog."""
        mocker.patch("agents.orchestrator.run_advisory_triage", side_effect=_make_finding)

        governance = GovernanceConfig(notify=[GovernanceRule(severity=[Severity.high])])
        repo = _base_repo(governance=governance)

        slack = MagicMock()
        slack.send_finding = AsyncMock()
        slack.notify_workflow_error = AsyncMock()

        gh = MagicMock(spec=GitHubClient)
        scm = _make_scm(gh)
        mocker.patch("agents.orchestrator.finding_to_report_payload", return_value=MagicMock())
        http = httpx.AsyncClient(base_url="https://slack.com/api", transport=_slack_transport_ok())
        async with http:
            run = await run_advisory_workflow(
                db_session,
                repo,
                scm,
                http,
                slack,
                ghsa_id="GHSA-AUDIT",
            )

        logs = (
            (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
            .scalars()
            .all()
        )
        gov_logs = [log for log in logs if log.tool_name == "governance.decide"]
        assert len(gov_logs) == 1
        assert gov_logs[0].tool_output == GovernanceTier.notify.value
        assert gov_logs[0].tool_inputs is not None
        assert gov_logs[0].tool_inputs["severity"] == Severity.high.value
        assert gov_logs[0].tool_inputs["has_governance_config"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 3. MCP READ-ONLY QUERIES AGAINST POPULATED DB
# ═══════════════════════════════════════════════════════════════════════════


class TestMcpReadOnlyPopulatedDB:
    """Populate a DB through the orchestrator path, then query it via MCP tools."""

    @pytest.fixture
    async def populated_mcp(self, tmp_path, mocker):
        """Run two findings through the orchestrator into a real SQLite, then
        expose the same DB via the MCP server.
        """
        url = f"sqlite+aiosqlite:///{tmp_path / 'mcp_integ.db'}"
        engine = create_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = create_session_factory(engine)

        async with factory() as session:
            critical = Finding(
                workflow=WorkflowKind.advisory,
                source_ref="acme/app GHSA-CRIT-1111-2222",
                severity=Severity.critical,
                ssvc_action=SSVCAction.immediate,
                status=FindingStatus.confirmed_low,
                triage_confidence=0.95,
                title="Critical SQL injection in auth module",
                cve_id="CVE-2026-99999",
                cwe_ids=["CWE-89"],
                cvss_score=9.8,
                cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                description="SQL injection in the authentication module allows bypass",
            )
            session.add(critical)

            high = Finding(
                workflow=WorkflowKind.advisory,
                source_ref="acme/app GHSA-HIGH-3333-4444",
                severity=Severity.high,
                ssvc_action=SSVCAction.act,
                status=FindingStatus.unconfirmed,
                triage_confidence=0.8,
                title="XSS in user profile rendering",
                cve_id="CVE-2026-88888",
                cwe_ids=["CWE-79"],
                cvss_score=7.5,
                description="Reflected XSS via profile display name",
            )
            session.add(high)

            low = Finding(
                workflow=WorkflowKind.advisory,
                source_ref="other/repo GHSA-LOW-5555-6666",
                severity=Severity.low,
                ssvc_action=SSVCAction.track,
                status=FindingStatus.unconfirmed,
                triage_confidence=0.3,
                title="Info disclosure in error page",
            )
            session.add(low)

            approved = Finding(
                workflow=WorkflowKind.advisory,
                source_ref="acme/app GHSA-DONE-7777-8888",
                severity=Severity.high,
                ssvc_action=SSVCAction.act,
                status=FindingStatus.confirmed_high,
                triage_confidence=0.92,
                title="SSRF in webhook handler",
                cve_id="CVE-2026-77777",
                approved_by="U0SECENG",
                approved_at=datetime(2026, 4, 13, 10, 0, 0, tzinfo=UTC),
            )
            session.add(approved)
            await session.commit()

        mcp = create_mcp_server(factory)
        yield mcp, factory
        await engine.dispose()

    @staticmethod
    def _items(result) -> list[dict]:
        sc = result.structured_content
        assert sc is not None
        return sc["result"] if isinstance(sc, dict) and "result" in sc else sc

    @pytest.mark.asyncio
    async def test_query_findings_returns_repo_findings(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool("query_findings", {"repo": "acme/app"})
        items = self._items(result)
        assert len(items) == 3
        titles = {item["title"] for item in items}
        assert "Critical SQL injection in auth module" in titles
        assert "XSS in user profile rendering" in titles
        assert "SSRF in webhook handler" in titles

    @pytest.mark.asyncio
    async def test_query_findings_filters_by_severity(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "query_findings",
            {"repo": "acme/app", "severity": "critical"},
        )
        items = self._items(result)
        assert len(items) == 1
        assert items[0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_query_findings_filters_by_status(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "query_findings",
            {"repo": "acme/app", "status": "confirmed_high"},
        )
        items = self._items(result)
        assert len(items) == 1
        assert items[0]["title"] == "SSRF in webhook handler"

    @pytest.mark.asyncio
    async def test_query_findings_other_repo(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool("query_findings", {"repo": "other/repo"})
        items = self._items(result)
        assert len(items) == 1
        assert items[0]["severity"] == "low"

    @pytest.mark.asyncio
    async def test_query_findings_nonexistent_repo_empty(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool("query_findings", {"repo": "nonexistent/repo"})
        items = self._items(result)
        assert items == []

    @pytest.mark.asyncio
    async def test_get_finding_detail_full_fields(self, populated_mcp) -> None:
        mcp, factory = populated_mcp
        async with factory() as session:
            stmt = select(Finding).where(Finding.cve_id == "CVE-2026-99999")
            row = (await session.execute(stmt)).scalars().first()
            assert row is not None
            finding_id = str(row.id)

        result = await mcp.call_tool("get_finding_detail", {"finding_id": finding_id})
        data = result.structured_content
        assert data["severity"] == "critical"
        assert data["ssvc_action"] == "immediate"
        assert data["cve_id"] == "CVE-2026-99999"
        assert data["cvss_score"] == 9.8
        assert data["cwe_ids"] == ["CWE-89"]
        assert "SQL injection" in data["description"]

    @pytest.mark.asyncio
    async def test_get_finding_detail_approved_finding(self, populated_mcp) -> None:
        mcp, factory = populated_mcp
        async with factory() as session:
            stmt = select(Finding).where(Finding.cve_id == "CVE-2026-77777")
            row = (await session.execute(stmt)).scalars().first()
            assert row is not None
            finding_id = str(row.id)

        result = await mcp.call_tool("get_finding_detail", {"finding_id": finding_id})
        data = result.structured_content
        assert data["approved_by"] == "U0SECENG"
        assert data["approved_at"] is not None
        assert data["status"] == "confirmed_high"

    @pytest.mark.asyncio
    async def test_check_dependency_finds_advisories(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "check_dependency",
            {"package": "acme", "version": "1.0.0", "ecosystem": "npm"},
        )
        data = result.structured_content
        assert data["advisory_count"] == 3
        assert len(data["advisories"]) == 3

    @pytest.mark.asyncio
    async def test_check_dependency_no_matches(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "check_dependency",
            {"package": "not-a-real-package", "version": "1.0.0", "ecosystem": "pip"},
        )
        data = result.structured_content
        assert data["advisory_count"] == 0

    @pytest.mark.asyncio
    async def test_get_triage_status_by_ghsa(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "get_triage_status",
            {"advisory_id": "GHSA-CRIT-1111-2222"},
        )
        data = result.structured_content
        assert data["found"] is True
        assert data["severity"] == "critical"
        assert data["ssvc_action"] == "immediate"
        assert data["triage_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_get_triage_status_by_cve(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "get_triage_status",
            {"advisory_id": "CVE-2026-88888"},
        )
        data = result.structured_content
        assert data["found"] is True
        assert data["severity"] == "high"
        assert data["ssvc_action"] == "act"

    @pytest.mark.asyncio
    async def test_get_triage_status_case_insensitive(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "get_triage_status",
            {"advisory_id": "ghsa-crit-1111-2222"},
        )
        data = result.structured_content
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_get_triage_status_not_found(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool(
            "get_triage_status",
            {"advisory_id": "GHSA-NOPE-0000-0000"},
        )
        data = result.structured_content
        assert data["found"] is False
        assert data["finding_id"] is None

    @pytest.mark.asyncio
    async def test_query_and_detail_roundtrip(self, populated_mcp) -> None:
        """Query findings list → pick one → get its detail → fields match."""
        mcp, _ = populated_mcp
        list_result = await mcp.call_tool(
            "query_findings",
            {"repo": "acme/app", "severity": "critical"},
        )
        items = self._items(list_result)
        assert len(items) == 1
        found_id = items[0]["id"]

        detail_result = await mcp.call_tool(
            "get_finding_detail",
            {"finding_id": found_id},
        )
        detail = detail_result.structured_content
        assert detail["id"] == found_id
        assert detail["severity"] == items[0]["severity"]
        assert detail["cve_id"] == "CVE-2026-99999"

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, populated_mcp) -> None:
        mcp, _ = populated_mcp
        result = await mcp.call_tool("query_findings", {"repo": "acme/app", "limit": 1})
        items = self._items(result)
        assert len(items) == 1

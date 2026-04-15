# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest
import structlog.testing
from sqlalchemy import select

from agents.approval import ApprovalContext, ApprovalOutcome, handle_slack_approval
from agents.orchestrator import AdvisoryWorkflowState
from config import (
    AppConfig,
    GovernanceApprover,
    RepoConfig,
    ReposManifest,
    Settings,
)
from models import (
    AgentActionLog,
    Finding,
    FindingStatus,
    Severity,
    SSVCAction,
    TriageAccuracy,
    TriageDecision,
    WorkflowKind,
    WorkflowRun,
)
from tools.slack import ApprovalButtonContext, SlackClient
from webhooks.slack import SlackActionId


def _app_config(*, approvers: list[GovernanceApprover] | None = None) -> AppConfig:
    repo = RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        approvers=approvers or [],
    )
    return AppConfig(
        settings=Settings(),
        repos=ReposManifest(repos=[repo]),
        repos_yaml_sha256="0" * 64,
        repos_yaml_path=Path("/tmp/repos.yaml"),
    )


async def _seed_awaiting_approval(
    session,
    *,
    ssvc_action: SSVCAction | None = SSVCAction.act,
    triage_confidence: float | None = 0.75,
) -> tuple[Finding, WorkflowRun]:
    finding = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/acme/app/security/advisories/GHSA-TEST",
        severity=Severity.high,
        ssvc_action=ssvc_action,
        status=FindingStatus.unconfirmed,
        title="Sample advisory",
        triage_confidence=triage_confidence,
    )
    session.add(finding)
    await session.flush()
    run = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=AdvisoryWorkflowState.awaiting_approval.value,
        retry_count=0,
        finding_id=finding.id,
    )
    session.add(run)
    await session.flush()
    return finding, run


class _RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.calls.append(body)
        return httpx.Response(200, json={"ok": True, "channel": str(body.get("channel", "C1")), "ts": "9.9"})


def _httpx_and_transport() -> tuple[httpx.AsyncClient, _RecordingTransport]:
    transport = _RecordingTransport()
    client = httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(transport))
    return client, transport


@pytest.mark.asyncio
async def test_handle_approve_marks_run_done_and_sets_finding_approved(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, transport = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01APPROVER",
            channel_id="C123",
            message_ts="1111.2222",
        )

    assert outcome == ApprovalOutcome.approved
    await db_session.refresh(run)
    await db_session.refresh(finding)
    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None
    assert finding.approved_by == "U01APPROVER"
    assert finding.approved_at is not None

    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "approval.approved" for log in logs)
    approved_log = next(log for log in logs if log.tool_name == "approval.approved")
    assert approved_log.tool_inputs is not None
    assert approved_log.tool_inputs["slack_user_id"] == "U01APPROVER"

    # Thread reply posted with thread_ts pointing at the original message.
    assert any(call.get("thread_ts") == "1111.2222" for call in transport.calls)
    assert any("Approved by" in str(call.get("text", "")) for call in transport.calls)


@pytest.mark.asyncio
async def test_handle_reject_marks_finding_false_positive(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.reject,
            user_id="U02REJECT",
            channel_id="C123",
            message_ts="1111.2222",
        )

    assert outcome == ApprovalOutcome.rejected
    await db_session.refresh(run)
    await db_session.refresh(finding)
    assert finding.status == FindingStatus.false_positive
    assert run.state == AdvisoryWorkflowState.done.value


@pytest.mark.asyncio
async def test_handle_escalate_dms_configured_approvers_and_keeps_run_open(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    approvers = [
        GovernanceApprover(slack_user="U0APPROVER1"),
        GovernanceApprover(slack_user="U0APPROVER2"),
    ]
    app_config = _app_config(approvers=approvers)
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, transport = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.escalate,
            user_id="U0REPORTER",
            channel_id="C123",
            message_ts="1111.2222",
        )

    assert outcome == ApprovalOutcome.escalated
    await db_session.refresh(run)
    assert run.state == AdvisoryWorkflowState.awaiting_approval.value
    assert run.completed_at is None

    dm_channels = [call["channel"] for call in transport.calls if call.get("channel", "").startswith("U0APPROVER")]
    assert set(dm_channels) == {"U0APPROVER1", "U0APPROVER2"}
    # Thread confirmation was also posted.
    assert any(call.get("thread_ts") == "1111.2222" for call in transport.calls)


@pytest.mark.asyncio
async def test_handle_escalate_skips_self_dm(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    approvers = [GovernanceApprover(slack_user="U0REPORTER")]
    app_config = _app_config(approvers=approvers)
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, transport = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.escalate,
            user_id="U0REPORTER",
            channel_id="C1",
            message_ts="1.0",
        )

    dm_channels = [call["channel"] for call in transport.calls if call["channel"] == "U0REPORTER"]
    assert dm_channels == []


@pytest.mark.asyncio
async def test_handle_second_click_is_noop(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    run.state = AdvisoryWorkflowState.done.value
    finding.approved_by = "U01OTHER"
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, transport = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U02",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == ApprovalOutcome.already_resolved
    await db_session.refresh(finding)
    assert finding.approved_by == "U01OTHER"
    assert any("already resolved" in str(c.get("text", "")) for c in transport.calls)


@pytest.mark.asyncio
async def test_button_value_roundtrip_through_context() -> None:
    encoded = ApprovalButtonContext(
        finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
        repo_name="demo",
    ).encode()
    ctx = ApprovalContext.from_button_value(encoded)
    assert ctx.repo_name == "demo"
    assert ctx.finding_id == uuid.UUID("11111111-2222-3333-4444-555555555555")
    assert ctx.workflow_run_id == uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa")


@pytest.mark.asyncio
async def test_unknown_workflow_run_returns_already_resolved(db_session) -> None:
    app_config = _app_config()
    fake_run_id = uuid.uuid4()
    fake_finding_id = uuid.uuid4()
    ctx = ApprovalContext(finding_id=fake_finding_id, workflow_run_id=fake_run_id, repo_name="demo")

    http, transport = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == ApprovalOutcome.already_resolved
    assert any("Could not find" in str(c.get("text", "")) for c in transport.calls)


@pytest.mark.asyncio
async def test_mismatched_finding_returns_already_resolved(db_session) -> None:
    _finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    wrong_finding_id = uuid.uuid4()
    ctx = ApprovalContext(finding_id=wrong_finding_id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == ApprovalOutcome.already_resolved


@pytest.mark.asyncio
async def test_unknown_finding_returns_already_resolved(db_session) -> None:
    """Simulate a dangling finding_id (DB inconsistency) by deleting the finding row."""
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    fid = finding.id
    rid = run.id

    from sqlalchemy import text

    await db_session.execute(text("PRAGMA foreign_keys = OFF"))
    await db_session.execute(text("DELETE FROM findings WHERE id = :id"), {"id": str(fid).replace("-", "")})
    await db_session.commit()
    await db_session.execute(text("PRAGMA foreign_keys = ON"))
    db_session.expire_all()

    app_config = _app_config()
    ctx = ApprovalContext(finding_id=fid, workflow_run_id=rid, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == ApprovalOutcome.already_resolved


@pytest.mark.asyncio
async def test_unknown_repo_returns_already_resolved(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="nonexistent-repo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == ApprovalOutcome.already_resolved


async def _fetch_accuracy(session, finding_id: uuid.UUID) -> list[TriageAccuracy]:
    result = await session.execute(select(TriageAccuracy).where(TriageAccuracy.finding_id == finding_id))
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_approve_records_triage_accuracy_with_positive_outcome_signal(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01APPROVER",
            channel_id="C1",
            message_ts="1.0",
        )

    rows = await _fetch_accuracy(db_session, finding.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.human_decision == TriageDecision.approved
    assert row.outcome_signal == 1.0
    assert row.predicted_ssvc_action == SSVCAction.act
    assert row.predicted_confidence == 0.75
    assert row.slack_user_id == "U01APPROVER"
    assert row.workflow_run_id == run.id


@pytest.mark.asyncio
async def test_reject_records_negative_outcome_signal(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.reject,
            user_id="U02REJECT",
            channel_id="C1",
            message_ts="1.0",
        )

    rows = await _fetch_accuracy(db_session, finding.id)
    assert len(rows) == 1
    assert rows[0].human_decision == TriageDecision.rejected
    assert rows[0].outcome_signal == -1.0


@pytest.mark.asyncio
async def test_escalate_records_zero_outcome_signal(db_session) -> None:
    finding, run = await _seed_awaiting_approval(
        db_session,
        ssvc_action=SSVCAction.immediate,
        triage_confidence=0.9,
    )
    await db_session.commit()
    app_config = _app_config(approvers=[GovernanceApprover(slack_user="U0APPROVER1")])
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.escalate,
            user_id="U0REPORTER",
            channel_id="C1",
            message_ts="1.0",
        )

    rows = await _fetch_accuracy(db_session, finding.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.human_decision == TriageDecision.escalated
    assert row.outcome_signal == 0.0
    assert row.predicted_ssvc_action == SSVCAction.immediate
    assert row.predicted_confidence == 0.9


@pytest.mark.asyncio
async def test_triage_accuracy_delta_metric_emitted(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    with structlog.testing.capture_logs() as captured:
        async with http, SlackClient("xoxb-t", client=http) as slack:
            await handle_slack_approval(
                db_session,
                app_config,
                slack,
                ctx=ctx,
                action=SlackActionId.reject,
                user_id="U02REJECT",
                channel_id="C1",
                message_ts="1.0",
            )

    metric_events = [e for e in captured if e.get("metric_name") == "triage_accuracy_delta"]
    assert len(metric_events) == 1
    evt = metric_events[0]
    assert evt["value"] == -1.0
    assert evt["human_decision"] == TriageDecision.rejected.value
    assert evt["predicted_ssvc_action"] == SSVCAction.act.value
    assert evt["predicted_confidence"] == 0.75


@pytest.mark.asyncio
async def test_accuracy_not_recorded_on_already_resolved(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    run.state = AdvisoryWorkflowState.done.value
    finding.approved_by = "U0PRIOR"
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    rows = await _fetch_accuracy(db_session, finding.id)
    assert rows == []


@pytest.mark.asyncio
async def test_accuracy_records_null_prediction_fields(db_session) -> None:
    finding, run = await _seed_awaiting_approval(
        db_session,
        ssvc_action=None,
        triage_confidence=None,
    )
    await db_session.commit()
    app_config = _app_config()
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.approve,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    rows = await _fetch_accuracy(db_session, finding.id)
    assert len(rows) == 1
    assert rows[0].predicted_ssvc_action is None
    assert rows[0].predicted_confidence is None
    assert rows[0].outcome_signal == 1.0


@pytest.mark.asyncio
async def test_duplicate_escalation_does_not_insert_second_accuracy_row(db_session) -> None:
    finding, run = await _seed_awaiting_approval(db_session)
    await db_session.commit()
    app_config = _app_config(approvers=[GovernanceApprover(slack_user="U0APPROVER1")])
    ctx = ApprovalContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _httpx_and_transport()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.escalate,
            user_id="U0REPORTER",
            channel_id="C1",
            message_ts="1.0",
        )
        await handle_slack_approval(
            db_session,
            app_config,
            slack,
            ctx=ctx,
            action=SlackActionId.escalate,
            user_id="U0REPORTER",
            channel_id="C1",
            message_ts="1.0",
        )

    rows = await _fetch_accuracy(db_session, finding.id)
    assert len(rows) == 1
    assert rows[0].human_decision == TriageDecision.escalated

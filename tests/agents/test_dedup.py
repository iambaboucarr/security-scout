# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid

import httpx
import pytest
import structlog.testing
from sqlalchemy import select

from agents.dedup import (
    DedupAction,
    DedupContext,
    DedupOutcome,
    dedup_action_from_action_id,
    handle_slack_dedup_decision,
    is_dedup_action_id,
)
from agents.orchestrator import AdvisoryWorkflowState
from models import (
    AgentActionLog,
    Finding,
    FindingStatus,
    KnownStatus,
    Severity,
    SSVCAction,
    WorkflowKind,
    WorkflowRun,
)
from tools.slack import ApprovalButtonContext, SlackClient
from webhooks.slack import SlackActionId


async def _seed_finding_and_run(
    session,
    *,
    state: str = AdvisoryWorkflowState.awaiting_approval.value,
    status: FindingStatus = FindingStatus.unconfirmed,
    duplicate_of: str | None = "JIRA-SEC-1234",
    duplicate_tracker: str | None = "jira",
) -> tuple[Finding, WorkflowRun]:
    finding = Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/acme/app/security/advisories/GHSA-TEST",
        severity=Severity.high,
        ssvc_action=SSVCAction.act,
        status=status,
        triage_confidence=0.8,
        title="Sample dedup finding",
        duplicate_of=duplicate_of,
        duplicate_tracker=duplicate_tracker,
        duplicate_url="https://acme.atlassian.net/browse/SEC-1234" if duplicate_of else None,
    )
    session.add(finding)
    await session.flush()
    run = WorkflowRun(
        workflow_type=WorkflowKind.advisory,
        state=state,
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


def _slack() -> tuple[httpx.AsyncClient, _RecordingTransport]:
    transport = _RecordingTransport()
    client = httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(transport))
    return client, transport


def test_is_dedup_action_id_recognises_all_six() -> None:
    assert is_dedup_action_id("security_scout:dedup_confirm")
    assert is_dedup_action_id("security_scout:dedup_new_instance")
    assert is_dedup_action_id("security_scout:dedup_reopen")
    assert is_dedup_action_id("security_scout:dedup_resolved")
    assert is_dedup_action_id("security_scout:risk_still_accepted")
    assert is_dedup_action_id("security_scout:risk_reevaluate")


def test_is_dedup_action_id_rejects_approval_actions() -> None:
    assert not is_dedup_action_id("security_scout:approve")
    assert not is_dedup_action_id("security_scout:reject")
    assert not is_dedup_action_id("security_scout:escalate")


def test_dedup_action_from_action_id_invalid_raises() -> None:
    with pytest.raises(ValueError, match="unknown dedup action_id"):
        dedup_action_from_action_id("nope")


def test_dedup_context_roundtrips_via_button_value() -> None:
    encoded = ApprovalButtonContext(
        finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
        repo_name="demo",
    ).encode()
    ctx = DedupContext.from_button_value(encoded)
    assert ctx.repo_name == "demo"
    assert ctx.finding_id == uuid.UUID("11111111-2222-3333-4444-555555555555")
    assert ctx.workflow_run_id == uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa")


@pytest.mark.asyncio
async def test_confirm_duplicate_sets_known_status_and_closes_run(db_session) -> None:
    finding, run = await _seed_finding_and_run(db_session)
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, transport = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.dedup_confirm,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == DedupOutcome.recorded
    await db_session.refresh(finding)
    await db_session.refresh(run)
    assert finding.known_status == KnownStatus.duplicate
    assert finding.approved_by == "U01"
    assert run.state == AdvisoryWorkflowState.done.value
    assert run.completed_at is not None

    logs = (
        (await db_session.execute(select(AgentActionLog).where(AgentActionLog.workflow_run_id == run.id)))
        .scalars()
        .all()
    )
    assert any(log.tool_name == "dedup.confirm_duplicate" for log in logs)
    assert any(call.get("thread_ts") == "1.0" for call in transport.calls)


@pytest.mark.asyncio
async def test_new_instance_marks_known_status_and_keeps_run_open(db_session) -> None:
    finding, run = await _seed_finding_and_run(db_session)
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.dedup_new_instance,
            user_id="U02",
            channel_id="C1",
            message_ts="1.0",
        )

    await db_session.refresh(finding)
    await db_session.refresh(run)
    assert finding.known_status == KnownStatus.new_instance
    assert run.state == AdvisoryWorkflowState.awaiting_approval.value
    assert run.completed_at is None
    # Approval not auto-recorded; standard approve/reject buttons still drive that.
    assert finding.approved_by is None


@pytest.mark.asyncio
async def test_reopen_clears_known_status_and_keeps_run_open(db_session) -> None:
    finding, run = await _seed_finding_and_run(db_session)
    finding.known_status = KnownStatus.known_resolved
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.dedup_reopen,
            user_id="U03",
            channel_id="C1",
            message_ts="1.0",
        )

    await db_session.refresh(finding)
    await db_session.refresh(run)
    assert finding.known_status is None
    assert run.state == AdvisoryWorkflowState.awaiting_approval.value


@pytest.mark.asyncio
async def test_confirm_resolved_closes_run_and_marks_known_resolved(db_session) -> None:
    finding, run = await _seed_finding_and_run(db_session)
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.dedup_resolved,
            user_id="U04",
            channel_id="C1",
            message_ts="1.0",
        )

    await db_session.refresh(finding)
    await db_session.refresh(run)
    assert finding.known_status == KnownStatus.known_resolved
    assert run.state == AdvisoryWorkflowState.done.value
    assert finding.approved_by == "U04"


@pytest.mark.asyncio
async def test_risk_still_accepted_keeps_finding_accepted(db_session) -> None:
    finding, run = await _seed_finding_and_run(
        db_session,
        status=FindingStatus.unconfirmed,
    )
    finding.known_status = KnownStatus.known_accepted_risk
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, transport = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.risk_still_accepted,
            user_id="U05",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == DedupOutcome.risk_still_accepted
    await db_session.refresh(finding)
    await db_session.refresh(run)
    assert finding.status == FindingStatus.accepted_risk
    assert finding.known_status == KnownStatus.known_accepted_risk
    assert run.state == AdvisoryWorkflowState.done.value
    assert any("Risk still accepted" in str(c.get("text", "")) for c in transport.calls)


@pytest.mark.asyncio
async def test_risk_reevaluate_routes_back_through_approval(db_session) -> None:
    finding, run = await _seed_finding_and_run(
        db_session,
        state=AdvisoryWorkflowState.done.value,
    )
    finding.known_status = KnownStatus.known_accepted_risk
    run.state = AdvisoryWorkflowState.done.value
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.risk_reevaluate,
            user_id="U06",
            channel_id="C1",
            message_ts="1.0",
        )

    assert outcome == DedupOutcome.risk_reevaluating
    await db_session.refresh(finding)
    await db_session.refresh(run)
    assert finding.known_status == KnownStatus.new_instance
    assert run.state == AdvisoryWorkflowState.awaiting_approval.value
    assert run.completed_at is None


@pytest.mark.asyncio
async def test_unknown_workflow_run_returns_unknown_run(db_session) -> None:
    ctx = DedupContext(finding_id=uuid.uuid4(), workflow_run_id=uuid.uuid4(), repo_name="demo")
    http, transport = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.dedup_confirm,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )
    assert outcome == DedupOutcome.unknown_run
    assert any("Could not find" in str(c.get("text", "")) for c in transport.calls)


@pytest.mark.asyncio
async def test_mismatched_finding_returns_mismatch(db_session) -> None:
    _finding, run = await _seed_finding_and_run(db_session)
    await db_session.commit()
    ctx = DedupContext(finding_id=uuid.uuid4(), workflow_run_id=run.id, repo_name="demo")
    http, _ = _slack()
    async with http, SlackClient("xoxb-t", client=http) as slack:
        outcome = await handle_slack_dedup_decision(
            db_session,
            slack,
            ctx=ctx,
            action_id=SlackActionId.dedup_confirm,
            user_id="U01",
            channel_id="C1",
            message_ts="1.0",
        )
    assert outcome == DedupOutcome.mismatched_finding


@pytest.mark.asyncio
async def test_dedup_human_decision_metric_emitted(db_session) -> None:
    finding, run = await _seed_finding_and_run(db_session)
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    with structlog.testing.capture_logs() as captured:
        async with http, SlackClient("xoxb-t", client=http) as slack:
            await handle_slack_dedup_decision(
                db_session,
                slack,
                ctx=ctx,
                action_id=SlackActionId.dedup_confirm,
                user_id="U01",
                channel_id="C1",
                message_ts="1.0",
            )

    metric_events = [e for e in captured if e.get("metric_name") == "dedup_human_decision"]
    assert len(metric_events) == 1
    assert metric_events[0]["action"] == DedupAction.confirm_duplicate.value


@pytest.mark.asyncio
async def test_dedup_false_duplicate_rate_emitted_only_for_new_instance_after_match(db_session) -> None:
    finding, run = await _seed_finding_and_run(db_session)
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    with structlog.testing.capture_logs() as captured:
        async with http, SlackClient("xoxb-t", client=http) as slack:
            await handle_slack_dedup_decision(
                db_session,
                slack,
                ctx=ctx,
                action_id=SlackActionId.dedup_new_instance,
                user_id="U01",
                channel_id="C1",
                message_ts="1.0",
            )

    fd_events = [e for e in captured if e.get("metric_name") == "dedup_false_duplicate_rate"]
    assert len(fd_events) == 1


@pytest.mark.asyncio
async def test_dedup_false_duplicate_rate_not_emitted_when_no_prior_match(db_session) -> None:
    finding, run = await _seed_finding_and_run(
        db_session,
        duplicate_of=None,
        duplicate_tracker=None,
    )
    await db_session.commit()
    ctx = DedupContext(finding_id=finding.id, workflow_run_id=run.id, repo_name="demo")

    http, _ = _slack()
    with structlog.testing.capture_logs() as captured:
        async with http, SlackClient("xoxb-t", client=http) as slack:
            await handle_slack_dedup_decision(
                db_session,
                slack,
                ctx=ctx,
                action_id=SlackActionId.dedup_new_instance,
                user_id="U01",
                channel_id="C1",
                message_ts="1.0",
            )

    fd_events = [e for e in captured if e.get("metric_name") == "dedup_false_duplicate_rate"]
    assert fd_events == []

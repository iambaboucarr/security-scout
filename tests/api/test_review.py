# SPDX-License-Identifier: Apache-2.0
"""HTTP review endpoint + governance gate behaviour."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agents.api_review import ReviewDecision
from api.tokens import create_token
from api.v1 import create_v1_router
from config import (
    AppConfig,
    GovernanceConfig,
    GovernanceRule,
    RepoConfig,
    ReposManifest,
    Settings,
)
from db import create_engine, create_session_factory, session_scope
from models import (
    AdvisoryWorkflowState,
    AgentActionLog,
    ApiTokenScope,
    Base,
    Finding,
    FindingStatus,
    Severity,
    SSVCAction,
    TriageAccuracy,
    TriageDecision,
    WorkflowKind,
    WorkflowRun,
)


@pytest.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'review_api.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()


def _app_config(*, governance: GovernanceConfig | None = None) -> AppConfig:
    repo = RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        governance=governance,
    )
    return AppConfig(
        settings=Settings(),
        repos=ReposManifest(repos=[repo]),
        repos_yaml_sha256="0" * 64,
        repos_yaml_path=Path("/tmp/repos.yaml"),
    )


def _build_app(
    factory: async_sessionmaker[AsyncSession],
    *,
    governance: GovernanceConfig | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.state.app_config = _app_config(governance=governance)
    app.include_router(create_v1_router())
    return app


async def _bearer_for(
    factory: async_sessionmaker[AsyncSession],
    *,
    scopes: list[ApiTokenScope],
) -> tuple[str, uuid.UUID]:
    async with session_scope(factory) as session:
        issued = await create_token(session, name="rev-test", scopes=scopes, owner_slack_id="U_REV")
    return issued.raw_token, issued.record.id


async def _seed_finding(
    factory: async_sessionmaker[AsyncSession],
    *,
    state: AdvisoryWorkflowState = AdvisoryWorkflowState.awaiting_approval,
    severity: Severity = Severity.high,
    ssvc_action: SSVCAction | None = SSVCAction.act,
) -> tuple[uuid.UUID, uuid.UUID]:
    async with session_scope(factory) as session:
        finding = Finding(
            workflow=WorkflowKind.advisory,
            repo_name="acme/app",
            source_ref="https://github.com/acme/app/security/advisories/GHSA-XXXX",
            severity=severity,
            ssvc_action=ssvc_action,
            status=FindingStatus.unconfirmed,
            title="Sample advisory",
            triage_confidence=0.7,
            created_at=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        )
        session.add(finding)
        await session.flush()
        run = WorkflowRun(
            workflow_type=WorkflowKind.advisory,
            state=state.value,
            retry_count=0,
            finding_id=finding.id,
            repo_name="acme/app",
        )
        session.add(run)
        await session.flush()
        return finding.id, run.id


async def _make_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_review_requires_write_scope(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    read_token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve"},
            headers={"Authorization": f"Bearer {read_token}"},
        )
    assert resp.status_code == 403


async def test_review_unauthenticated_returns_401(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve"},
        )
    assert resp.status_code == 401


async def test_review_approve_marks_finding_approved(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, run_id = await _seed_finding(session_factory)
    token, token_id = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve", "context": "Verified upstream patch landed."},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == ReviewDecision.approve.value
    assert body["finding_id"] == str(finding_id)
    assert body["workflow_run_id"] == str(run_id)
    assert body["reviewer_id"] == f"api:{token_id}"
    assert body["workflow_state"] == AdvisoryWorkflowState.done.value

    async with session_scope(session_factory) as session:
        finding = await session.get(Finding, finding_id)
        run = await session.get(WorkflowRun, run_id)
        assert finding is not None
        assert run is not None
        assert finding.approved_by == f"api:{token_id}"
        assert finding.approved_at is not None
        assert run.state == AdvisoryWorkflowState.done.value
        assert run.completed_at is not None
        accuracy = (await session.execute(select(TriageAccuracy))).scalars().all()
        assert len(accuracy) == 1
        assert accuracy[0].human_decision == TriageDecision.approved
        assert accuracy[0].outcome_signal == 1.0
        logs = (
            (await session.execute(select(AgentActionLog).where(AgentActionLog.agent == "api_review"))).scalars().all()
        )
        assert len(logs) == 1
        assert logs[0].tool_name == "review.approve"
        assert logs[0].tool_inputs is not None
        assert logs[0].tool_inputs["token_id"] == str(token_id)
        assert logs[0].tool_inputs["governance_tier"] == "approve"
        assert "Verified upstream patch landed." in logs[0].tool_inputs["context"]


async def test_review_reject_marks_false_positive(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    token, token_id = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "reject"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert resp.json()["finding_status"] == FindingStatus.false_positive.value

    async with session_scope(session_factory) as session:
        finding = await session.get(Finding, finding_id)
        assert finding is not None
        assert finding.status == FindingStatus.false_positive
        assert finding.approved_by == f"api:{token_id}"


async def test_review_escalate_keeps_run_open(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, run_id = await _seed_finding(session_factory)
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "escalate"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert resp.json()["workflow_state"] == AdvisoryWorkflowState.awaiting_approval.value

    async with session_scope(session_factory) as session:
        run = await session.get(WorkflowRun, run_id)
        finding = await session.get(Finding, finding_id)
        assert run is not None
        assert finding is not None
        assert run.state == AdvisoryWorkflowState.awaiting_approval.value
        assert finding.approved_by is None


async def test_review_missing_finding_returns_404(session_factory: async_sessionmaker[AsyncSession]) -> None:
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)
    missing = uuid.uuid4()

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{missing}/review",
            json={"decision": "approve"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 404


async def test_review_run_not_awaiting_returns_409(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory, state=AdvisoryWorkflowState.done)
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 409


async def test_review_blocked_when_governance_auto_resolve(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    governance = GovernanceConfig(
        auto_resolve=[GovernanceRule(severity=[Severity.high])],
        notify=[],
        approve=[],
    )
    app = _build_app(session_factory, governance=governance)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 409
    assert "governance" in resp.json()["detail"].lower()


async def test_review_rejects_unknown_decision(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "obliterate"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 422


async def test_review_rejects_extra_fields(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    app = _build_app(session_factory)

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve", "smuggled": "value"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 422


async def test_review_missing_app_config_returns_503(session_factory: async_sessionmaker[AsyncSession]) -> None:
    finding_id, _ = await _seed_finding(session_factory)
    token, _ = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])

    app = FastAPI()
    app.state.session_factory = session_factory
    app.include_router(create_v1_router())

    async with await _make_client(app) as client:
        resp = await client.post(
            f"/api/v1/findings/{finding_id}/review",
            json={"decision": "approve"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 503

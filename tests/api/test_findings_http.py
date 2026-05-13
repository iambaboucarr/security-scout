# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.findings import create_findings_router
from api.tokens import create_token, hash_token
from db import create_engine, create_session_factory, session_scope
from models import ApiToken, ApiTokenScope, Base, Finding, FindingStatus, Severity, SSVCAction, WorkflowKind


@pytest.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'findings_api.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()


def _build_findings_app(factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(create_findings_router())
    return app


@pytest.fixture
async def findings_client(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncClient]:
    app = _build_findings_app(session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _bearer_for(factory: async_sessionmaker[AsyncSession], *, scopes: list[ApiTokenScope]) -> str:
    async with session_scope(factory) as session:
        issued = await create_token(session, name="http-test", scopes=scopes, owner_slack_id="U_HTTP")
    return issued.raw_token


@pytest.fixture
async def read_token(session_factory: async_sessionmaker[AsyncSession]) -> str:
    return await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])


@pytest.fixture
async def populated_finding(session_factory: async_sessionmaker[AsyncSession]) -> uuid.UUID:
    fid = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    async with session_scope(session_factory) as session:
        session.add(
            Finding(
                id=fid,
                title="Sample finding",
                workflow=WorkflowKind.advisory,
                repo_name="acme/app",
                source_ref="acme/app GHSA-TEST-TEST-TEST",
                severity=Severity.high,
                ssvc_action=SSVCAction.act,
                status=FindingStatus.unconfirmed,
                created_at=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
            ),
        )
    return fid


async def test_findings_list_requires_auth(findings_client: AsyncClient) -> None:
    resp = await findings_client.get("/api/v1/findings", params={"repo": "acme/app"})
    assert resp.status_code == 401


async def test_findings_list_returns_data(
    findings_client: AsyncClient,
    read_token: str,
    populated_finding: uuid.UUID,
) -> None:
    resp = await findings_client.get(
        "/api/v1/findings",
        params={"repo": "acme/app"},
        headers={"Authorization": f"Bearer {read_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == str(populated_finding)
    assert data[0]["severity"] == "high"
    assert data[0]["title"] == "Sample finding"


async def test_findings_list_write_scope_rejected(
    findings_client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    write_only = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_write])
    resp = await findings_client.get(
        "/api/v1/findings",
        params={"repo": "acme/app"},
        headers={"Authorization": f"Bearer {write_only}"},
    )
    assert resp.status_code == 403


async def test_findings_detail_not_found(
    findings_client: AsyncClient,
    read_token: str,
) -> None:
    missing = uuid.uuid4()
    resp = await findings_client.get(
        f"/api/v1/findings/{missing}",
        headers={"Authorization": f"Bearer {read_token}"},
    )
    assert resp.status_code == 404


async def test_findings_detail_invalid_uuid(
    findings_client: AsyncClient,
    read_token: str,
) -> None:
    resp = await findings_client.get(
        "/api/v1/findings/not-a-uuid",
        headers={"Authorization": f"Bearer {read_token}"},
    )
    assert resp.status_code == 422


async def test_triage_and_dependency_smoke(
    findings_client: AsyncClient,
    read_token: str,
    populated_finding: uuid.UUID,
) -> None:
    _ = populated_finding
    headers = {"Authorization": f"Bearer {read_token}"}
    r1 = await findings_client.get(
        "/api/v1/triage/GHSA-TEST-TEST-TEST",
        headers=headers,
    )
    assert r1.status_code == 200
    assert r1.json()["found"] is True

    r2 = await findings_client.get(
        "/api/v1/dependencies/check",
        params={"package": "acme", "version": "1", "ecosystem": "npm"},
        headers=headers,
    )
    assert r2.status_code == 200
    assert r2.json()["advisory_count"] >= 1


async def test_unknown_scope_values_in_db_still_allow_known_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    raw = secrets.token_urlsafe(32)
    async with session_scope(session_factory) as session:
        session.add(
            ApiToken(
                name="tainted",
                token_hash=hash_token(raw),
                scopes=["findings:read", "findings:invalid-label", "garbage"],
                owner_slack_id="U1",
            ),
        )

    app = _build_findings_app(session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/v1/findings",
            params={"repo": "acme/app"},
            headers={"Authorization": f"Bearer {raw}"},
        )
    assert resp.status_code == 200


async def test_only_unknown_scopes_in_db_gets_403(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    raw = secrets.token_urlsafe(32)
    async with session_scope(session_factory) as session:
        session.add(
            ApiToken(
                name="noop",
                token_hash=hash_token(raw),
                scopes=["not-valid", "fake"],
                owner_slack_id="U1",
            ),
        )

    app = _build_findings_app(session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/v1/findings",
            params={"repo": "acme/app"},
            headers={"Authorization": f"Bearer {raw}"},
        )
    assert resp.status_code == 403

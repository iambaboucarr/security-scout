# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from api.auth import require_scope
from api.tokens import create_token
from db import session_scope
from models import AgentActionLog, ApiToken, ApiTokenScope


async def _issue(factory, *, scopes: list[ApiTokenScope], owner: str = "U1", name: str = "t") -> str:
    async with session_scope(factory) as session:
        issued = await create_token(session, name=name, scopes=scopes, owner_slack_id=owner)
        return issued.raw_token


async def test_protected_endpoint_accepts_valid_token(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "t"


async def test_missing_header_rejected_401(make_client, session_factory) -> None:
    _ = session_factory
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_malformed_scheme_rejected_401(make_client, session_factory) -> None:
    _ = session_factory
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": "Token abc"})
    assert resp.status_code == 401


async def test_empty_bearer_rejected_401(make_client, session_factory) -> None:
    _ = session_factory
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": "Bearer    "})
    assert resp.status_code == 401


async def test_unknown_token_rejected_401(make_client, session_factory) -> None:
    _ = session_factory
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get(
            "/protected",
            headers={"Authorization": "Bearer not-a-real-token-12345"},
        )
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_revoked_token_rejected_401(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    async with session_scope(session_factory) as session:
        row = await session.scalar(select(ApiToken))
        assert row is not None
        row.revoked_at = datetime.now(UTC)

    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


async def test_scope_mismatch_rejected_403(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    client = await make_client(ApiTokenScope.findings_write)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "scope" in resp.json()["detail"].lower()


async def test_successful_auth_updates_last_used_at(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    client = await make_client(ApiTokenScope.findings_read)
    before = datetime.now(UTC).replace(tzinfo=None)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200

    async with session_scope(session_factory) as session:
        row = await session.scalar(select(ApiToken))
        assert row is not None
        assert row.last_used_at is not None
        # SQLite drops tz on storage; Postgres preserves it. Normalise both sides.
        stored = row.last_used_at.replace(tzinfo=None) if row.last_used_at.tzinfo else row.last_used_at
        assert stored >= before


async def test_successful_auth_writes_action_log(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200

    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "authenticate")))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.agent == "api_auth"
        assert row.tool_inputs is not None
        assert row.tool_inputs["outcome"] == "ok"
        assert row.tool_inputs["required_scope"] == "findings:read"
        assert row.tool_inputs["path"] == "/protected"


async def test_scope_mismatch_writes_action_log(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    client = await make_client(ApiTokenScope.findings_write)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403

    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "authenticate")))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].tool_inputs is not None
        assert rows[0].tool_inputs["outcome"] == "scope_mismatch"


async def test_unknown_token_does_not_write_action_log(make_client, session_factory) -> None:
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get(
            "/protected",
            headers={"Authorization": "Bearer never-issued-token"},
        )
    assert resp.status_code == 401

    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "authenticate")))
            .scalars()
            .all()
        )
        assert rows == []


async def test_revoked_token_does_not_write_action_log(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    async with session_scope(session_factory) as session:
        row = await session.scalar(select(ApiToken))
        assert row is not None
        row.revoked_at = datetime.now(UTC)

    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        resp = await client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401

    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "authenticate")))
            .scalars()
            .all()
        )
        assert rows == []


async def test_missing_session_factory_returns_503() -> None:
    from fastapi import Depends, FastAPI
    from fastapi.responses import JSONResponse
    from httpx import ASGITransport, AsyncClient

    app = FastAPI()
    dep = Depends(require_scope(ApiTokenScope.findings_read))

    @app.get("/protected")
    async def protected(token: ApiToken = dep) -> JSONResponse:
        return JSONResponse({"id": str(token.id)})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/protected", headers={"Authorization": "Bearer any-token"})
    assert resp.status_code == 503
    assert "session" in resp.json()["detail"].lower()


def test_require_scope_rejects_non_enum_input() -> None:
    with pytest.raises(TypeError):
        require_scope("findings:read")  # type: ignore[arg-type]


async def test_multiple_calls_accumulate_logs(make_client, session_factory) -> None:
    token = await _issue(session_factory, scopes=[ApiTokenScope.findings_read])
    client = await make_client(ApiTokenScope.findings_read)
    async with client:
        for _ in range(3):
            resp = await client.get(
                "/protected",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "authenticate")))
            .scalars()
            .all()
        )
        assert len(rows) == 3

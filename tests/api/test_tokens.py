# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from api.tokens import create_token, hash_token, revoke_token
from models import AgentActionLog, ApiToken, ApiTokenScope


async def test_create_token_returns_raw_and_persists_hash(db_session) -> None:
    issued = await create_token(
        db_session,
        name="ci-pipeline",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U123ABC",
    )
    await db_session.commit()

    assert issued.raw_token
    assert len(issued.raw_token) >= 32
    assert issued.record.token_hash == hash_token(issued.raw_token)
    assert issued.record.token_hash != issued.raw_token
    assert issued.record.scopes == ["findings:read"]
    assert issued.record.owner_slack_id == "U123ABC"
    assert issued.record.revoked_at is None
    assert issued.record.last_used_at is None

    persisted = await db_session.get(ApiToken, issued.record.id)
    assert persisted is not None
    assert persisted.token_hash == issued.record.token_hash


async def test_create_token_appends_audit_log(db_session) -> None:
    issued = await create_token(
        db_session,
        name="ci",
        scopes=[ApiTokenScope.findings_read, ApiTokenScope.findings_write],
        owner_slack_id="U999",
    )
    await db_session.commit()

    result = await db_session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "create_token"))
    row = result.scalar_one()
    assert row.agent == "api_token"
    assert row.tool_inputs is not None
    assert row.tool_inputs["token_id"] == str(issued.record.id)
    assert row.tool_inputs["scopes"] == ["findings:read", "findings:write"]
    assert row.tool_inputs["owner_slack_id"] == "U999"
    assert "token_hash" not in row.tool_inputs
    assert "raw_token" not in row.tool_inputs


async def test_create_token_rejects_empty_name(db_session) -> None:
    with pytest.raises(ValueError, match="name"):
        await create_token(
            db_session,
            name="   ",
            scopes=[ApiTokenScope.findings_read],
            owner_slack_id="U1",
        )


async def test_create_token_rejects_empty_owner(db_session) -> None:
    with pytest.raises(ValueError, match="owner_slack_id"):
        await create_token(
            db_session,
            name="x",
            scopes=[ApiTokenScope.findings_read],
            owner_slack_id="",
        )


async def test_create_token_rejects_empty_scopes(db_session) -> None:
    with pytest.raises(ValueError, match="scopes"):
        await create_token(
            db_session,
            name="x",
            scopes=[],
            owner_slack_id="U1",
        )


async def test_create_token_rejects_unknown_scope(db_session) -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        await create_token(
            db_session,
            name="x",
            scopes=["findings:nuke"],
            owner_slack_id="U1",
        )


async def test_create_token_deduplicates_scopes(db_session) -> None:
    issued = await create_token(
        db_session,
        name="x",
        scopes=[ApiTokenScope.findings_read, ApiTokenScope.findings_read, "findings:read"],
        owner_slack_id="U1",
    )
    assert issued.record.scopes == ["findings:read"]


async def test_create_token_unique_hash(db_session) -> None:
    first = await create_token(
        db_session,
        name="a",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    second = await create_token(
        db_session,
        name="b",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U2",
    )
    await db_session.commit()
    assert first.raw_token != second.raw_token
    assert first.record.token_hash != second.record.token_hash


async def test_revoke_token_sets_revoked_at_and_audits(db_session) -> None:
    issued = await create_token(
        db_session,
        name="x",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    await db_session.commit()

    revoked = await revoke_token(db_session, token_id=issued.record.id, actor_slack_id="U_ADMIN")
    await db_session.commit()
    assert revoked is not None
    assert revoked.revoked_at is not None

    result = await db_session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "revoke_token"))
    row = result.scalar_one()
    assert row.tool_inputs is not None
    assert row.tool_inputs["token_id"] == str(issued.record.id)
    assert row.tool_inputs["actor_slack_id"] == "U_ADMIN"


async def test_revoke_token_unknown_id_returns_none(db_session) -> None:
    out = await revoke_token(db_session, token_id=uuid.uuid4(), actor_slack_id="U_ADMIN")
    assert out is None

    result = await db_session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "revoke_token"))
    assert result.scalars().all() == []


async def test_revoke_token_is_idempotent(db_session) -> None:
    issued = await create_token(
        db_session,
        name="x",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    await db_session.commit()

    first = await revoke_token(db_session, token_id=issued.record.id, actor_slack_id="U_ADMIN")
    await db_session.commit()
    assert first is not None
    first_ts = first.revoked_at

    second = await revoke_token(db_session, token_id=issued.record.id, actor_slack_id="U_ADMIN")
    await db_session.commit()
    assert second is not None
    assert second.revoked_at == first_ts

    result = await db_session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "revoke_token"))
    rows = result.scalars().all()
    assert len(rows) == 1


def test_hash_token_is_deterministic() -> None:
    assert hash_token("abc") == hash_token("abc")
    assert hash_token("abc") != hash_token("abd")
    assert len(hash_token("anything")) == 64

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

import admin
from api.tokens import create_token, hash_token
from db import create_engine
from models import AgentActionLog, ApiToken, ApiTokenScope, Base


def _ns(**fields: object) -> argparse.Namespace:
    return argparse.Namespace(**fields)


# ── Parser ───────────────────────────────────────────────────────────────────


def test_parser_requires_subcommand() -> None:
    parser = admin._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_create_token_requires_name_scope_owner() -> None:
    parser = admin._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["create_token", "--name", "x", "--owner", "U1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["create_token", "--scope", "findings:read", "--owner", "U1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["create_token", "--name", "x", "--scope", "findings:read"])


def test_parser_rejects_unknown_scope() -> None:
    parser = admin._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["create_token", "--name", "x", "--scope", "findings:nuke", "--owner", "U1"],
        )


def test_parser_accepts_multiple_scopes() -> None:
    parser = admin._build_parser()
    args = parser.parse_args(
        [
            "create_token",
            "--name",
            "ci",
            "--scope",
            "findings:read",
            "--scope",
            "findings:write",
            "--owner",
            "U1",
        ],
    )
    assert args.command == "create_token"
    assert args.scope == ["findings:read", "findings:write"]


def test_parser_revoke_requires_id() -> None:
    parser = admin._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["revoke_token"])


def test_parser_list_tokens_include_revoked_flag() -> None:
    parser = admin._build_parser()
    args = parser.parse_args(["list_tokens"])
    assert args.include_revoked is False
    args = parser.parse_args(["list_tokens", "--include-revoked"])
    assert args.include_revoked is True


# ── handle_create_token ──────────────────────────────────────────────────────


async def test_handle_create_token_writes_record_and_prints_raw_once(db_session) -> None:
    args = _ns(
        command="create_token",
        name="ci-pipeline",
        scope=["findings:read"],
        owner="U123ABC",
    )
    code, message = await admin.handle_create_token(db_session, args)
    await db_session.commit()

    assert code == admin.EXIT_OK
    assert "Token created" in message
    assert "cannot be retrieved later" in message

    raw_line = next(line for line in message.splitlines() if line.strip().startswith("token:"))
    raw = raw_line.split("token:", 1)[1].strip()
    assert len(raw) >= 32

    persisted = (await db_session.execute(select(ApiToken))).scalar_one()
    assert persisted.token_hash == hash_token(raw)
    assert persisted.token_hash not in message
    assert persisted.scopes == ["findings:read"]
    assert persisted.owner_slack_id == "U123ABC"


async def test_handle_create_token_rejects_empty_name(db_session) -> None:
    args = _ns(command="create_token", name="   ", scope=["findings:read"], owner="U1")
    code, message = await admin.handle_create_token(db_session, args)
    assert code == admin.EXIT_INVALID
    assert message.startswith("error:")
    # No token row written.
    rows = (await db_session.execute(select(ApiToken))).scalars().all()
    assert rows == []


# ── handle_list_tokens ───────────────────────────────────────────────────────


async def test_handle_list_tokens_empty(db_session) -> None:
    code, message = await admin.handle_list_tokens(
        db_session,
        _ns(command="list_tokens", include_revoked=False),
    )
    assert code == admin.EXIT_OK
    assert message == "(no tokens)"


async def test_handle_list_tokens_renders_table_and_hides_revoked_by_default(db_session) -> None:
    active = await create_token(
        db_session,
        name="active",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    revoked_issued = await create_token(
        db_session,
        name="revoked",
        scopes=[ApiTokenScope.findings_write],
        owner_slack_id="U2",
    )
    await db_session.commit()

    from api.tokens import revoke_token as _revoke

    await _revoke(db_session, token_id=revoked_issued.record.id, actor_slack_id="U_ADMIN")
    await db_session.commit()

    code, message = await admin.handle_list_tokens(
        db_session,
        _ns(command="list_tokens", include_revoked=False),
    )
    assert code == admin.EXIT_OK
    assert str(active.record.id) in message
    assert str(revoked_issued.record.id) not in message
    assert "U1" in message
    assert "U2" not in message
    # Raw token never appears in listings.
    assert active.raw_token not in message
    assert revoked_issued.raw_token not in message


async def test_handle_list_tokens_include_revoked(db_session) -> None:
    issued = await create_token(
        db_session,
        name="t",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    await db_session.commit()

    from api.tokens import revoke_token as _revoke

    await _revoke(db_session, token_id=issued.record.id, actor_slack_id="U_ADMIN")
    await db_session.commit()

    code, message = await admin.handle_list_tokens(
        db_session,
        _ns(command="list_tokens", include_revoked=True),
    )
    assert code == admin.EXIT_OK
    assert str(issued.record.id) in message


# ── handle_revoke_token ──────────────────────────────────────────────────────


async def test_handle_revoke_token_marks_revoked(db_session) -> None:
    issued = await create_token(
        db_session,
        name="t",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    await db_session.commit()

    args = _ns(command="revoke_token", token_id=str(issued.record.id), actor="U_ADMIN")
    code, message = await admin.handle_revoke_token(db_session, args)
    await db_session.commit()

    assert code == admin.EXIT_OK
    assert "revoked" in message
    refreshed = await db_session.get(ApiToken, issued.record.id)
    assert refreshed is not None
    assert refreshed.revoked_at is not None

    audit = (
        await db_session.execute(
            select(AgentActionLog).where(AgentActionLog.tool_name == "revoke_token"),
        )
    ).scalar_one()
    assert audit.tool_inputs is not None
    assert audit.tool_inputs["actor_slack_id"] == "U_ADMIN"


async def test_handle_revoke_token_unknown_returns_not_found(db_session) -> None:
    args = _ns(command="revoke_token", token_id=str(uuid.uuid4()), actor="U_ADMIN")
    code, message = await admin.handle_revoke_token(db_session, args)
    assert code == admin.EXIT_NOT_FOUND
    assert message.startswith("error:")


async def test_handle_revoke_token_rejects_non_uuid(db_session) -> None:
    args = _ns(command="revoke_token", token_id="not-a-uuid", actor="U_ADMIN")
    code, message = await admin.handle_revoke_token(db_session, args)
    assert code == admin.EXIT_INVALID
    assert "not a valid UUID" in message


async def test_handle_revoke_token_rejects_oversize_actor(db_session) -> None:
    issued = await create_token(
        db_session,
        name="t",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    await db_session.commit()
    args = _ns(
        command="revoke_token",
        token_id=str(issued.record.id),
        actor="x" * (admin._ACTOR_MAX_CHARS + 1),
    )
    code, message = await admin.handle_revoke_token(db_session, args)
    assert code == admin.EXIT_INVALID
    assert str(admin._ACTOR_MAX_CHARS) in message
    refreshed = await db_session.get(ApiToken, issued.record.id)
    assert refreshed is not None
    assert refreshed.revoked_at is None


async def test_handle_revoke_token_defaults_actor_to_cli_user(db_session, monkeypatch) -> None:
    monkeypatch.setattr(admin, "_default_actor", lambda: "cli:tester")
    issued = await create_token(
        db_session,
        name="t",
        scopes=[ApiTokenScope.findings_read],
        owner_slack_id="U1",
    )
    await db_session.commit()

    args = _ns(command="revoke_token", token_id=str(issued.record.id), actor=None)
    code, _ = await admin.handle_revoke_token(db_session, args)
    await db_session.commit()
    assert code == admin.EXIT_OK

    audit = (
        await db_session.execute(
            select(AgentActionLog).where(AgentActionLog.tool_name == "revoke_token"),
        )
    ).scalar_one()
    assert audit.tool_inputs is not None
    assert audit.tool_inputs["actor_slack_id"] == "cli:tester"


# ── main() end-to-end ────────────────────────────────────────────────────────


@pytest.fixture
async def admin_db_url(tmp_path: Path, monkeypatch) -> str:
    db_path = tmp_path / "admin.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    return url


def test_main_create_then_list_then_revoke_roundtrip(
    admin_db_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = admin.main(
        ["create_token", "--name", "ci", "--scope", "findings:read", "--owner", "U1"],
    )
    assert code == admin.EXIT_OK
    out = capsys.readouterr().out
    assert "Token created" in out
    token_id = next(line.split("id:", 1)[1].strip() for line in out.splitlines() if "id:" in line)
    raw_token = next(
        line.split("token:", 1)[1].strip() for line in out.splitlines() if line.strip().startswith("token:")
    )
    assert len(raw_token) >= 32

    code = admin.main(["list_tokens"])
    assert code == admin.EXIT_OK
    out = capsys.readouterr().out
    assert token_id in out
    assert raw_token not in out  # never re-printed

    code = admin.main(["revoke_token", token_id])
    assert code == admin.EXIT_OK
    out = capsys.readouterr().out
    assert "revoked" in out
    assert token_id in out

    code = admin.main(["list_tokens"])
    assert code == admin.EXIT_OK
    out = capsys.readouterr().out
    assert token_id not in out  # hidden by default after revoke

    code = admin.main(["list_tokens", "--include-revoked"])
    assert code == admin.EXIT_OK
    out = capsys.readouterr().out
    assert token_id in out


def test_main_revoke_unknown_id_exits_nonzero(
    admin_db_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bogus = str(uuid.uuid4())
    code = admin.main(["revoke_token", bogus])
    assert code == admin.EXIT_NOT_FOUND
    err = capsys.readouterr().err
    assert "error" in err
    assert bogus in err


def test_main_revoke_invalid_uuid_exits_nonzero(
    admin_db_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = admin.main(["revoke_token", "definitely-not-a-uuid"])
    assert code == admin.EXIT_INVALID
    err = capsys.readouterr().err
    assert "not a valid UUID" in err


def test_default_actor_format() -> None:
    actor = admin._default_actor()
    assert actor.startswith("cli:")
    assert len(actor) > len("cli:")


def test_default_actor_falls_back_when_getpass_fails(monkeypatch) -> None:
    def _boom() -> str:
        raise OSError("no user")

    monkeypatch.setattr(admin.getpass, "getuser", _boom)
    assert admin._default_actor() == "cli:unknown"

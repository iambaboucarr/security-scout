# SPDX-License-Identifier: Apache-2.0
"""Host-only admin CLI for API token management.

Invoked by an operator on the host as ``python -m src.admin``. Subcommands:

* ``create_token --name --scope --owner`` — issue a token; the raw token is
  printed exactly once on stdout. The plaintext value is never persisted or
  re-derivable from the database, so capture it before the process exits.
* ``list_tokens [--include-revoked]`` — tabulate token metadata. The plaintext
  is never available here.
* ``revoke_token <id> [--actor <id>]`` — mark a token revoked; idempotent.

There is intentionally no HTTP endpoint and no Slack self-service. Token
issuance is a privileged action that runs on the API host with direct database
access. See ``documentation/admin-cli-followups.md`` for the planned path to
exposing this as an authenticated HTTP API + Slack workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog

# Ensure ``src/`` is on ``sys.path`` so the top-level imports below resolve
# whether the entry point is invoked as ``python -m src.admin`` from the repo
# root or with ``PYTHONPATH=src`` (the Makefile convention).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:  # pragma: no cover - depends on launch context
    sys.path.insert(0, str(_SRC_DIR))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker  # noqa: E402

from api.tokens import IssuedToken, create_token, list_tokens, revoke_token  # noqa: E402
from config import Settings, configure_logging  # noqa: E402
from db import create_engine, create_session_factory, session_scope  # noqa: E402
from models import ApiToken, ApiTokenScope  # noqa: E402

_LOG = structlog.get_logger(__name__)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INVALID = 3
EXIT_NOT_FOUND = 4

_ACTOR_MAX_CHARS = 64

_TABLE_HEADERS: tuple[str, ...] = (
    "id",
    "name",
    "scopes",
    "owner",
    "created",
    "last_used",
    "revoked",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.admin",
        description=(
            "Administer Security Scout API tokens. Host-local; do not expose over a network. "
            "Raw tokens are shown exactly once on creation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create = sub.add_parser(
        "create_token",
        help="Issue a new API token. Prints the raw token once on stdout.",
    )
    create.add_argument("--name", required=True, help="Human-readable label (audit trail only).")
    create.add_argument(
        "--scope",
        action="append",
        required=True,
        choices=[s.value for s in ApiTokenScope],
        help="Scope to grant. Repeat the flag for multiple scopes.",
    )
    create.add_argument(
        "--owner",
        required=True,
        help="Owner Slack user ID (e.g. U123ABC) — the human accountable for the token.",
    )

    listing = sub.add_parser("list_tokens", help="List active tokens (metadata only).")
    listing.add_argument(
        "--include-revoked",
        action="store_true",
        help="Also include tokens that have already been revoked.",
    )

    revoke = sub.add_parser("revoke_token", help="Revoke a token by UUID. Idempotent.")
    revoke.add_argument("token_id", help="Token UUID as printed by create_token / list_tokens.")
    revoke.add_argument(
        "--actor",
        default=None,
        help="Recorded actor for the audit row. Defaults to ``cli:<host-user>``.",
    )

    return parser


def _default_actor() -> str:
    try:
        user = getpass.getuser()
    except OSError:
        user = "unknown"
    return f"cli:{user or 'unknown'}"


def _format_dt(value: object) -> str:
    if value is None:
        return "-"
    iso = getattr(value, "isoformat", None)
    return iso(timespec="seconds") if callable(iso) else str(value)


def _row(record: ApiToken) -> tuple[str, ...]:
    return (
        str(record.id),
        record.name,
        ",".join(record.scopes),
        record.owner_slack_id,
        _format_dt(record.created_at),
        _format_dt(record.last_used_at),
        _format_dt(record.revoked_at),
    )


def _render_table(rows: list[tuple[str, ...]]) -> str:
    table = [_TABLE_HEADERS, *rows]
    widths = [max(len(cell) for cell in column) for column in zip(*table, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(fmt.format(*row) for row in table)


async def handle_create_token(session: AsyncSession, args: argparse.Namespace) -> tuple[int, str]:
    try:
        issued: IssuedToken = await create_token(
            session,
            name=args.name,
            scopes=list(args.scope),
            owner_slack_id=args.owner,
        )
    except ValueError as exc:
        return EXIT_INVALID, f"error: {exc}"

    _LOG.info(
        "admin_cli_create_token",
        metric_name="admin_cli_create_token_total",
        actor=_default_actor(),
        token_id=str(issued.record.id),
        name=issued.record.name,
        scopes=list(issued.record.scopes),
        owner_slack_id=issued.record.owner_slack_id,
    )

    lines = [
        "Token created. Copy the value below now — it cannot be retrieved later.",
        f"  id:     {issued.record.id}",
        f"  name:   {issued.record.name}",
        f"  scopes: {','.join(issued.record.scopes)}",
        f"  owner:  {issued.record.owner_slack_id}",
        f"  token:  {issued.raw_token}",
    ]
    return EXIT_OK, "\n".join(lines)


async def handle_list_tokens(session: AsyncSession, args: argparse.Namespace) -> tuple[int, str]:
    records = await list_tokens(session, include_revoked=bool(args.include_revoked))
    _LOG.info(
        "admin_cli_list_tokens",
        metric_name="admin_cli_list_tokens_total",
        actor=_default_actor(),
        count=len(records),
        include_revoked=bool(args.include_revoked),
    )
    if not records:
        return EXIT_OK, "(no tokens)"
    return EXIT_OK, _render_table([_row(record) for record in records])


async def handle_revoke_token(session: AsyncSession, args: argparse.Namespace) -> tuple[int, str]:
    try:
        token_id = uuid.UUID(str(args.token_id))
    except ValueError:
        return EXIT_INVALID, f"error: {args.token_id!r} is not a valid UUID"

    actor = (args.actor or "").strip() or _default_actor()
    if len(actor) > _ACTOR_MAX_CHARS:
        return (
            EXIT_INVALID,
            f"error: --actor exceeds {_ACTOR_MAX_CHARS} characters",
        )
    record = await revoke_token(session, token_id=token_id, actor_slack_id=actor)
    if record is None:
        return EXIT_NOT_FOUND, f"error: no token with id {token_id}"

    _LOG.info(
        "admin_cli_revoke_token",
        metric_name="admin_cli_revoke_token_total",
        actor=actor,
        token_id=str(record.id),
    )
    return EXIT_OK, f"revoked: {record.id} (revoked_at={_format_dt(record.revoked_at)})"


_HANDLERS: dict[str, Callable[[AsyncSession, argparse.Namespace], Awaitable[tuple[int, str]]]] = {
    "create_token": handle_create_token,
    "list_tokens": handle_list_tokens,
    "revoke_token": handle_revoke_token,
}


async def _dispatch(
    factory: async_sessionmaker[AsyncSession],
    args: argparse.Namespace,
) -> tuple[int, str]:
    handler = _HANDLERS[args.command]
    async with session_scope(factory) as session:
        return await handler(session, args)


async def _async_main(args: argparse.Namespace) -> tuple[int, str]:
    settings = Settings()
    configure_logging(settings.log_level)
    engine = create_engine(settings.database_url)
    try:
        factory = create_session_factory(engine)
        return await _dispatch(factory, args)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    code, message = asyncio.run(_async_main(args))
    stream = sys.stdout if code == EXIT_OK else sys.stderr
    print(message, file=stream)
    return code


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())


__all__ = [
    "EXIT_INVALID",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_USAGE",
    "handle_create_token",
    "handle_list_tokens",
    "handle_revoke_token",
    "main",
]

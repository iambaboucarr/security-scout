# SPDX-License-Identifier: Apache-2.0
"""API token issuance, hashing, and revocation.

Raw tokens are returned to the caller exactly once (on creation) and are never
persisted — only their SHA-256 hex digest is stored. Every mutation appends an
``AgentActionLog`` row so token issuance and revocation are auditable.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from models import AgentActionLog, ApiToken, ApiTokenScope

_LOG = structlog.get_logger(__name__)

_AGENT_NAME = "api_token"
_TOKEN_BYTES = 32


@dataclass(frozen=True)
class IssuedToken:
    """Returned only once at creation. ``raw_token`` is never persisted."""

    raw_token: str
    record: ApiToken


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _normalise_scopes(scopes: list[str] | list[ApiTokenScope]) -> list[str]:
    if not scopes:
        raise ValueError("scopes must be non-empty")
    seen: set[str] = set()
    normalised: list[str] = []
    for item in scopes:
        value = item.value if isinstance(item, ApiTokenScope) else item
        if value not in {s.value for s in ApiTokenScope}:
            raise ValueError(f"unknown scope: {value!r}")
        if value in seen:
            continue
        seen.add(value)
        normalised.append(value)
    return normalised


async def create_token(
    session: AsyncSession,
    *,
    name: str,
    scopes: list[str] | list[ApiTokenScope],
    owner_slack_id: str,
) -> IssuedToken:
    if not name.strip():
        raise ValueError("name must not be empty")
    if not owner_slack_id.strip():
        raise ValueError("owner_slack_id must not be empty")

    canonical_scopes = _normalise_scopes(scopes)
    raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
    record = ApiToken(
        id=uuid.uuid4(),
        name=name.strip(),
        token_hash=hash_token(raw_token),
        scopes=canonical_scopes,
        owner_slack_id=owner_slack_id.strip(),
    )
    session.add(record)
    await session.flush()

    session.add(
        AgentActionLog(
            agent=_AGENT_NAME,
            tool_name="create_token",
            tool_inputs={
                "token_id": str(record.id),
                "name": record.name,
                "scopes": canonical_scopes,
                "owner_slack_id": record.owner_slack_id,
            },
            tool_output="created",
            workflow_run_id=None,
        )
    )
    await session.flush()

    _LOG.info(
        "api_token_created",
        metric_name="api_token_created_total",
        token_id=str(record.id),
        name=record.name,
        scopes=canonical_scopes,
        owner_slack_id=record.owner_slack_id,
    )
    return IssuedToken(raw_token=raw_token, record=record)


async def revoke_token(
    session: AsyncSession,
    *,
    token_id: uuid.UUID,
    actor_slack_id: str,
) -> ApiToken | None:
    record = await session.get(ApiToken, token_id)
    if record is None:
        return None
    if record.revoked_at is not None:
        return record

    record.revoked_at = datetime.now(UTC)
    session.add(
        AgentActionLog(
            agent=_AGENT_NAME,
            tool_name="revoke_token",
            tool_inputs={
                "token_id": str(record.id),
                "actor_slack_id": actor_slack_id,
            },
            tool_output="revoked",
            workflow_run_id=None,
        )
    )
    await session.flush()

    _LOG.info(
        "api_token_revoked",
        metric_name="api_token_revoked_total",
        token_id=str(record.id),
        actor_slack_id=actor_slack_id,
    )
    return record


__all__ = [
    "IssuedToken",
    "create_token",
    "hash_token",
    "revoke_token",
]

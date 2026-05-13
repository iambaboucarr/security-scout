# SPDX-License-Identifier: Apache-2.0
"""Bearer token authentication for the HTTP API.

A FastAPI dependency factory: ``require_scope(ApiTokenScope.findings_read)``
returns a coroutine dep that resolves the ``Authorization: Bearer <token>``
header, looks up the SHA-256 hash, enforces scope, refreshes ``last_used_at``,
and appends an ``AgentActionLog`` row. Failures map to 401 (no/unknown/revoked
token) or 403 (scope mismatch) — never leaking which exact reason caused the
401.

The raw token is never logged or persisted; only the database ``token_id`` is
referenced in logs and audit rows.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import structlog
from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.tokens import hash_token
from models import AgentActionLog, ApiToken, ApiTokenScope

_LOG = structlog.get_logger(__name__)

_AGENT_NAME = "api_auth"
_BEARER_PREFIX = "Bearer "
_INVALID_TOKEN_DETAIL = "Invalid or revoked token"  # noqa: S105 — HTTP error string, not a secret
_INSUFFICIENT_SCOPE_DETAIL = "Token lacks required scope"


def _extract_bearer(request: Request) -> str:
    header = request.headers.get("authorization")
    if header is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_TOKEN_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not header.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_TOKEN_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = header[len(_BEARER_PREFIX) :].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_TOKEN_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _resolve_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] | None = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session factory not initialised",
        )
    return factory


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=_INVALID_TOKEN_DETAIL,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=_INSUFFICIENT_SCOPE_DETAIL,
    )


def _validated_api_scopes(raw: object) -> list[str]:
    """Drop unknown or duplicate scope strings so JSON cannot smuggle invalid labels."""
    if not isinstance(raw, list):
        return []
    allowed = frozenset(s.value for s in ApiTokenScope)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str) and item in allowed and item not in seen:
            seen.add(item)
            out.append(item)
    return out


async def _authenticate(request: Request, required_scope: ApiTokenScope) -> ApiToken:
    raw_token = _extract_bearer(request)
    token_hash = hash_token(raw_token)
    session_factory = _resolve_session_factory(request)

    # Manage the session manually rather than via ``session_scope`` so we can
    # commit the audit row on the scope-mismatch path before raising 403 — an
    # HTTPException inside ``session_scope`` would trigger rollback and erase
    # the audit trail of the rejected request.
    session: AsyncSession = session_factory()
    try:
        record = await session.scalar(select(ApiToken).where(ApiToken.token_hash == token_hash))

        if record is None:
            _LOG.info(
                "api_auth_rejected",
                metric_name="api_auth_rejected_total",
                reason="token_not_found",
                path=request.url.path,
            )
            raise _unauthorized()

        if record.revoked_at is not None:
            _LOG.info(
                "api_auth_rejected",
                metric_name="api_auth_rejected_total",
                reason="token_revoked",
                token_id=str(record.id),
                path=request.url.path,
            )
            raise _unauthorized()

        validated_scopes = _validated_api_scopes(record.scopes)
        if required_scope.value not in validated_scopes:
            _LOG.info(
                "api_auth_rejected",
                metric_name="api_auth_rejected_total",
                reason="scope_mismatch",
                token_id=str(record.id),
                required_scope=required_scope.value,
                path=request.url.path,
            )
            session.add(
                AgentActionLog(
                    agent=_AGENT_NAME,
                    tool_name="authenticate",
                    tool_inputs={
                        "token_id": str(record.id),
                        "required_scope": required_scope.value,
                        "path": request.url.path,
                        "outcome": "scope_mismatch",
                    },
                    tool_output="forbidden",
                    workflow_run_id=None,
                )
            )
            await session.commit()
            raise _forbidden()

        record.last_used_at = datetime.now(UTC)
        session.add(
            AgentActionLog(
                agent=_AGENT_NAME,
                tool_name="authenticate",
                tool_inputs={
                    "token_id": str(record.id),
                    "required_scope": required_scope.value,
                    "path": request.url.path,
                    "outcome": "ok",
                },
                tool_output="ok",
                workflow_run_id=None,
            )
        )
        await session.commit()

        _LOG.debug(
            "api_auth_ok",
            token_id=str(record.id),
            required_scope=required_scope.value,
            path=request.url.path,
        )
        return record
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def require_scope(scope: ApiTokenScope) -> Callable[[Request], Awaitable[ApiToken]]:
    if not isinstance(scope, ApiTokenScope):
        raise TypeError(f"scope must be ApiTokenScope, got {type(scope).__name__}")

    async def _dep(request: Request) -> ApiToken:
        return await _authenticate(request, scope)

    return _dep


__all__ = ["require_scope"]

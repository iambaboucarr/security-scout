# SPDX-License-Identifier: Apache-2.0
"""Per-token sliding-window rate limit for authenticated HTTP API requests.

Provides a FastAPI dependency that wraps :class:`SlidingWindowRateLimiter`
and counts requests against the authenticated token's ``id``. The limiter
operation name is ``api_token`` and the scope is the token UUID, so circuit
breaches and rate-limit counters never collide with other rate-limited
operations (Slack posts, advisory polls, …).

Quota exhaustion maps to ``429 Too Many Requests`` with a stable JSON body.
When the limiter's circuit breaker is open (sustained breaches for that
token's scope), responses use ``503 Service Unavailable`` with a distinct
detail string so clients and operators do not confuse backend back-pressure
with per-minute quota. When the shared Redis pool is missing the dependency
falls open so the API stays available even if the rate-limit backend is
degraded — matching the limiter's own fail-open posture.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog
from fastapi import HTTPException, Request, status

from models import ApiToken
from tools.rate_limiter import (
    RateLimiterCircuitOpen,
    RateLimitExceeded,
    RedisLike,
    SlidingWindowRateLimiter,
)

_LOG = structlog.get_logger(__name__)

API_TOKEN_RATE_LIMIT = 60
API_TOKEN_RATE_WINDOW_SECONDS = 60
API_TOKEN_OPERATION = "api_token"  # noqa: S105 — rate-limiter scope label, not a credential

# Distinct from the 429 quota message — circuit open means the limiter paused this scope.
API_RATE_LIMIT_CIRCUIT_UNAVAILABLE_DETAIL = "Rate limiting is temporarily unavailable; retry later."


def _resolve_redis(request: Request) -> RedisLike | None:
    redis = getattr(request.app.state, "redis_pool", None)
    if redis is None:
        return None
    if not isinstance(redis, RedisLike):
        return None
    return redis


async def enforce_token_rate_limit(request: Request, token: ApiToken) -> None:
    """Increment the per-token sliding window counter or raise ``HTTPException``.

    Separated from the FastAPI dependency wrapper so it can be invoked
    inline by callers that already hold a resolved :class:`ApiToken`
    (e.g. composite dependencies in route handlers).
    """
    redis = _resolve_redis(request)
    if redis is None:
        _LOG.debug(
            "api_rate_limit_skipped_no_redis",
            token_id=str(token.id),
            path=request.url.path,
        )
        return

    limiter = SlidingWindowRateLimiter(redis)
    try:
        await limiter.check_and_increment(
            operation=API_TOKEN_OPERATION,
            scope=str(token.id),
            limit=API_TOKEN_RATE_LIMIT,
            window_seconds=API_TOKEN_RATE_WINDOW_SECONDS,
        )
    except RateLimitExceeded as exc:
        _LOG.info(
            "api_rate_limited",
            metric_name="api_rate_limited_total",
            token_id=str(token.id),
            path=request.url.path,
            limit=exc.limit,
            window_seconds=exc.window_seconds,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(exc.window_seconds)},
        ) from exc
    except RateLimiterCircuitOpen as exc:
        _LOG.warning(
            "api_rate_limit_circuit_open",
            metric_name="api_rate_limit_circuit_open_total",
            token_id=str(token.id),
            path=request.url.path,
            remaining_seconds=exc.remaining_seconds,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=API_RATE_LIMIT_CIRCUIT_UNAVAILABLE_DETAIL,
            headers={"Retry-After": str(max(1, exc.remaining_seconds))},
        ) from exc


def require_token_rate_limit(
    resolver: Callable[[Request], Awaitable[ApiToken]],
) -> Callable[[Request], Awaitable[ApiToken]]:
    """Wrap a bearer-token resolver so each authenticated call increments the rate limit.

    The returned dependency resolves the token first (preserving 401/403
    semantics) and only then consults the limiter, so unauthenticated
    traffic never burns rate-limit budget against a real token.
    """

    async def _dep(request: Request) -> ApiToken:
        token = await resolver(request)
        await enforce_token_rate_limit(request, token)
        return token

    return _dep


__all__ = [
    "API_RATE_LIMIT_CIRCUIT_UNAVAILABLE_DETAIL",
    "API_TOKEN_OPERATION",
    "API_TOKEN_RATE_LIMIT",
    "API_TOKEN_RATE_WINDOW_SECONDS",
    "enforce_token_rate_limit",
    "require_token_rate_limit",
]

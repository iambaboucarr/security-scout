# SPDX-License-Identifier: Apache-2.0
"""Shared Redis sliding window rate limiter with per-scope circuit breaker.

Uses Redis sorted sets for sliding window counting and a Lua script for
atomic check-and-increment.  A circuit breaker tracks repeated rate-limit
breaches per scope; when a scope exceeds CIRCUIT_BREACH_THRESHOLD breaches
within CIRCUIT_BREACH_WINDOW_SEC, all operations for that scope are blocked
for CIRCUIT_PAUSE_SEC.

Fail-open: if Redis is unavailable, operations are allowed and a warning is
logged rather than blocking the pipeline.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import structlog

from exceptions import SecurityScoutError


@runtime_checkable
class RedisLike(Protocol):
    """Minimal Redis interface used by the rate limiter.

    Keeps the rate limiter testable with in-memory doubles while providing
    type safety beyond ``Any``.
    """

    async def eval(self, script: str, numkeys: int, *args: str) -> int: ...
    async def ttl(self, key: str) -> int: ...
    async def setex(self, key: str, seconds: int, value: str) -> None: ...
    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool | None: ...
    async def delete(self, *keys: str) -> int: ...
    def pipeline(self, transaction: bool = False) -> Any: ...


_LOG = structlog.get_logger(__name__)

_SLIDING_WINDOW_LUA = """\
local key = KEYS[1]
local cutoff = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now_ts = tonumber(ARGV[3])
local member = ARGV[4]
local ttl = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now_ts, member)
    redis.call('EXPIRE', key, ttl)
    return 0
else
    return 1
end
"""


class RateLimitExceeded(SecurityScoutError):
    """Operation denied because the sliding-window rate limit was exceeded."""

    def __init__(
        self,
        message: str | None = None,
        *,
        operation: str,
        scope: str,
        limit: int,
        window_seconds: int,
        should_alert: bool = False,
        circuit_opened: bool = False,
    ) -> None:
        super().__init__(message, is_transient=True, is_resource_error=False)
        self.operation = operation
        self.scope = scope
        self.limit = limit
        self.window_seconds = window_seconds
        self.should_alert = should_alert
        self.circuit_opened = circuit_opened


class RateLimiterCircuitOpen(SecurityScoutError):
    """All operations for this scope are paused due to sustained breaches."""

    def __init__(
        self,
        message: str | None = None,
        *,
        scope: str,
        remaining_seconds: int,
    ) -> None:
        super().__init__(message, is_transient=True, is_resource_error=False)
        self.scope = scope
        self.remaining_seconds = remaining_seconds


class SlidingWindowRateLimiter:
    """Redis-backed sliding window rate limiter with per-scope circuit breaker.

    *circuit_scope* in :meth:`check_and_increment` controls which scope the
    circuit breaker tracks.  For example Slack findings are rate-limited per
    channel (``scope``), but the circuit breaker fires per repo
    (``circuit_scope``), pausing all write operations for that repo.
    """

    CIRCUIT_BREACH_THRESHOLD = 3
    CIRCUIT_BREACH_WINDOW_SEC = 3600
    CIRCUIT_PAUSE_SEC = 3600
    ALERT_DEDUP_SEC = 3600

    _PREFIX = "rl"

    def __init__(
        self,
        redis: RedisLike,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._redis = redis
        self._now = now_fn or time.time

    # -- key helpers ----------------------------------------------------------

    def _rate_key(self, operation: str, scope: str) -> str:
        return f"{self._PREFIX}:{operation}:{scope}"

    def _breach_key(self, scope: str) -> str:
        return f"{self._PREFIX}:breach:{scope}"

    def _circuit_key(self, scope: str) -> str:
        return f"{self._PREFIX}:circuit:{scope}"

    def _alert_key(self, operation: str, scope: str) -> str:
        return f"{self._PREFIX}:alert:{operation}:{scope}"

    # -- internal helpers -----------------------------------------------------

    async def _is_circuit_open(self, scope: str) -> tuple[bool, int]:
        ttl: int = await self._redis.ttl(self._circuit_key(scope))
        if ttl > 0:
            return True, ttl
        return False, 0

    async def _record_breach(self, scope: str) -> bool:
        """Record a rate-limit breach. Returns ``True`` if the circuit opened."""
        now = self._now()
        breach_key = self._breach_key(scope)
        cutoff = now - self.CIRCUIT_BREACH_WINDOW_SEC

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(breach_key, "-inf", cutoff)
        pipe.zadd(breach_key, {uuid.uuid4().hex: now})
        pipe.zcard(breach_key)
        pipe.expire(breach_key, self.CIRCUIT_BREACH_WINDOW_SEC + 120)
        results = await pipe.execute()

        count: int = results[2]
        if count >= self.CIRCUIT_BREACH_THRESHOLD:
            await self._redis.setex(
                self._circuit_key(scope),
                self.CIRCUIT_PAUSE_SEC,
                "1",
            )
            await self._redis.delete(breach_key)
            _LOG.warning(
                "rate_limiter_circuit_opened",
                metric_name="rate_limit_circuit_opened_total",
                scope=scope,
                breach_count=count,
                pause_seconds=self.CIRCUIT_PAUSE_SEC,
            )
            return True
        return False

    async def _should_alert(self, operation: str, scope: str) -> bool:
        """Return ``True`` once per ALERT_DEDUP_SEC per operation+scope."""
        alert_key = self._alert_key(operation, scope)
        was_set = await self._redis.set(
            alert_key,
            "1",
            nx=True,
            ex=self.ALERT_DEDUP_SEC,
        )
        return was_set is not None

    # -- public API -----------------------------------------------------------

    async def check_and_increment(
        self,
        *,
        operation: str,
        scope: str,
        limit: int,
        window_seconds: int,
        circuit_scope: str | None = None,
    ) -> None:
        """Check the sliding-window counter and increment if under the limit.

        Parameters
        ----------
        operation:
            Logical name (e.g. ``"slack_finding"``, ``"post_pr_comment"``).
        scope:
            Rate-counting key (e.g. a Slack channel or repo name).
        limit:
            Maximum allowed operations within *window_seconds*.
        window_seconds:
            Width of the sliding window.
        circuit_scope:
            Scope for circuit-breaker breach tracking.  Defaults to *scope*.

        Raises
        ------
        ValueError
            If *limit* < 0 or *window_seconds* <= 0.
        RateLimiterCircuitOpen
            If the circuit breaker is open for *circuit_scope*.
        RateLimitExceeded
            If the counter already reached *limit*.
        """
        if limit < 0:
            msg = "limit must be non-negative"
            raise ValueError(msg)
        if window_seconds <= 0:
            msg = "window_seconds must be positive"
            raise ValueError(msg)

        cs = circuit_scope if circuit_scope is not None else scope

        try:
            is_open, remaining = await self._is_circuit_open(cs)
            if is_open:
                msg = f"rate limiter circuit open for {cs!r}, {remaining}s remaining"
                _LOG.warning(
                    "rate_limiter_circuit_blocked",
                    metric_name="rate_limit_circuit_blocked_total",
                    scope=cs,
                    remaining_seconds=remaining,
                    operation=operation,
                )
                raise RateLimiterCircuitOpen(
                    msg,
                    scope=cs,
                    remaining_seconds=remaining,
                )

            now = self._now()
            cutoff = now - window_seconds
            rate_key = self._rate_key(operation, scope)
            member = uuid.uuid4().hex
            ttl = window_seconds + 120

            denied: int = await self._redis.eval(
                _SLIDING_WINDOW_LUA,
                1,
                rate_key,
                str(cutoff),
                str(limit),
                str(now),
                member,
                str(ttl),
            )

            if denied:
                circuit_opened = await self._record_breach(cs)
                should_alert = await self._should_alert(operation, scope)

                _LOG.warning(
                    "rate_limit_exceeded",
                    metric_name="rate_limit_exceeded_total",
                    operation=operation,
                    scope=scope,
                    limit=limit,
                    window_seconds=window_seconds,
                    circuit_scope=cs,
                    circuit_opened=circuit_opened,
                    should_alert=should_alert,
                )
                msg = f"rate limit exceeded: {operation} {limit}/{window_seconds}s for {scope!r}"
                raise RateLimitExceeded(
                    msg,
                    operation=operation,
                    scope=scope,
                    limit=limit,
                    window_seconds=window_seconds,
                    should_alert=should_alert,
                    circuit_opened=circuit_opened,
                )

        except RateLimitExceeded, RateLimiterCircuitOpen:
            raise
        except Exception:
            _LOG.warning(
                "rate_limiter_redis_error",
                operation=operation,
                scope=scope,
                exc_info=True,
            )


__all__ = [
    "RateLimitExceeded",
    "RateLimiterCircuitOpen",
    "RedisLike",
    "SlidingWindowRateLimiter",
]

# SPDX-License-Identifier: Apache-2.0
"""Tests for tools.rate_limiter — sliding window rate limiter with circuit breaker."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from tools.rate_limiter import (
    RateLimiterCircuitOpen,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
)

# ---------------------------------------------------------------------------
# Shared clock + FakeRedis
# ---------------------------------------------------------------------------


class _Clock:
    """Mutable monotonic clock for deterministic time control in tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakePipeline:
    """Minimal pipeline mock that executes commands against *_FakeRedis*."""

    def __init__(self, redis: _FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...]]] = []

    def zremrangebyscore(
        self,
        key: str,
        min_score: str | float,
        max_score: str | float,
    ) -> _FakePipeline:
        self._commands.append(("zremrangebyscore", (key, min_score, max_score)))
        return self

    def zadd(self, key: str, mapping: dict[str, float]) -> _FakePipeline:
        self._commands.append(("zadd", (key, mapping)))
        return self

    def zcard(self, key: str) -> _FakePipeline:
        self._commands.append(("zcard", (key,)))
        return self

    def expire(self, key: str, seconds: int) -> _FakePipeline:
        self._commands.append(("expire", (key, seconds)))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for cmd, args in self._commands:
            if cmd == "zremrangebyscore":
                key, _min_s, max_s = args
                max_val = float(max_s) if max_s != "+inf" else math.inf
                zset = self._redis.sorted_sets.get(key, {})
                before = len(zset)
                cleaned = {k: v for k, v in zset.items() if v > max_val}
                self._redis.sorted_sets[key] = cleaned
                results.append(before - len(cleaned))
            elif cmd == "zadd":
                key, mapping = args
                zset = self._redis.sorted_sets.setdefault(key, {})
                added = sum(1 for m in mapping if m not in zset)
                zset.update(mapping)
                results.append(added)
            elif cmd == "zcard":
                (key,) = args
                results.append(len(self._redis.sorted_sets.get(key, {})))
            elif cmd == "expire":
                key, seconds = args
                self._redis.expiry[key] = self._redis.now_fn() + seconds
                results.append(True)
        self._commands.clear()
        return results


class _FakeRedis:
    """In-memory Redis stand-in that implements the subset used by *SlidingWindowRateLimiter*."""

    def __init__(self, now_fn: Callable[[], float]) -> None:
        self.now_fn = now_fn
        self.sorted_sets: dict[str, dict[str, float]] = {}
        self.strings: dict[str, str] = {}
        self.expiry: dict[str, float] = {}
        self.raise_on_next: Exception | None = None

    def _clean(self, key: str) -> None:
        if key in self.expiry and self.now_fn() >= self.expiry[key]:
            self.sorted_sets.pop(key, None)
            self.strings.pop(key, None)
            self.expiry.pop(key, None)

    async def eval(
        self,
        script: str,
        numkeys: int,
        *args: str,
    ) -> int:
        if self.raise_on_next is not None:
            err = self.raise_on_next
            self.raise_on_next = None
            raise err
        key = args[0]
        cutoff = float(args[1])
        limit = int(args[2])
        now_ts = float(args[3])
        member = args[4]
        ttl = int(args[5])

        self._clean(key)
        zset = self.sorted_sets.setdefault(key, {})
        zset_clean = {k: v for k, v in zset.items() if v > cutoff}
        self.sorted_sets[key] = zset_clean

        if len(zset_clean) < limit:
            zset_clean[member] = now_ts
            self.expiry[key] = self.now_fn() + ttl
            return 0
        return 1

    async def ttl(self, key: str) -> int:
        if self.raise_on_next is not None:
            err = self.raise_on_next
            self.raise_on_next = None
            raise err
        self._clean(key)
        if key not in self.expiry:
            return -2
        remaining = self.expiry[key] - self.now_fn()
        if remaining <= 0:
            self._clean(key)
            return -2
        return int(remaining)

    async def setex(self, key: str, seconds: int, value: str) -> None:
        self.strings[key] = value
        self.expiry[key] = self.now_fn() + seconds

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool | None:
        self._clean(key)
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        if ex is not None:
            self.expiry[key] = self.now_fn() + ex
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            removed = False
            if key in self.sorted_sets:
                del self.sorted_sets[key]
                removed = True
            if key in self.strings:
                del self.strings[key]
                removed = True
            self.expiry.pop(key, None)
            if removed:
                count += 1
        return count

    def pipeline(self, transaction: bool = False) -> _FakePipeline:
        return _FakePipeline(self)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def fake_redis(clock: _Clock) -> _FakeRedis:
    return _FakeRedis(clock)


@pytest.fixture
def limiter(fake_redis: _FakeRedis, clock: _Clock) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(fake_redis, now_fn=clock)


# ---------------------------------------------------------------------------
# Basic sliding-window behaviour
# ---------------------------------------------------------------------------


async def test_allows_operations_under_limit(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for _ in range(5):
        await limiter.check_and_increment(
            operation="slack_finding",
            scope="#channel",
            limit=5,
            window_seconds=3600,
        )


async def test_blocks_at_limit(limiter: SlidingWindowRateLimiter) -> None:
    for _ in range(3):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=3,
            window_seconds=60,
        )
    with pytest.raises(RateLimitExceeded, match="rate limit exceeded"):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=3,
            window_seconds=60,
        )


async def test_sliding_window_expires_old_entries(
    limiter: SlidingWindowRateLimiter,
    clock: _Clock,
) -> None:
    for _ in range(3):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=3,
            window_seconds=60,
        )

    clock.advance(61)

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=3,
        window_seconds=60,
    )


async def test_zero_limit_always_denies(
    limiter: SlidingWindowRateLimiter,
) -> None:
    with pytest.raises(RateLimitExceeded):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=0,
            window_seconds=60,
        )


async def test_separate_scopes_are_independent(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for _ in range(3):
        await limiter.check_and_increment(
            operation="op",
            scope="scope-a",
            limit=3,
            window_seconds=60,
        )

    await limiter.check_and_increment(
        operation="op",
        scope="scope-b",
        limit=3,
        window_seconds=60,
    )


async def test_separate_operations_are_independent(
    limiter: SlidingWindowRateLimiter,
) -> None:
    for _ in range(3):
        await limiter.check_and_increment(
            operation="op-a",
            scope="s",
            limit=3,
            window_seconds=60,
        )

    await limiter.check_and_increment(
        operation="op-b",
        scope="s",
        limit=3,
        window_seconds=60,
    )


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


async def test_circuit_opens_after_threshold_breaches(
    limiter: SlidingWindowRateLimiter,
    clock: _Clock,
) -> None:
    for breach_num in range(SlidingWindowRateLimiter.CIRCUIT_BREACH_THRESHOLD):
        for _ in range(2):
            await limiter.check_and_increment(
                operation="op",
                scope="s",
                limit=2,
                window_seconds=60,
            )
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check_and_increment(
                operation="op",
                scope="s",
                limit=2,
                window_seconds=60,
            )
        is_last_breach = breach_num == SlidingWindowRateLimiter.CIRCUIT_BREACH_THRESHOLD - 1
        assert exc_info.value.circuit_opened is is_last_breach

        clock.advance(61)

    with pytest.raises(RateLimiterCircuitOpen):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=2,
            window_seconds=60,
        )


async def test_circuit_blocks_all_operations_for_scope(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
) -> None:
    circuit_key = limiter._circuit_key("repo-x")
    await fake_redis.setex(circuit_key, 900, "1")

    with pytest.raises(RateLimiterCircuitOpen) as exc_info:
        await limiter.check_and_increment(
            operation="op-a",
            scope="repo-x",
            limit=100,
            window_seconds=3600,
        )
    assert exc_info.value.remaining_seconds > 0

    with pytest.raises(RateLimiterCircuitOpen):
        await limiter.check_and_increment(
            operation="op-b",
            scope="repo-x",
            limit=100,
            window_seconds=3600,
        )


async def test_circuit_allows_after_pause_expires(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
    clock: _Clock,
) -> None:
    circuit_key = limiter._circuit_key("repo-x")
    await fake_redis.setex(circuit_key, 60, "1")

    with pytest.raises(RateLimiterCircuitOpen):
        await limiter.check_and_increment(
            operation="op",
            scope="repo-x",
            limit=10,
            window_seconds=3600,
        )

    clock.advance(61)

    await limiter.check_and_increment(
        operation="op",
        scope="repo-x",
        limit=10,
        window_seconds=3600,
    )


async def test_circuit_scope_differs_from_rate_scope(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
) -> None:
    circuit_key = limiter._circuit_key("my-repo")
    await fake_redis.setex(circuit_key, 900, "1")

    with pytest.raises(RateLimiterCircuitOpen, match="my-repo"):
        await limiter.check_and_increment(
            operation="slack_finding",
            scope="#security-channel",
            limit=30,
            window_seconds=3600,
            circuit_scope="my-repo",
        )


# ---------------------------------------------------------------------------
# Alert deduplication
# ---------------------------------------------------------------------------


async def test_first_breach_should_alert(
    limiter: SlidingWindowRateLimiter,
) -> None:
    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=1,
            window_seconds=60,
        )
    assert exc_info.value.should_alert is True


async def test_subsequent_breach_same_hour_no_alert(
    limiter: SlidingWindowRateLimiter,
    clock: _Clock,
) -> None:
    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=1,
            window_seconds=60,
        )
    assert exc_info.value.should_alert is True

    clock.advance(61)

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=1,
            window_seconds=60,
        )
    assert exc_info.value.should_alert is False


async def test_alert_resets_after_dedup_window(
    limiter: SlidingWindowRateLimiter,
    clock: _Clock,
) -> None:
    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=1,
            window_seconds=60,
        )
    assert exc_info.value.should_alert is True

    clock.advance(SlidingWindowRateLimiter.ALERT_DEDUP_SEC + 1)

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=1,
            window_seconds=60,
        )
    assert exc_info.value.should_alert is True


# ---------------------------------------------------------------------------
# Fail-open on Redis error
# ---------------------------------------------------------------------------


async def test_fail_open_on_redis_error(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
) -> None:
    fake_redis.raise_on_next = ConnectionError("Redis unavailable")

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )


async def test_fail_open_on_circuit_check_error(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
) -> None:
    fake_redis.raise_on_next = TimeoutError("Redis timeout")

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=1,
        window_seconds=60,
    )


# ---------------------------------------------------------------------------
# Exception attributes
# ---------------------------------------------------------------------------


async def test_rate_limit_exceeded_attributes(
    limiter: SlidingWindowRateLimiter,
) -> None:
    await limiter.check_and_increment(
        operation="slack_finding",
        scope="#ch",
        limit=1,
        window_seconds=3600,
    )
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_and_increment(
            operation="slack_finding",
            scope="#ch",
            limit=1,
            window_seconds=3600,
        )
    exc = exc_info.value
    assert exc.operation == "slack_finding"
    assert exc.scope == "#ch"
    assert exc.limit == 1
    assert exc.window_seconds == 3600
    assert exc.is_transient is True


async def test_circuit_open_attributes(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
) -> None:
    await fake_redis.setex(limiter._circuit_key("repo"), 600, "1")
    with pytest.raises(RateLimiterCircuitOpen) as exc_info:
        await limiter.check_and_increment(
            operation="op",
            scope="repo",
            limit=10,
            window_seconds=3600,
        )
    exc = exc_info.value
    assert exc.scope == "repo"
    assert exc.remaining_seconds > 0
    assert exc.is_transient is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_negative_limit_raises_value_error(
    limiter: SlidingWindowRateLimiter,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=-1,
            window_seconds=60,
        )


async def test_zero_window_raises_value_error(
    limiter: SlidingWindowRateLimiter,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=10,
            window_seconds=0,
        )


async def test_negative_window_raises_value_error(
    limiter: SlidingWindowRateLimiter,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=10,
            window_seconds=-1,
        )


# ---------------------------------------------------------------------------
# Default circuit_scope fallback
# ---------------------------------------------------------------------------


async def test_circuit_scope_defaults_to_scope(
    limiter: SlidingWindowRateLimiter,
    fake_redis: _FakeRedis,
) -> None:
    circuit_key = limiter._circuit_key("my-scope")
    await fake_redis.setex(circuit_key, 300, "1")

    with pytest.raises(RateLimiterCircuitOpen, match="my-scope"):
        await limiter.check_and_increment(
            operation="op",
            scope="my-scope",
            limit=10,
            window_seconds=60,
        )


# ---------------------------------------------------------------------------
# Partial window occupancy
# ---------------------------------------------------------------------------


async def test_mixed_expired_and_current_entries(
    limiter: SlidingWindowRateLimiter,
    clock: _Clock,
) -> None:
    """Old entries expire; recent ones count toward the limit."""
    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=3,
        window_seconds=60,
    )
    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=3,
        window_seconds=60,
    )

    clock.advance(40)

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=3,
        window_seconds=60,
    )

    clock.advance(25)

    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=3,
        window_seconds=60,
    )
    await limiter.check_and_increment(
        operation="op",
        scope="s",
        limit=3,
        window_seconds=60,
    )

    with pytest.raises(RateLimitExceeded):
        await limiter.check_and_increment(
            operation="op",
            scope="s",
            limit=3,
            window_seconds=60,
        )

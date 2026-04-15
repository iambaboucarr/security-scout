# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from tools.circuit_breaker import ExternalApiCircuitBreaker


def test_blocked_seconds_after_five_failures_opens_circuit() -> None:
    t = 1_000_000.0
    breaker = ExternalApiCircuitBreaker(now_fn=lambda: t)
    for _ in range(4):
        assert breaker.record_failure("github") is False
    assert breaker.record_failure("github") is True
    assert breaker.blocked_seconds_remaining("github") > 0


def test_resume_event_after_pause_expires() -> None:
    t = [1000.0]

    def clock() -> float:
        return t[0]

    breaker = ExternalApiCircuitBreaker(now_fn=clock)
    for _ in range(5):
        breaker.record_failure("slack")
    assert breaker.blocked_seconds_remaining("slack") > 0
    t[0] += ExternalApiCircuitBreaker.PAUSE_SEC + 1
    assert breaker.blocked_seconds_remaining("slack") == 0
    assert breaker.take_resume_log_event("slack") is True
    assert breaker.take_resume_log_event("slack") is False

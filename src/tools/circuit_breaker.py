# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

ExternalApiName = Literal["github", "slack"]


class ExternalApiCircuitBreaker:
    FAILURE_WINDOW_SEC = 600
    FAILURE_THRESHOLD = 5
    PAUSE_SEC = 300

    __slots__ = ("_failures", "_now", "_open_until", "_pending_resume_log")

    def __init__(self, *, now_fn: Callable[[], float] | None = None) -> None:
        self._now = now_fn or time.time
        self._failures: dict[str, list[float]] = {"github": [], "slack": []}
        self._open_until: dict[str, float] = {"github": 0.0, "slack": 0.0}
        self._pending_resume_log: dict[str, bool] = {"github": False, "slack": False}

    def _prune_window(self, api: ExternalApiName) -> None:
        now = self._now()
        window = self.FAILURE_WINDOW_SEC
        self._failures[api] = [t for t in self._failures[api] if now - t <= window]

    def _check_resume_from_pause(self, api: ExternalApiName) -> None:
        now = self._now()
        until = self._open_until[api]
        if until > 0.0 and now >= until:
            self._open_until[api] = 0.0
            self._pending_resume_log[api] = True

    def blocked_seconds_remaining(self, api: ExternalApiName) -> int:
        """Seconds to wait before calling this API; 0 if calls are allowed."""
        self._check_resume_from_pause(api)
        until = self._open_until[api]
        now = self._now()
        if until <= 0.0 or now >= until:
            return 0
        return max(1, int(until - now) + 1)

    def record_failure(self, api: ExternalApiName) -> bool:
        """Record an API failure. Returns True if the circuit just opened."""
        self._prune_window(api)
        now = self._now()
        self._failures[api].append(now)
        self._prune_window(api)
        if len(self._failures[api]) < self.FAILURE_THRESHOLD:
            return False
        if self._open_until[api] > 0.0 and now < self._open_until[api]:
            return False
        self._open_until[api] = now + self.PAUSE_SEC
        self._failures[api].clear()
        self._pending_resume_log[api] = False
        return True

    def take_resume_log_event(self, api: ExternalApiName) -> bool:
        """Return True once after a pause ends (for AgentActionLog)."""
        self._check_resume_from_pause(api)
        if self._pending_resume_log[api]:
            self._pending_resume_log[api] = False
            return True
        return False


__all__ = ["ExternalApiCircuitBreaker", "ExternalApiName"]

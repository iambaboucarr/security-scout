# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai.provider import LLMProvider
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.issue_tracker import IssueTrackerCredentials
from tools.rate_limiter import SlidingWindowRateLimiter


@dataclass(frozen=True, slots=True)
class ScheduleRetryParams:
    """Arguments for ARQ ``enqueue_job`` with ``_defer_by`` (or equivalent delayed execution)."""

    workflow_run_id: uuid.UUID
    delay_seconds: int
    state: str
    reason: str


@dataclass(frozen=True, slots=True)
class AdvisoryWorkflowParams:
    """Keyword-style inputs for :func:`run_advisory_workflow` (bundled to satisfy a lean call surface)."""

    ghsa_id: str
    advisory_source: Literal["repository", "global"] = "repository"
    run_id: uuid.UUID | None = None
    llm: LLMProvider | None = None
    reasoning_model: str = "claude-sonnet-4-6"
    circuit_breaker: ExternalApiCircuitBreaker | None = None
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None = None
    resume_workflow_run_id: uuid.UUID | None = None
    rate_limiter: SlidingWindowRateLimiter | None = None
    tracker_credentials: IssueTrackerCredentials | None = None
    work_dir: Path | None = None
    container_socket: str = "unix:///var/run/docker.sock"

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator.params import ScheduleRetryParams
from ai.provider import LLMProvider
from config import RepoConfig
from tools.issue_tracker import IssueTrackerCredentials
from tools.rate_limiter import SlidingWindowRateLimiter
from tools.scm.protocol import SCMProvider
from tools.slack import SlackClient


@dataclass(frozen=True, slots=True)
class _AdvisoryDeps:
    session: AsyncSession
    repo: RepoConfig
    scm: SCMProvider
    http: httpx.AsyncClient
    slack: SlackClient
    ghsa_id: str
    advisory_source: Literal["repository", "global"]
    run_id: uuid.UUID | None
    llm: LLMProvider | None
    reasoning_model: str
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None
    rate_limiter: SlidingWindowRateLimiter | None
    tracker_credentials: IssueTrackerCredentials | None
    work_dir: Path | None
    container_socket: str

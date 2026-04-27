# SPDX-License-Identifier: Apache-2.0
"""Deduplication helpers for advisory workflow enqueue and worker.

Redis SET idempotency keys and durable checks against ``WorkflowRun`` / ``Finding``
rows. ``repo_slug`` is ``{github_org}/{github_repo}`` lowercased, matching
``WorkflowRun.repo_name`` and ``Finding.repo_name`` — not the short ``repo``
name from ``repos.yaml``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from arq.connections import ArqRedis
from sqlalchemy import ColumnElement, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import AdvisoryWorkflowState, Finding, FindingStatus, WorkflowKind, WorkflowRun
from tools.json_predicate import json_text_at_upper_trimmed
from tools.scm import normalise_ghsa_id

_LOG = structlog.get_logger(__name__)

_DEDUP_KEY_PREFIX = "dedup:advisory:"
_MIN_DEDUP_LOCK_TTL_SEC = 300
_FINDING_STATUS_BLOCKS_NEW_ADVISORY_RUN: frozenset[FindingStatus] = frozenset(
    (FindingStatus.unconfirmed, FindingStatus.confirmed_high, FindingStatus.confirmed_low),
)
_RECOVERABLE_ERROR_STATES = frozenset(
    {
        AdvisoryWorkflowState.error_triage.value,
        AdvisoryWorkflowState.error_sandbox.value,
        AdvisoryWorkflowState.error_reporting.value,
    },
)


def default_advisory_dedup_lock_ttl_seconds(poll_interval_seconds: int | None) -> int:
    """TTL for Redis idempotency keys: at least 5 minutes, or the poll interval if larger."""
    if poll_interval_seconds is None or poll_interval_seconds < 1:
        return _MIN_DEDUP_LOCK_TTL_SEC
    return max(_MIN_DEDUP_LOCK_TTL_SEC, int(poll_interval_seconds))


def advisory_dedup_lock_key(*, repo_slug: str, ghsa: str) -> str:
    return f"{_DEDUP_KEY_PREFIX}{repo_slug}:{ghsa}"


async def try_acquire_advisory_dedup_lock(
    redis: ArqRedis,
    *,
    repo_slug: str,
    ghsa: str,
    ttl_seconds: int,
) -> bool:
    """Return True if a new idempotency key was set (SET NX with TTL)."""
    if ttl_seconds < 1:
        msg = "ttl_seconds must be >= 1"
        raise ValueError(msg)
    key = advisory_dedup_lock_key(repo_slug=repo_slug, ghsa=ghsa)
    was_set = await redis.set(key, "1", nx=True, ex=ttl_seconds)
    return was_set is not None


def _ghsa_match_clause(ghsa: str) -> ColumnElement[bool]:
    return or_(
        WorkflowRun.advisory_ghsa_id == ghsa,
        and_(
            WorkflowRun.advisory_ghsa_id.is_(None),
            WorkflowRun.finding_id == Finding.id,
            json_text_at_upper_trimmed(Finding.evidence, "ghsa_id") == ghsa,
        ),
    )


def _run_blocks_dedupe_sql(now: datetime) -> ColumnElement[bool]:
    """States where an existing run should block a new enqueue (see lookback table)."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    window_start = now - timedelta(hours=24)
    return or_(
        WorkflowRun.completed_at.is_(None),
        WorkflowRun.state == AdvisoryWorkflowState.error_unrecoverable.value,
        and_(
            WorkflowRun.state.in_(_RECOVERABLE_ERROR_STATES),
            WorkflowRun.completed_at.is_not(None),
            WorkflowRun.completed_at >= window_start,
        ),
    )


async def has_active_workflow_run(
    session: AsyncSession,
    *,
    repo_slug: str,
    ghsa_id: str,
    now: datetime,
) -> bool:
    """True when a prior run should block a new advisory workflow for the same repo + GHSA."""
    g = normalise_ghsa_id(ghsa_id)
    stmt = (
        select(WorkflowRun.id)
        .select_from(WorkflowRun)
        .outerjoin(Finding, WorkflowRun.finding_id == Finding.id)
        .where(
            WorkflowRun.workflow_type == WorkflowKind.advisory,
            WorkflowRun.repo_name == repo_slug,
            _ghsa_match_clause(g),
            _run_blocks_dedupe_sql(now),
        )
        .limit(1)
    )
    r = await session.execute(stmt)
    return r.scalar_one_or_none() is not None


async def has_existing_advisory_finding(
    session: AsyncSession,
    *,
    repo_slug: str,
    ghsa_id: str,
) -> bool:
    """True if a still-relevant advisory ``Finding`` exists for this repo + GHSA.

    Human-terminal rows (false positive, accepted risk) and triage/execution
    error rows do not block a new advisory workflow so operators can re-run
    or respond to a fresh event.
    """
    g = normalise_ghsa_id(ghsa_id)
    stmt = (
        select(Finding.id)
        .where(
            Finding.workflow == WorkflowKind.advisory,
            Finding.repo_name == repo_slug,
            Finding.status.in_(_FINDING_STATUS_BLOCKS_NEW_ADVISORY_RUN),
            json_text_at_upper_trimmed(Finding.evidence, "ghsa_id") == g,
        )
        .limit(1)
    )
    r = await session.execute(stmt)
    return r.scalar_one_or_none() is not None


async def try_enqueue_advisory(
    redis: ArqRedis,
    *,
    repo_config_name: str,
    repo_slug: str,
    ghsa_id: str,
    advisory_source: str = "repository",
    resume_workflow_run_id: str | None = None,
    force: bool = False,
    poll_interval_seconds: int | None = None,
) -> str | None:
    """Attempt SET NX then enqueue ``process_advisory_workflow_job``. Returns ARQ job id or None.

    When *resume_workflow_run_id* is set, or *force* is true, the Redis idempotency key
    is not acquired (resume and operator overrides must not be blocked by the enqueue token).
    """
    try:
        g = normalise_ghsa_id(ghsa_id)
    except ValueError as e:
        _LOG.error(
            "advisory_enqueue_invalid_ghsa",
            ghsa_id=ghsa_id,
            err=str(e),
        )
        return None

    if resume_workflow_run_id is None and not force:
        ttl = default_advisory_dedup_lock_ttl_seconds(poll_interval_seconds)
        acquired = await try_acquire_advisory_dedup_lock(redis, repo_slug=repo_slug, ghsa=g, ttl_seconds=ttl)
        if not acquired:
            _LOG.info(
                "advisory_dedupe_skip_enqueue",
                metric_name="advisory_poll_skipped_dedupe_total",
                reason="redis_lock",
                repo=repo_config_name,
                repo_slug=repo_slug,
                ghsa_id=g,
            )
            return None

    job = await redis.enqueue_job(
        "process_advisory_workflow_job",
        repo_name=repo_config_name,
        ghsa_id=ghsa_id,
        advisory_source=advisory_source,
        resume_workflow_run_id=resume_workflow_run_id,
    )
    if job is None:
        return None
    if isinstance(job, str):
        return job
    jid = getattr(job, "job_id", None)
    return str(jid) if jid is not None else None

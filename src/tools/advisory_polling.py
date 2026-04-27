# SPDX-License-Identifier: Apache-2.0
"""Deduplication helpers for advisory workflow enqueue and worker.

Redis SET idempotency keys and durable checks against ``WorkflowRun`` / ``Finding``
rows. ``repo_slug`` is ``{github_org}/{github_repo}`` lowercased, matching
``WorkflowRun.repo_name`` and ``Finding.repo_name`` — not the short ``repo``
name from ``repos.yaml``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import httpx
import structlog
from arq.connections import ArqRedis
from redis.exceptions import RedisError
from sqlalchemy import ColumnElement, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import AppConfig, RepoConfig, Settings
from exceptions import SecurityScoutError
from models import AdvisoryWorkflowState, Finding, FindingStatus, WorkflowKind, WorkflowRun
from tools.json_predicate import json_text_at_upper_trimmed
from tools.rate_limiter import RateLimiterCircuitOpen, RateLimitExceeded, SlidingWindowRateLimiter
from tools.scm import normalise_ghsa_id
from tools.scm.github import GitHubSCMProvider
from tools.scm.models import AdvisoryData

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


# --- Repository advisory list sync (polling) --------------------------------

_SYNC_CONCURRENCY = 8
_POLL_WM_PREFIX = "poll:advisory:wm:"


def advisory_list_watermark_key(*, repo_slug: str, state: str) -> str:
    return f"{_POLL_WM_PREFIX}{repo_slug}:{state.strip().lower()}"


def _parse_watermark_value(raw: str | bytes | None) -> datetime | None:
    if raw is None:
        return None
    s = raw.decode() if isinstance(raw, bytes) else str(raw)
    s = s.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _watermark_iso_utc(dt: datetime) -> str:
    dt_utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt_utc.isoformat()


def _advisory_utc(a: AdvisoryData) -> datetime | None:
    if a.updated_at is not None:
        return a.updated_at
    return a.published_at


class _GlobalEnqueueBudget:
    """Per-tick global successful enqueue count (cap-protected, serialised)."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self.n = 0
        self._lock = asyncio.Lock()

    async def try_one_enqueue(
        self,
        build: Callable[[], Awaitable[str | None]],
    ) -> tuple[str | None, bool]:
        """Run *build()* if under the global cap. Returns (job_id, hit_global_cap)."""
        async with self._lock:
            if self.n >= self._cap:
                return None, True
            result = await build()
            if result is not None:
                self.n += 1
            return result, False


def _advisory_rate_limiter_build(
    rate_limiter: SlidingWindowRateLimiter,
    settings: Settings,
) -> Awaitable[None]:
    return rate_limiter.check_and_increment(
        operation="advisory_poll",
        scope="advisory_poll",
        limit=settings.advisory_poll_rate_per_hour,
        window_seconds=3600,
    )


async def _sync_one_poll_state(
    *,
    redis: ArqRedis,
    scm: GitHubSCMProvider,
    rate_limiter: SlidingWindowRateLimiter,
    settings: Settings,
    repo: RepoConfig,
    repo_config_name: str,
    repo_slug: str,
    state: str,
    global_budget: _GlobalEnqueueBudget,
) -> None:
    st = state.strip().lower()
    wm_key = advisory_list_watermark_key(repo_slug=repo_slug, state=st)
    raw = await redis.get(wm_key)
    old_w = _parse_watermark_value(raw) if raw else None

    if old_w is None and repo.advisory_poll_seed_without_enqueue:
        it = aiter(
            scm.iter_list_advisories(
                repo_slug,
                state=st,
                per_page=repo.advisory_poll_per_page,
                max_pages=1,
            )
        )
        try:
            await _advisory_rate_limiter_build(rate_limiter, settings)
            first: tuple[AdvisoryData, ...] = await it.__anext__()
        except StopAsyncIteration, RateLimitExceeded, RateLimiterCircuitOpen:
            first = ()
        uas = [u for a in first if (u := _advisory_utc(a)) is not None]
        if uas:
            seed_max = max(uas)
            await redis.set(wm_key, _watermark_iso_utc(seed_max))
        _LOG.info(
            "advisory_poll_seed_only",
            metric_name="advisory_poll_listed_total",
            repo=repo_config_name,
            state=st,
            listed=len(first),
        )
        return

    per_repo_enqueues = 0
    hit_enq_cap = False
    seen_ua: list[datetime] = []
    enqueued_ua: list[datetime] = []
    natural_stop = False
    exhausted_pages = False
    hit_rl = False
    listed = 0

    it2 = aiter(
        scm.iter_list_advisories(
            repo_slug,
            state=st,
            per_page=repo.advisory_poll_per_page,
            max_pages=repo.advisory_poll_max_pages,
        )
    )
    while True:
        if hit_enq_cap or hit_rl or natural_stop:
            break
        try:
            await _advisory_rate_limiter_build(rate_limiter, settings)
        except RateLimitExceeded, RateLimiterCircuitOpen:
            hit_rl = True
            break
        try:
            page: tuple[AdvisoryData, ...] = await it2.__anext__()
        except StopAsyncIteration:
            exhausted_pages = True
            break
        listed += len(page)
        for adv in page:
            ua = _advisory_utc(adv)
            if old_w is not None and ua is not None and ua <= old_w:
                natural_stop = True
                break
            if ua is not None:
                seen_ua.append(ua)

            if per_repo_enqueues >= repo.advisory_poll_max_enqueues_per_tick:
                hit_enq_cap = True
                break

            ghsa = adv.ghsa_id

            async def _do_try_enqueue(ghsa_id: str = ghsa) -> str | None:
                return await try_enqueue_advisory(
                    redis,
                    repo_config_name=repo_config_name,
                    repo_slug=repo_slug,
                    ghsa_id=ghsa_id,
                    advisory_source="repository",
                    poll_interval_seconds=settings.advisory_poll_interval_seconds,
                )

            job, hit_g = await global_budget.try_one_enqueue(_do_try_enqueue)
            if hit_g:
                hit_enq_cap = True
                break
            if job is not None:
                per_repo_enqueues += 1
                if ua is not None:
                    enqueued_ua.append(ua)
                try:
                    gid = normalise_ghsa_id(ghsa)
                except ValueError:
                    gid = ghsa
                _LOG.info(
                    "advisory_poll_enqueued",
                    metric_name="advisory_poll_enqueued_total",
                    repo=repo_config_name,
                    state=st,
                    ghsa_id=gid,
                )
        if natural_stop or hit_enq_cap:
            break

    if listed:
        _LOG.info(
            "advisory_poll_listed_batch",
            metric_name="advisory_poll_listed_total",
            repo=repo_config_name,
            state=st,
            listed=listed,
        )

    if hit_rl or hit_enq_cap:
        new_w = old_w
    elif enqueued_ua:
        new_w = min(enqueued_ua)
    elif not enqueued_ua and seen_ua and not hit_enq_cap and not hit_rl and (natural_stop or exhausted_pages):
        new_w = min(seen_ua)
    else:
        new_w = old_w

    if new_w != old_w and new_w is not None:
        await redis.set(wm_key, _watermark_iso_utc(new_w))

    if hit_rl or hit_enq_cap:
        _LOG.info(
            "advisory_poll_state_sync_end",
            repo=repo_config_name,
            state=st,
            hit_rate_limit=hit_rl,
            hit_enqueue_cap=hit_enq_cap,
        )


async def _run_one_repo_for_sync(
    sem: asyncio.Semaphore,
    scm: GitHubSCMProvider,
    *,
    redis: ArqRedis,
    rate_limiter: SlidingWindowRateLimiter,
    settings: Settings,
    repo: RepoConfig,
    repo_slug: str,
    global_budget: _GlobalEnqueueBudget,
) -> str | None:
    async with sem:
        if "published" in (x.strip().lower() for x in repo.advisory_poll_states):
            _LOG.info(
                "advisory_poll_published_in_poll_states",
                metric_name="advisory_poll_redundant_with_webhook_total",
                repo=repo.name,
            )
        for st in repo.advisory_poll_states:
            try:
                await _sync_one_poll_state(
                    redis=redis,
                    scm=scm,
                    rate_limiter=rate_limiter,
                    settings=settings,
                    repo=repo,
                    repo_config_name=repo.name,
                    repo_slug=repo_slug,
                    state=st,
                    global_budget=global_budget,
                )
            except (
                SecurityScoutError,
                httpx.RequestError,
                RedisError,
            ):
                _LOG.exception(
                    "advisory_poll_repo_state_failed",
                    repo=repo.name,
                    state=st,
                )
                return "partial"
    return "ok"


async def run_repository_advisories_sync(
    *,
    settings: Settings,
    app_config: AppConfig,
    redis: ArqRedis | None,
    rate_limiter: SlidingWindowRateLimiter | None,
) -> None:
    """List repository security advisories per config, enqueue new runs, update Redis watermarks.

    Skips with a log when Redis or the rate limiter is missing, or when ``scm_provider`` is not
    ``github``, or when no repository has a non-empty ``advisory_poll_states`` list.
    """
    t0 = time.perf_counter()
    if settings.scm_provider != "github":
        _LOG.info("advisory_poll_skipped", reason="non_github_scm", scm=settings.scm_provider)
        return
    if redis is None or rate_limiter is None:
        _LOG.error("advisory_poll_skipped", reason="missing_redis_or_rate_limiter")
        return
    to_poll = [r for r in app_config.repos.repos if r.advisory_poll_states]
    if not to_poll:
        return

    cap_g = min(settings.advisory_poll_max_enqueues_per_tick_global, 10**9)
    budget = _GlobalEnqueueBudget(cap_g)
    any_partial = False
    any_fail = False
    sem = asyncio.Semaphore(_SYNC_CONCURRENCY)

    async with GitHubSCMProvider(settings.github_pat) as scm:
        tasks = [
            _run_one_repo_for_sync(
                sem,
                scm,
                redis=redis,
                rate_limiter=rate_limiter,
                settings=settings,
                repo=r,
                repo_slug=f"{r.github_org}/{r.github_repo}".lower(),
                global_budget=budget,
            )
            for r in to_poll
        ]
        out = await asyncio.gather(*tasks, return_exceptions=True)
    for i, o in enumerate(out):
        if isinstance(o, BaseException) and not isinstance(o, Exception):
            raise o
        if isinstance(o, Exception):
            any_fail = True
            _LOG.exception(
                "advisory_poll_repo_unhandled",
                repo=to_poll[i].name,
            )
        elif o == "partial":
            any_partial = True

    elapsed = time.perf_counter() - t0
    if any_fail and not any_partial:
        result: str = "fail"
    elif any_fail or any_partial:
        result = "partial"
    else:
        result = "ok"
    _LOG.info(
        "advisory_poll_tick",
        metric_name="advisory_poll_tick_total",
        result=result,
    )
    _LOG.info(
        "advisory_poll_tick_duration",
        metric_name="advisory_poll_tick_duration_seconds",
        seconds=round(elapsed, 3),
    )


async def run_repository_advisories_sync_from_worker_ctx(ctx: dict[str, object]) -> None:
    """Load config from a worker job context and run :func:`run_repository_advisories_sync`."""
    settings = ctx.get("settings")
    app = ctx.get("app_config")
    redis = ctx.get("redis")
    rl = ctx.get("rate_limiter")
    if not isinstance(settings, Settings) or not isinstance(app, AppConfig):
        _LOG.error("advisory_sync_missing_ctx")
        return
    if redis is not None and not isinstance(redis, ArqRedis):
        _LOG.error("advisory_sync_invalid_ctx", field="redis")
        return
    await run_repository_advisories_sync(
        settings=settings,
        app_config=app,
        redis=redis,
        rate_limiter=rl if isinstance(rl, SlidingWindowRateLimiter) else None,
    )

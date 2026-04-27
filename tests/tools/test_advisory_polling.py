# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy import delete

from db import create_engine, create_session_factory
from models import (
    AdvisoryWorkflowState,
    Base,
    Finding,
    FindingStatus,
    Severity,
    WorkflowKind,
    WorkflowRun,
)
from tools.advisory_polling import (
    _log_advisory_poll_api_error,
    _log_advisory_poll_ratelimit_gauge,
    default_advisory_dedup_lock_ttl_seconds,
    has_active_workflow_run,
    has_existing_advisory_finding,
    try_acquire_advisory_dedup_lock,
    try_enqueue_advisory,
)
from tools.github import GitHubAPIError, GitHubMalformedResponseError

_GH = "GHSA-ABCD-ABCD-ABCD"


@pytest.mark.asyncio
async def test_try_acquire_dedup_lock_fakeredis_second_fails() -> None:
    from fakeredis import FakeAsyncRedis

    r = FakeAsyncRedis()
    assert await try_acquire_advisory_dedup_lock(r, repo_slug="o/r", ghsa=_GH, ttl_seconds=60) is True
    assert await try_acquire_advisory_dedup_lock(r, repo_slug="o/r", ghsa=_GH, ttl_seconds=60) is False


@pytest.mark.asyncio
async def test_try_acquire_dedup_lock_fakeredis_concurrent_single_winner() -> None:
    from fakeredis import FakeAsyncRedis

    r = FakeAsyncRedis()

    async def one() -> bool:
        return await try_acquire_advisory_dedup_lock(r, repo_slug="o/r", ghsa=_GH, ttl_seconds=60)

    results = await asyncio.gather(*[one() for _ in range(24)])
    assert sum(1 for x in results if x) == 1
    assert sum(1 for x in results if not x) == 23


def test_default_advisory_dedup_lock_ttl_seconds() -> None:
    assert default_advisory_dedup_lock_ttl_seconds(None) == 300
    assert default_advisory_dedup_lock_ttl_seconds(0) == 300
    assert default_advisory_dedup_lock_ttl_seconds(200) == 300
    assert default_advisory_dedup_lock_ttl_seconds(400) == 400


@pytest.mark.asyncio
async def test_has_existing_advisory_finding_db_session(db_session) -> None:
    fid = uuid.uuid4()
    db_session.add(
        Finding(
            id=fid,
            workflow=WorkflowKind.advisory,
            repo_name="acme/app",
            source_ref="https://example.com",
            severity=Severity.high,
            title="t",
            status=FindingStatus.unconfirmed,
            evidence={"ghsa_id": _GH},
        ),
    )
    await db_session.commit()
    on = await has_existing_advisory_finding(db_session, repo_slug="acme/app", ghsa_id=_GH)
    off = await has_existing_advisory_finding(db_session, repo_slug="other/thing", ghsa_id=_GH)
    assert on is True
    assert off is False


@pytest.mark.asyncio
async def test_has_existing_advisory_finding_ignores_false_positive(db_session) -> None:
    db_session.add(
        Finding(
            id=uuid.uuid4(),
            workflow=WorkflowKind.advisory,
            repo_name="acme/app",
            source_ref="https://example.com",
            severity=Severity.high,
            title="t",
            status=FindingStatus.false_positive,
            evidence={"ghsa_id": _GH},
        ),
    )
    await db_session.commit()
    out = await has_existing_advisory_finding(db_session, repo_slug="acme/app", ghsa_id=_GH)
    assert out is False


@pytest.mark.asyncio
async def test_has_active_incomplete_workflow_run_blocks(db_session) -> None:
    now = datetime.now(UTC)
    rid = uuid.uuid4()
    db_session.add(
        WorkflowRun(
            id=rid,
            workflow_type=WorkflowKind.advisory,
            repo_name="acme/app",
            advisory_ghsa_id=_GH,
            state=AdvisoryWorkflowState.triaging.value,
            completed_at=None,
        ),
    )
    await db_session.commit()
    assert await has_active_workflow_run(db_session, repo_slug="acme/app", ghsa_id=_GH, now=now) is True


@pytest.mark.asyncio
async def test_has_active_workflow_run_done_does_not_block(db_session) -> None:
    now = datetime.now(UTC)
    rid = uuid.uuid4()
    db_session.add(
        WorkflowRun(
            id=rid,
            workflow_type=WorkflowKind.advisory,
            repo_name="acme/app",
            advisory_ghsa_id=_GH,
            state=AdvisoryWorkflowState.done.value,
            completed_at=now - timedelta(hours=1),
        ),
    )
    await db_session.commit()
    assert await has_active_workflow_run(db_session, repo_slug="acme/app", ghsa_id=_GH, now=now) is False


@pytest.mark.asyncio
async def test_has_active_unrecoverable_error_blocks_forever(db_session) -> None:
    now = datetime.now(UTC)
    rid = uuid.uuid4()
    past = now - timedelta(days=10)
    db_session.add(
        WorkflowRun(
            id=rid,
            workflow_type=WorkflowKind.advisory,
            repo_name="acme/app",
            advisory_ghsa_id=_GH,
            state=AdvisoryWorkflowState.error_unrecoverable.value,
            completed_at=past,
        ),
    )
    await db_session.commit()
    assert await has_active_workflow_run(db_session, repo_slug="acme/app", ghsa_id=_GH, now=now) is True


@pytest.mark.asyncio
async def test_has_active_recoverable_error_outside_24h_does_not_block(db_session) -> None:
    now = datetime.now(UTC)
    rid = uuid.uuid4()
    past = now - timedelta(hours=30)
    db_session.add(
        WorkflowRun(
            id=rid,
            workflow_type=WorkflowKind.advisory,
            repo_name="acme/app",
            advisory_ghsa_id=_GH,
            state=AdvisoryWorkflowState.error_triage.value,
            completed_at=past,
        ),
    )
    await db_session.commit()
    assert await has_active_workflow_run(db_session, repo_slug="acme/app", ghsa_id=_GH, now=now) is False


@pytest.mark.parametrize(
    ("state", "completed_hours_ago", "expected_blocks"),
    [
        (AdvisoryWorkflowState.done.value, 1, False),
        (AdvisoryWorkflowState.error_unrecoverable.value, 240, True),
        (AdvisoryWorkflowState.error_triage.value, 1, True),
        (AdvisoryWorkflowState.error_triage.value, 30, False),
        (AdvisoryWorkflowState.error_sandbox.value, 2, True),
        (AdvisoryWorkflowState.error_sandbox.value, 30, False),
        (AdvisoryWorkflowState.error_reporting.value, 2, True),
        (AdvisoryWorkflowState.error_reporting.value, 30, False),
        (AdvisoryWorkflowState.pre_flight_blocked.value, 6, False),
        (AdvisoryWorkflowState.done.value, None, True),
    ],
)
@pytest.mark.asyncio
async def test_has_active_workflow_run_terminal_policy_table(
    db_session,
    state: str,
    completed_hours_ago: int | None,
    expected_blocks: bool,
) -> None:
    now = datetime.now(UTC)
    rid = uuid.uuid4()
    completed_at = None if completed_hours_ago is None else now - timedelta(hours=completed_hours_ago)
    db_session.add(
        WorkflowRun(
            id=rid,
            workflow_type=WorkflowKind.advisory,
            repo_name="acme/app",
            advisory_ghsa_id=_GH,
            state=state,
            completed_at=completed_at,
        ),
    )
    await db_session.commit()
    out = await has_active_workflow_run(db_session, repo_slug="acme/app", ghsa_id=_GH, now=now)
    assert out is expected_blocks


@pytest.mark.asyncio
async def test_try_enqueue_returns_none_when_lock_not_acquired() -> None:
    from fakeredis import FakeAsyncRedis

    r = FakeAsyncRedis()
    await r.set("dedup:advisory:acme/app:GHSA-ABCD-ABCD-ABCD", "1", ex=300)
    out = await try_enqueue_advisory(
        r,
        repo_config_name="demo",
        repo_slug="acme/app",
        ghsa_id=_GH,
    )
    assert out is None


@pytest.mark.asyncio
async def test_try_enqueue_enqueues_and_returns_job_id() -> None:
    from fakeredis import FakeAsyncRedis

    class _P:
        def __init__(self) -> None:
            self._r: Any = FakeAsyncRedis()

        async def set(
            self,
            name: str,
            value: str,
            *,
            nx: bool = False,
            ex: int | None = None,
        ) -> bool | None:
            return await self._r.set(name, value, nx=nx, ex=ex)

        async def enqueue_job(self, *_a: object, **_kwargs: object) -> str:
            return "j1"

    out = await try_enqueue_advisory(
        _P(),
        repo_config_name="demo",
        repo_slug="acme/app",
        ghsa_id=_GH,
    )
    assert out == "j1"


@pytest.mark.asyncio
async def test_try_enqueue_concurrent_fakeredis_at_most_one_job() -> None:
    from fakeredis import FakeAsyncRedis

    class _P:
        def __init__(self) -> None:
            self._r: Any = FakeAsyncRedis()
            self.enqueue_job = AsyncMock(return_value="job-id")

        async def set(
            self,
            name: str,
            value: str,
            *,
            nx: bool = False,
            ex: int | None = None,
        ) -> bool | None:
            return await self._r.set(name, value, nx=nx, ex=ex)

    redis_wrapper = _P()
    results = await asyncio.gather(
        *[
            try_enqueue_advisory(
                redis_wrapper,
                repo_config_name="demo",
                repo_slug="acme/app",
                ghsa_id=_GH,
            )
            for _ in range(40)
        ],
    )
    assert sum(1 for x in results if x is not None) == 1
    assert redis_wrapper.enqueue_job.await_count == 1


@pytest.mark.asyncio
async def test_try_enqueue_bypasses_lock_for_resume() -> None:
    m = MagicMock()
    m.set = AsyncMock()
    m.enqueue_job = AsyncMock(return_value="jr")

    out = await try_enqueue_advisory(
        m,
        repo_config_name="demo",
        repo_slug="acme/app",
        ghsa_id=_GH,
        resume_workflow_run_id="00000000-0000-0000-0000-000000000099",
    )
    assert out == "jr"
    m.set.assert_not_called()


def _require_postgres_url() -> str:
    url = os.environ.get("POSTGRES_TEST_URL", "").strip()
    if not url:
        pytest.fail(
            "POSTGRES_TEST_URL is not set. For local runs: docker compose up -d postgres "
            "and export POSTGRES_TEST_URL (see Makefile POSTGRES_TEST_URL default).",
        )
    return url


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_has_active_workflow_run_postgres_outerjoin() -> None:
    from tools.advisory_polling import has_active_workflow_run as hawr

    url = _require_postgres_url()
    engine = create_engine(url)
    now = datetime.now(UTC)
    rid = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            if conn.dialect.name == "postgresql":
                await conn.execute(
                    sa.text(
                        "ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS advisory_ghsa_id VARCHAR(32)",
                    ),
                )
                await conn.execute(
                    sa.text(
                        "CREATE INDEX IF NOT EXISTS ix_workflow_runs_advisory_ghsa_id "
                        "ON workflow_runs (advisory_ghsa_id)",
                    ),
                )
        factory = create_session_factory(engine)
        async with factory() as session:
            session.add(
                WorkflowRun(
                    id=rid,
                    workflow_type=WorkflowKind.advisory,
                    repo_name="acme/app",
                    advisory_ghsa_id=_GH,
                    state=AdvisoryWorkflowState.received.value,
                    completed_at=None,
                ),
            )
            await session.commit()
        async with factory() as session:
            assert await hawr(session, repo_slug="acme/app", ghsa_id=_GH, now=now) is True
            await session.execute(delete(WorkflowRun).where(WorkflowRun.id == rid))
            await session.commit()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_log_advisory_poll_ratelimit_gauge_emits_structlog_metric() -> None:
    r = httpx.Response(200, headers={"x-ratelimit-remaining": "10"})
    with patch("tools.advisory_polling._LOG") as mlog:
        await _log_advisory_poll_ratelimit_gauge(repo="reponame", response=r)
    mlog.info.assert_called_once()
    assert mlog.info.call_args[0][0] == "advisory_poll_ratelimit_remaining"
    assert mlog.info.call_args[1]["metric_name"] == "advisory_poll_ratelimit_remaining"
    assert mlog.info.call_args[1]["repo"] == "reponame"
    assert mlog.info.call_args[1]["remaining"] == 10


def test_log_advisory_poll_api_error_status_mapping() -> None:
    with patch("tools.advisory_polling._LOG") as m:
        _log_advisory_poll_api_error(repo="r1", exc=GitHubAPIError("x", is_transient=True, http_status=502))
        assert m.warning.call_args[1]["status"] == "502"
    with patch("tools.advisory_polling._LOG") as m:
        _log_advisory_poll_api_error(repo="r1", exc=GitHubMalformedResponseError("x"))
        assert m.warning.call_args[1]["status"] == "parse_error"
    with patch("tools.advisory_polling._LOG") as m:
        _log_advisory_poll_api_error(repo="r1", exc=httpx.ReadTimeout("t"))
        assert m.warning.call_args[1]["status"] == "request_error"
    with patch("tools.advisory_polling._LOG") as m:
        _log_advisory_poll_api_error(repo="r1", exc=RuntimeError("n"))
        assert m.warning.call_args[1]["status"] == "RuntimeError"

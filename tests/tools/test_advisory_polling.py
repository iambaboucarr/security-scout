# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
    default_advisory_dedup_lock_ttl_seconds,
    has_active_workflow_run,
    has_existing_advisory_finding,
    try_acquire_advisory_dedup_lock,
    try_enqueue_advisory,
)

_GH = "GHSA-ABCD-ABCD-ABCD"


@pytest.mark.asyncio
async def test_try_acquire_dedup_lock_fakeredis_second_fails() -> None:
    from fakeredis import FakeAsyncRedis

    r = FakeAsyncRedis()
    assert await try_acquire_advisory_dedup_lock(r, repo_slug="o/r", ghsa=_GH, ttl_seconds=60) is True
    assert await try_acquire_advisory_dedup_lock(r, repo_slug="o/r", ghsa=_GH, ttl_seconds=60) is False


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

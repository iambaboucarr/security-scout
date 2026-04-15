# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from models import WorkflowKind, WorkflowRun
from tools.ops_alerts import (
    _percentile,
    check_error_rate,
    check_latency_p95,
    check_stuck_workflows,
    run_ops_alerts,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _make_run(
    *,
    state: str = "done",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    workflow_type: WorkflowKind = WorkflowKind.advisory,
) -> WorkflowRun:
    now = _utc_now()
    return WorkflowRun(
        id=uuid.uuid4(),
        workflow_type=workflow_type,
        state=state,
        retry_count=0,
        started_at=started_at or now,
        completed_at=completed_at,
    )


# ── check_stuck_workflows ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_stuck_workflows_returns_none_when_no_workflows(db_session: AsyncSession) -> None:
    result = await check_stuck_workflows(db_session, stuck_threshold_minutes=10)
    assert result is None


@pytest.mark.asyncio
async def test_stuck_workflows_ignores_completed_runs(db_session: AsyncSession) -> None:
    now = _utc_now()
    run = _make_run(
        state="done",
        started_at=now - timedelta(minutes=30),
        completed_at=now - timedelta(minutes=20),
    )
    db_session.add(run)
    await db_session.flush()

    result = await check_stuck_workflows(db_session, stuck_threshold_minutes=10, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_stuck_workflows_ignores_terminal_error_states(db_session: AsyncSession) -> None:
    now = _utc_now()
    for state in ("error_triage", "error_reporting", "error_unrecoverable"):
        run = _make_run(state=state, started_at=now - timedelta(minutes=30))
        run.completed_at = now - timedelta(minutes=25)
        db_session.add(run)
    await db_session.flush()

    result = await check_stuck_workflows(db_session, stuck_threshold_minutes=10, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_stuck_workflows_detects_old_non_terminal_run(db_session: AsyncSession) -> None:
    now = _utc_now()
    run = _make_run(state="triaging", started_at=now - timedelta(minutes=15))
    db_session.add(run)
    await db_session.flush()

    result = await check_stuck_workflows(db_session, stuck_threshold_minutes=10, now=now)
    assert result is not None
    assert len(result.workflow_run_ids) == 1
    assert run.id in result.workflow_run_ids
    assert "1 workflow" in result.detail


@pytest.mark.asyncio
async def test_stuck_workflows_ignores_recent_non_terminal_run(db_session: AsyncSession) -> None:
    now = _utc_now()
    run = _make_run(state="triaging", started_at=now - timedelta(minutes=5))
    db_session.add(run)
    await db_session.flush()

    result = await check_stuck_workflows(db_session, stuck_threshold_minutes=10, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_stuck_workflows_reports_multiple(db_session: AsyncSession) -> None:
    now = _utc_now()
    run1 = _make_run(state="triaging", started_at=now - timedelta(minutes=20))
    run2 = _make_run(state="reporting", started_at=now - timedelta(minutes=12))
    db_session.add_all([run1, run2])
    await db_session.flush()

    result = await check_stuck_workflows(db_session, stuck_threshold_minutes=10, now=now)
    assert result is not None
    assert len(result.workflow_run_ids) == 2
    assert "2 workflow" in result.detail


# ── check_error_rate ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_rate_returns_none_when_no_workflows(db_session: AsyncSession) -> None:
    result = await check_error_rate(db_session, threshold=0.20, window_minutes=60)
    assert result is None


@pytest.mark.asyncio
async def test_error_rate_returns_none_when_below_threshold(db_session: AsyncSession) -> None:
    now = _utc_now()
    for _i in range(8):
        db_session.add(_make_run(state="done", started_at=now - timedelta(minutes=10)))
    db_session.add(_make_run(state="error_triage", started_at=now - timedelta(minutes=10)))
    await db_session.flush()

    result = await check_error_rate(db_session, threshold=0.20, window_minutes=60, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_error_rate_fires_when_above_threshold(db_session: AsyncSession) -> None:
    now = _utc_now()
    db_session.add(_make_run(state="done", started_at=now - timedelta(minutes=10)))
    db_session.add(_make_run(state="error_triage", started_at=now - timedelta(minutes=10)))
    db_session.add(_make_run(state="error_reporting", started_at=now - timedelta(minutes=10)))
    await db_session.flush()

    result = await check_error_rate(db_session, threshold=0.20, window_minutes=60, now=now)
    assert result is not None
    assert "66%" in result.detail or "67%" in result.detail
    assert "2/3" in result.detail


@pytest.mark.asyncio
async def test_error_rate_excludes_old_workflows(db_session: AsyncSession) -> None:
    now = _utc_now()
    db_session.add(_make_run(state="error_triage", started_at=now - timedelta(minutes=90)))
    db_session.add(_make_run(state="done", started_at=now - timedelta(minutes=5)))
    await db_session.flush()

    result = await check_error_rate(db_session, threshold=0.20, window_minutes=60, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_error_rate_only_counts_error_terminal_states(db_session: AsyncSession) -> None:
    now = _utc_now()
    for _ in range(3):
        db_session.add(_make_run(state="done", started_at=now - timedelta(minutes=10)))
    db_session.add(_make_run(state="error_unrecoverable", started_at=now - timedelta(minutes=10)))
    await db_session.flush()

    result = await check_error_rate(db_session, threshold=0.20, window_minutes=60, now=now)
    assert result is not None
    assert "25%" in result.detail


# ── check_latency_p95 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_latency_p95_returns_none_when_no_completed_workflows(db_session: AsyncSession) -> None:
    result = await check_latency_p95(db_session, threshold_seconds=60.0, window_minutes=60)
    assert result is None


@pytest.mark.asyncio
async def test_latency_p95_returns_none_when_below_threshold(db_session: AsyncSession) -> None:
    now = _utc_now()
    for i in range(10):
        db_session.add(
            _make_run(
                state="done",
                started_at=now - timedelta(minutes=30, seconds=i),
                completed_at=now - timedelta(minutes=30, seconds=i) + timedelta(seconds=5),
            )
        )
    await db_session.flush()

    result = await check_latency_p95(db_session, threshold_seconds=60.0, window_minutes=60, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_latency_p95_fires_when_above_threshold(db_session: AsyncSession) -> None:
    now = _utc_now()
    # 4 fast + 1 slow: p95 nearest-rank = ceil(5*95/100)-1 = 4 → the slow value
    for _i in range(4):
        db_session.add(
            _make_run(
                state="done",
                started_at=now - timedelta(minutes=20),
                completed_at=now - timedelta(minutes=20) + timedelta(seconds=10),
            )
        )
    db_session.add(
        _make_run(
            state="done",
            started_at=now - timedelta(minutes=20),
            completed_at=now - timedelta(minutes=20) + timedelta(seconds=120),
        )
    )
    await db_session.flush()

    result = await check_latency_p95(db_session, threshold_seconds=60.0, window_minutes=60, now=now)
    assert result is not None
    assert "120.0s" in result.detail


@pytest.mark.asyncio
async def test_latency_p95_excludes_non_done_workflows(db_session: AsyncSession) -> None:
    now = _utc_now()
    db_session.add(
        _make_run(
            state="error_triage",
            started_at=now - timedelta(minutes=20),
            completed_at=now - timedelta(minutes=20) + timedelta(seconds=200),
        )
    )
    db_session.add(
        _make_run(
            state="done",
            started_at=now - timedelta(minutes=20),
            completed_at=now - timedelta(minutes=20) + timedelta(seconds=5),
        )
    )
    await db_session.flush()

    result = await check_latency_p95(db_session, threshold_seconds=60.0, window_minutes=60, now=now)
    assert result is None


# ── _percentile ──────────────────────────────────────────────────────


def test_percentile_empty_returns_zero() -> None:
    assert _percentile([], 95) == 0.0


def test_percentile_single_value() -> None:
    assert _percentile([42.0], 95) == 42.0


def test_percentile_p95_of_20_values() -> None:
    values = sorted(float(i) for i in range(1, 21))
    assert _percentile(values, 95) == 19.0


def test_percentile_p50() -> None:
    values = sorted([1.0, 2.0, 3.0, 4.0])
    assert _percentile(values, 50) == 2.0


# ── run_ops_alerts ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_ops_alerts_fires_nothing_when_clean(db_session: AsyncSession) -> None:
    now = _utc_now()
    for _ in range(5):
        db_session.add(
            _make_run(
                state="done",
                started_at=now - timedelta(minutes=5),
                completed_at=now - timedelta(minutes=5) + timedelta(seconds=3),
            )
        )
    await db_session.flush()

    slack = AsyncMock()
    fired = await run_ops_alerts(
        db_session,
        slack,
        ops_channel="#ops",
        now=now,
    )
    assert fired == []
    slack.notify_workflow_error.assert_not_called()


@pytest.mark.asyncio
async def test_run_ops_alerts_delivers_stuck_workflow_alert(db_session: AsyncSession) -> None:
    now = _utc_now()
    run = _make_run(state="triaging", started_at=now - timedelta(minutes=15))
    db_session.add(run)
    await db_session.flush()

    slack = AsyncMock()
    fired = await run_ops_alerts(
        db_session,
        slack,
        ops_channel="#ops",
        now=now,
    )
    assert len(fired) == 1
    assert "Stuck" in fired[0].title
    slack.notify_workflow_error.assert_called_once()
    call_kw = slack.notify_workflow_error.call_args
    assert call_kw[0][0] == "#ops"
    assert "Stuck" in call_kw[1]["title"]


@pytest.mark.asyncio
async def test_run_ops_alerts_continues_on_slack_delivery_failure(db_session: AsyncSession) -> None:
    now = _utc_now()
    run = _make_run(state="triaging", started_at=now - timedelta(minutes=15))
    db_session.add(run)
    db_session.add(_make_run(state="error_triage", started_at=now - timedelta(minutes=5)))
    await db_session.flush()

    slack = AsyncMock()
    slack.notify_workflow_error.side_effect = RuntimeError("delivery failed")

    with patch("tools.ops_alerts._LOG") as mock_log:
        fired = await run_ops_alerts(
            db_session,
            slack,
            ops_channel="#ops",
            now=now,
        )

    assert len(fired) >= 1
    assert mock_log.exception.called

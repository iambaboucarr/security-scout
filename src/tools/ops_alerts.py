# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import WorkflowRun
from tools.slack import SlackClient

__all__ = [
    "AlertPayload",
    "check_error_rate",
    "check_latency_p95",
    "check_stuck_workflows",
    "run_ops_alerts",
]

_LOG = structlog.get_logger(__name__)

_TERMINAL_STATES = frozenset(
    {
        "done",
        "error_triage",
        "error_reporting",
        "error_unrecoverable",
    }
)


@dataclass(frozen=True, slots=True)
class AlertPayload:
    title: str
    detail: str
    workflow_run_ids: tuple[uuid.UUID, ...]


async def check_stuck_workflows(
    session: AsyncSession,
    *,
    stuck_threshold_minutes: int = 10,
    now: datetime | None = None,
) -> AlertPayload | None:
    """Return an alert if any workflow has been in a non-terminal state longer than the threshold."""
    ref = (now or datetime.now(UTC)) - timedelta(minutes=stuck_threshold_minutes)
    stmt = select(WorkflowRun).where(
        WorkflowRun.started_at <= ref,
        WorkflowRun.completed_at.is_(None),
        WorkflowRun.state.notin_(_TERMINAL_STATES),
    )
    result = await session.execute(stmt)
    stuck = result.scalars().all()
    if not stuck:
        return None

    run_ids = tuple(r.id for r in stuck)
    states = {r.state for r in stuck}
    detail = (
        f"{len(stuck)} workflow(s) stuck in non-terminal state(s) "
        f"({', '.join(sorted(states))}) for more than {stuck_threshold_minutes} minutes."
    )
    _LOG.warning(
        "ops_alert_stuck_workflows",
        metric_name="ops_alert_fired",
        alert="stuck_workflows",
        count=len(stuck),
        workflow_run_ids=[str(r) for r in run_ids],
    )
    return AlertPayload(
        title=f"Stuck workflows ({len(stuck)})",
        detail=detail,
        workflow_run_ids=run_ids,
    )


async def check_error_rate(
    session: AsyncSession,
    *,
    threshold: float = 0.20,
    window_minutes: int = 60,
    now: datetime | None = None,
) -> AlertPayload | None:
    """Return an alert if the ratio of error-state workflows to total exceeds the threshold."""
    cutoff = (now or datetime.now(UTC)) - timedelta(minutes=window_minutes)

    stmt_total = select(func.count()).select_from(WorkflowRun).where(WorkflowRun.started_at >= cutoff)
    total: int = (await session.execute(stmt_total)).scalar_one()
    if total == 0:
        return None

    error_states = tuple(s for s in _TERMINAL_STATES if s.startswith("error_"))
    stmt_errors = (
        select(func.count())
        .select_from(WorkflowRun)
        .where(
            WorkflowRun.started_at >= cutoff,
            WorkflowRun.state.in_(error_states),
        )
    )
    errors: int = (await session.execute(stmt_errors)).scalar_one()
    rate = errors / total

    if rate <= threshold:
        return None

    detail = (
        f"Error rate {rate:.0%} ({errors}/{total} workflows) over the last "
        f"{window_minutes} minutes exceeds {threshold:.0%} threshold."
    )
    _LOG.warning(
        "ops_alert_error_rate",
        metric_name="ops_alert_fired",
        alert="error_rate",
        rate=round(rate, 4),
        errors=errors,
        total=total,
        window_minutes=window_minutes,
    )
    return AlertPayload(
        title="High workflow error rate",
        detail=detail,
        workflow_run_ids=(),
    )


async def check_latency_p95(
    session: AsyncSession,
    *,
    threshold_seconds: float = 60.0,
    window_minutes: int = 60,
    now: datetime | None = None,
) -> AlertPayload | None:
    """Return an alert if p95 advisory-to-Slack latency exceeds the threshold.

    Latency is derived from completed advisory workflows: ``completed_at - started_at``.
    Only runs that reached ``done`` state are included (error workflows are excluded).
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(minutes=window_minutes)

    stmt = select(WorkflowRun).where(
        WorkflowRun.started_at >= cutoff,
        WorkflowRun.state == "done",
        WorkflowRun.completed_at.isnot(None),
    )
    result = await session.execute(stmt)
    runs = result.scalars().all()
    if not runs:
        return None

    durations = sorted(_run_duration_seconds(r) for r in runs)
    p95 = _percentile(durations, 95)

    if p95 <= threshold_seconds:
        return None

    detail = (
        f"Advisory-to-Slack p95 latency is {p95:.1f}s "
        f"(threshold {threshold_seconds:.0f}s) over {len(runs)} "
        f"completed workflow(s) in the last {window_minutes} minutes."
    )
    _LOG.warning(
        "ops_alert_latency_p95",
        metric_name="ops_alert_fired",
        alert="latency_p95",
        p95_seconds=round(p95, 2),
        threshold_seconds=threshold_seconds,
        sample_count=len(runs),
    )
    return AlertPayload(
        title="High advisory-to-Slack latency (p95)",
        detail=detail,
        workflow_run_ids=(),
    )


def _run_duration_seconds(run: WorkflowRun) -> float:
    started = run.started_at.replace(tzinfo=UTC) if run.started_at.tzinfo is None else run.started_at
    completed = run.completed_at
    if completed is None:
        return 0.0
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=UTC)
    return max(0.0, (completed - started).total_seconds())


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Nearest-rank percentile over a pre-sorted list."""
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(len(sorted_values) * pct / 100) - 1)
    return sorted_values[idx]


async def run_ops_alerts(
    session: AsyncSession,
    slack: SlackClient,
    *,
    ops_channel: str,
    stuck_threshold_minutes: int = 10,
    error_rate_threshold: float = 0.20,
    error_rate_window_minutes: int = 60,
    latency_p95_seconds: float = 60.0,
    now: datetime | None = None,
) -> list[AlertPayload]:
    """Run all operational alert checks and deliver any alerts to Slack."""
    fired: list[AlertPayload] = []

    checkers = [
        check_stuck_workflows(
            session,
            stuck_threshold_minutes=stuck_threshold_minutes,
            now=now,
        ),
        check_error_rate(
            session,
            threshold=error_rate_threshold,
            window_minutes=error_rate_window_minutes,
            now=now,
        ),
        check_latency_p95(
            session,
            threshold_seconds=latency_p95_seconds,
            window_minutes=error_rate_window_minutes,
            now=now,
        ),
    ]

    for coro in checkers:
        alert = await coro
        if alert is not None:
            fired.append(alert)
            try:
                await slack.notify_workflow_error(
                    ops_channel,
                    title=alert.title,
                    detail=alert.detail,
                )
            except Exception:
                _LOG.exception(
                    "ops_alert_delivery_failed",
                    alert_title=alert.title,
                    ops_channel=ops_channel,
                )

    if fired:
        _LOG.info(
            "ops_alerts_check_complete",
            metric_name="ops_alerts_check",
            alerts_fired=len(fired),
            alert_names=[a.title for a in fired],
        )
    else:
        _LOG.debug("ops_alerts_check_complete", metric_name="ops_alerts_check", alerts_fired=0)

    return fired

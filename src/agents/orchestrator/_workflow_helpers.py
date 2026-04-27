# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator._constants import _MAX_LOG_OUTPUT
from agents.orchestrator._logging import _LOG
from models import AgentActionLog
from tools.slack import SlackAPIError, SlackClient


def _truncate_log(text: str | None, max_chars: int = _MAX_LOG_OUTPUT) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _safe_exc_detail(exc: BaseException) -> str:
    return type(exc).__name__


async def _append_action_log(
    session: AsyncSession,
    *,
    workflow_run_id: uuid.UUID | None,
    agent: str,
    tool_name: str,
    tool_inputs: dict[str, Any] | None,
    tool_output: str | None,
) -> None:
    row = AgentActionLog(
        agent=agent,
        tool_name=tool_name,
        tool_inputs=tool_inputs,
        tool_output=_truncate_log(tool_output),
        workflow_run_id=workflow_run_id,
    )
    session.add(row)
    await session.flush()


def _now_utc() -> datetime:
    return datetime.now(UTC)


async def _best_effort_error_slack(
    slack: SlackClient,
    channel: str,
    *,
    title: str,
    detail: str,
    workflow_run_id: uuid.UUID | None,
    finding_id: str | None,
) -> None:
    try:
        await slack.notify_workflow_error(
            channel,
            title=title,
            detail=detail,
            workflow_run_id=workflow_run_id,
            finding_id=finding_id,
        )
    except SlackAPIError as e:
        _LOG.error(
            "workflow_error_slack_notify_failed",
            metric_name="workflow_error_total",
            workflow_run_id=str(workflow_run_id) if workflow_run_id else None,
            finding_id=finding_id,
            err=str(e),
        )

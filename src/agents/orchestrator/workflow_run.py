# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from models import WorkflowRun


async def _get_run(session: AsyncSession, run_id: uuid.UUID) -> WorkflowRun | None:
    return await session.get(WorkflowRun, run_id)


async def _require_run(session: AsyncSession, run_id: uuid.UUID, *, missing_message: str) -> WorkflowRun:
    row = await _get_run(session, run_id)
    if row is None:
        raise RuntimeError(missing_message)
    return row

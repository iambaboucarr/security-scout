# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator._workflow_helpers import _append_action_log, _now_utc
from agents.orchestrator.deps import _AdvisoryDeps
from agents.orchestrator.workflow_run import _require_run
from models import AdvisoryWorkflowState, WorkflowKind, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.scm import normalise_ghsa_id


async def _advisory_log_circuit_resumes(
    breaker: ExternalApiCircuitBreaker, session: AsyncSession, run_stable_id: uuid.UUID
) -> None:
    for api in ("github", "slack"):
        if breaker.take_resume_log_event(api):
            await _append_action_log(
                session,
                workflow_run_id=run_stable_id,
                agent="orchestrator",
                tool_name="circuit_breaker",
                tool_inputs={"api": api, "event": "resumed"},
                tool_output=None,
            )
            await session.commit()


async def _advisory_bootstrap(
    d: _AdvisoryDeps,
    log: Any,
    *,
    resume_workflow_run_id: uuid.UUID | None,
    repo_slug: str,
) -> tuple[uuid.UUID, bool]:
    if resume_workflow_run_id is None:
        workflow_started_at = _now_utc()
        run = WorkflowRun(
            workflow_type=WorkflowKind.advisory,
            repo_name=repo_slug,
            advisory_ghsa_id=normalise_ghsa_id(d.ghsa_id),
            state=AdvisoryWorkflowState.received.value,
            retry_count=0,
            finding_id=None,
            started_at=workflow_started_at,
        )
        d.session.add(run)
        await d.session.flush()
        run_stable_id = run.id
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=None,
            to_state=AdvisoryWorkflowState.received.value,
            workflow_run_id=str(run_stable_id),
        )
        run.state = AdvisoryWorkflowState.triaging.value
        await d.session.commit()
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.received.value,
            to_state=AdvisoryWorkflowState.triaging.value,
            workflow_run_id=str(run_stable_id),
        )
        return run_stable_id, True

    loaded = await _require_run(d.session, resume_workflow_run_id, missing_message="workflow run missing for resume")
    if loaded.completed_at is not None:
        msg = "cannot resume a completed workflow run"
        raise RuntimeError(msg)
    if loaded.workflow_type != WorkflowKind.advisory:
        msg = "resume only supported for advisory workflows"
        raise RuntimeError(msg)
    if loaded.state not in (
        AdvisoryWorkflowState.triaging.value,
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.building_env.value,
        AdvisoryWorkflowState.executing_sandbox.value,
        AdvisoryWorkflowState.sandbox_complete.value,
        AdvisoryWorkflowState.reporting.value,
    ):
        msg = f"cannot resume from state {loaded.state!r}"
        raise RuntimeError(msg)
    if loaded.repo_name is None:
        loaded.repo_name = repo_slug
    return loaded.id, loaded.state == AdvisoryWorkflowState.triaging.value

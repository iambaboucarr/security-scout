# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from typing import Any

from agents.orchestrator.deps import _AdvisoryDeps
from agents.orchestrator.sandbox_phase import _run_sandbox_phase
from agents.orchestrator.workflow_run import _require_run
from models import AdvisoryWorkflowState, Finding, WorkflowRun


async def _advisory_sandbox_path(
    d: _AdvisoryDeps,
    log: Any,
    *,
    run_stable_id: uuid.UUID,
    finding: Finding,
) -> WorkflowRun | None:
    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing before sandbox",
    )
    if run.state == AdvisoryWorkflowState.executing_sandbox.value:
        run = await _require_run(
            d.session,
            run_stable_id,
            missing_message="workflow run missing before sandbox resume",
        )
        run.state = AdvisoryWorkflowState.building_env.value
        await d.session.commit()
        log.info(
            "workflow_state_transition",
            metric_name="workflow_state_current",
            from_state=AdvisoryWorkflowState.executing_sandbox.value,
            to_state=AdvisoryWorkflowState.building_env.value,
            workflow_run_id=str(run_stable_id),
        )

    if run.state != AdvisoryWorkflowState.building_env.value:
        return None

    await _run_sandbox_phase(
        session=d.session,
        run_stable_id=run_stable_id,
        finding=finding,
        repo=d.repo,
        scm=d.scm,
        slack=d.slack,
        log=log,
        work_dir=d.work_dir,
        container_socket=d.container_socket,
        schedule_retry=d.schedule_retry,
    )
    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing after sandbox phase",
    )
    if run.state in (
        AdvisoryWorkflowState.error_sandbox.value,
        AdvisoryWorkflowState.error_unrecoverable.value,
    ):
        return run
    return None

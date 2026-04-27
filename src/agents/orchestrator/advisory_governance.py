# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from typing import Any

from agents.governance import GovernanceTier
from agents.orchestrator._workflow_helpers import _append_action_log, _now_utc
from agents.orchestrator.deps import _AdvisoryDeps
from models import AdvisoryWorkflowState, Finding, WorkflowRun


async def _advisory_log_governance_decision(
    d: _AdvisoryDeps, run_stable_id: uuid.UUID, finding: Finding, tier: GovernanceTier
) -> None:
    await _append_action_log(
        d.session,
        workflow_run_id=run_stable_id,
        agent="orchestrator",
        tool_name="governance.decide",
        tool_inputs={
            "severity": finding.severity.value,
            "ssvc_action": finding.ssvc_action.value if finding.ssvc_action else None,
            "known_status": finding.known_status.value if finding.known_status else None,
            "has_governance_config": d.repo.governance is not None,
        },
        tool_output=tier.value,
    )
    await d.session.commit()


async def _advisory_maybe_autoresolve(
    d: _AdvisoryDeps, log: Any, *, run_stable_id: uuid.UUID, run: WorkflowRun, finding: Finding, tier: GovernanceTier
) -> WorkflowRun | None:
    if tier != GovernanceTier.auto_resolve or run.state != AdvisoryWorkflowState.triage_complete.value:
        return None
    done_at = _now_utc()
    run.state = AdvisoryWorkflowState.done.value
    run.completed_at = done_at
    await d.session.commit()
    log.info(
        "advisory_auto_resolved",
        metric_name="advisory_auto_resolved_total",
        finding_id=str(finding.id),
        workflow_run_id=str(run_stable_id),
    )
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.triage_complete.value,
        to_state=AdvisoryWorkflowState.done.value,
        workflow_run_id=str(run_stable_id),
    )
    return run

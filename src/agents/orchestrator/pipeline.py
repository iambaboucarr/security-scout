# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from agents.governance import decide_governance_tier
from agents.orchestrator._logging import _LOG
from agents.orchestrator.advisory_governance import _advisory_log_governance_decision, _advisory_maybe_autoresolve
from agents.orchestrator.advisory_preflight import _advisory_preflight_path
from agents.orchestrator.advisory_triage import _advisory_triage_phase
from agents.orchestrator.bootstrap import _advisory_bootstrap, _advisory_log_circuit_resumes
from agents.orchestrator.deps import _AdvisoryDeps
from agents.orchestrator.params import AdvisoryWorkflowParams
from agents.orchestrator.reporting import (
    _advisory_enter_reporting_state,
    _advisory_finalize_slack_success,
    _advisory_rate_limit_gate,
    _advisory_send_slack_report,
    _advisory_slack_circuit_defer,
)
from agents.orchestrator.sandbox_path import _advisory_sandbox_path
from agents.orchestrator.workflow_run import _require_run
from config import RepoConfig
from models import AdvisoryWorkflowState, Finding, WorkflowRun
from tools.circuit_breaker import ExternalApiCircuitBreaker
from tools.scm.protocol import SCMProvider
from tools.slack import SlackClient


def _advisory_assert_reportable_state(run: WorkflowRun) -> None:
    if run.state not in (
        AdvisoryWorkflowState.triage_complete.value,
        AdvisoryWorkflowState.pre_flight.value,
        AdvisoryWorkflowState.building_env.value,
        AdvisoryWorkflowState.executing_sandbox.value,
        AdvisoryWorkflowState.sandbox_complete.value,
        AdvisoryWorkflowState.reporting.value,
    ):
        msg = f"unexpected workflow state before reporting: {run.state!r}"
        raise RuntimeError(msg)


async def run_advisory_workflow(
    session: AsyncSession,
    repo: RepoConfig,
    scm: SCMProvider,
    http: httpx.AsyncClient,
    slack: SlackClient,
    params: AdvisoryWorkflowParams,
) -> WorkflowRun:
    """Run or resume the advisory triage → Slack report workflow.

    When *params.resume_workflow_run_id* is ``None`` a fresh ``WorkflowRun`` is created.
    Pass an existing run's UUID to resume from where it left off (must be in
    ``triaging``, ``triage_complete``, ``building_env``, ``executing_sandbox``,
    ``sandbox_complete``, or ``reporting`` state).  The resumed run keeps its
    original ``id``, ``started_at``, and ``retry_count``.
    """
    log = _LOG.bind(
        agent="orchestrator",
        run_id=str(params.run_id) if params.run_id else None,
    )
    breaker = params.circuit_breaker or ExternalApiCircuitBreaker()
    d = _AdvisoryDeps(
        session=session,
        repo=repo,
        scm=scm,
        http=http,
        slack=slack,
        ghsa_id=params.ghsa_id,
        advisory_source=params.advisory_source,
        run_id=params.run_id,
        llm=params.llm,
        reasoning_model=params.reasoning_model,
        schedule_retry=params.schedule_retry,
        rate_limiter=params.rate_limiter,
        tracker_credentials=params.tracker_credentials,
        work_dir=params.work_dir,
        container_socket=params.container_socket,
    )
    repo_slug = f"{repo.github_org}/{repo.github_repo}".lower()
    run_stable_id, needs_triage = await _advisory_bootstrap(
        d, log, resume_workflow_run_id=params.resume_workflow_run_id, repo_slug=repo_slug
    )
    await _advisory_log_circuit_resumes(breaker, d.session, run_stable_id)
    tof = await _advisory_triage_phase(d, breaker, log, run_stable_id=run_stable_id, needs_triage=needs_triage)
    if isinstance(tof, WorkflowRun):
        return tof
    finding: Finding = tof
    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing before reporting",
    )
    _advisory_assert_reportable_state(run)
    tier = decide_governance_tier(finding, d.repo.governance)
    await _advisory_log_governance_decision(d, run_stable_id, finding, tier)
    early = await _advisory_maybe_autoresolve(d, log, run_stable_id=run_stable_id, run=run, finding=finding, tier=tier)
    if early is not None:
        return early
    pre = await _advisory_preflight_path(d, breaker, log, run_stable_id=run_stable_id, finding=finding, run=run)
    if pre is not None:
        return pre
    sbx = await _advisory_sandbox_path(d, log, run_stable_id=run_stable_id, finding=finding)
    if sbx is not None:
        return sbx
    run = await _require_run(
        d.session,
        run_stable_id,
        missing_message="workflow run missing after sandbox phase",
    )
    cdef = await _advisory_slack_circuit_defer(d, breaker, run_stable_id=run_stable_id)
    if cdef is not None:
        return cdef
    rlim = await _advisory_rate_limit_gate(d, log, run_stable_id=run_stable_id, run=run, finding=finding)
    if rlim is not None:
        return rlim
    run = await _advisory_enter_reporting_state(d, log, run_stable_id=run_stable_id, run=run)
    sent = await _advisory_send_slack_report(d, breaker, log, run_stable_id=run_stable_id, finding=finding, tier=tier)
    if sent is not None:
        return sent
    return await _advisory_finalize_slack_success(d, log, run_stable_id=run_stable_id, run=run, tier=tier)

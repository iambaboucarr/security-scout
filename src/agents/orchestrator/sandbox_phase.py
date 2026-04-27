# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agents.env_builder import EnvBuildResult, build_environment
from agents.orchestrator._constants import _MSG_RUN_MISSING_AFTER_ENV_BUILD
from agents.orchestrator._workflow_helpers import (
    _append_action_log,
    _best_effort_error_slack,
    _now_utc,
    _safe_exc_detail,
    _truncate_log,
)
from agents.orchestrator.params import ScheduleRetryParams
from agents.orchestrator.workflow_run import _require_run
from agents.sandbox_executor import ExecutionResult, PocType, execute_poc
from config import RepoConfig
from exceptions import SecurityScoutError
from models import AdvisoryWorkflowState, Finding
from tools.docker_sandbox import SandboxError
from tools.scm.protocol import SCMProvider
from tools.slack import SlackClient


def _sandbox_resolved_git_ref(finding: Finding, default_ref: str) -> str:
    oracle_ev = (finding.evidence or {}).get("oracle")
    if not isinstance(oracle_ev, dict):
        return default_ref
    vr = oracle_ev.get("vulnerable_ref")
    if isinstance(vr, str) and vr.strip():
        return vr.strip()
    return default_ref


async def _sandbox_on_env_scout_error(
    *,
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    slack: SlackClient,
    log: Any,
    repo_slug: str,
    e: SecurityScoutError,
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None,
) -> None:
    if e.is_transient:
        run = await _require_run(session, run_stable_id, missing_message=_MSG_RUN_MISSING_AFTER_ENV_BUILD)
        if run.retry_count < 3 and schedule_retry is not None:
            delay = max(1, 2**run.retry_count)
            run.retry_count += 1
            await session.commit()
            log.warning(
                "workflow_transient_retry",
                metric_name="workflow_error_total",
                phase="building_env",
                workflow_run_id=str(run_stable_id),
                retry_count=run.retry_count,
                delay_seconds=delay,
            )
            await schedule_retry(
                ScheduleRetryParams(
                    workflow_run_id=run_stable_id,
                    delay_seconds=delay,
                    state=AdvisoryWorkflowState.building_env.value,
                    reason="env_build_transient",
                ),
            )
            return
        run.state = AdvisoryWorkflowState.error_sandbox.value
        run.error_message = _truncate_log(str(e), 4000)
        run.completed_at = _now_utc()
        await session.commit()
        log.warning(
            "workflow_error",
            metric_name="workflow_error_total",
            phase="building_env",
            workflow_run_id=str(run_stable_id),
            err=str(e),
        )
        await _append_action_log(
            session,
            workflow_run_id=run_stable_id,
            agent="env_builder",
            tool_name="build_environment",
            tool_inputs={"repo": repo_slug},
            tool_output=str(e),
        )
        await session.commit()
        await _best_effort_error_slack(
            slack,
            repo.slack_channel,
            title="Environment build failed",
            detail=_safe_exc_detail(e),
            workflow_run_id=run_stable_id,
            finding_id=str(finding.id),
        )
        return

    run = await _require_run(session, run_stable_id, missing_message=_MSG_RUN_MISSING_AFTER_ENV_BUILD)
    run.state = AdvisoryWorkflowState.error_sandbox.value
    run.error_message = _truncate_log(str(e), 4000)
    run.completed_at = _now_utc()
    await session.commit()
    log.warning(
        "workflow_error",
        metric_name="workflow_error_total",
        phase="building_env",
        workflow_run_id=str(run_stable_id),
        err=str(e),
    )
    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="env_builder",
        tool_name="build_environment",
        tool_inputs={"repo": repo_slug},
        tool_output=str(e),
    )
    await session.commit()
    await _best_effort_error_slack(
        slack,
        repo.slack_channel,
        title="Environment build failed (permanent)",
        detail=_safe_exc_detail(e),
        workflow_run_id=run_stable_id,
        finding_id=str(finding.id),
    )


async def _sandbox_on_env_unexpected(
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    slack: SlackClient,
    log: Any,
    e: Exception,
) -> None:
    run = await _require_run(session, run_stable_id, missing_message=_MSG_RUN_MISSING_AFTER_ENV_BUILD)
    run.state = AdvisoryWorkflowState.error_unrecoverable.value
    run.error_message = _truncate_log(str(e), 4000)
    run.completed_at = _now_utc()
    await session.commit()
    log.exception(
        "workflow_unrecoverable",
        metric_name="workflow_error_total",
        phase="building_env",
        workflow_run_id=str(run_stable_id),
    )
    await _best_effort_error_slack(
        slack,
        repo.slack_channel,
        title="Environment build failed (unrecoverable)",
        detail=_safe_exc_detail(e),
        workflow_run_id=run_stable_id,
        finding_id=str(finding.id),
    )


async def _sandbox_on_poc_expected_failure(
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    slack: SlackClient,
    log: Any,
    image_tag: str,
    e: BaseException,
) -> None:
    run = await _require_run(session, run_stable_id, missing_message="run missing after sandbox exec failure")
    run.state = AdvisoryWorkflowState.error_sandbox.value
    run.error_message = _truncate_log(str(e), 4000)
    run.completed_at = _now_utc()
    await session.commit()
    log.warning(
        "workflow_error",
        metric_name="workflow_error_total",
        phase="executing_sandbox",
        workflow_run_id=str(run_stable_id),
        err=str(e),
    )
    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="sandbox_executor",
        tool_name="execute_poc",
        tool_inputs={"image": image_tag},
        tool_output=str(e),
    )
    await session.commit()
    await _best_effort_error_slack(
        slack,
        repo.slack_channel,
        title="Sandbox execution failed",
        detail=_safe_exc_detail(e),
        workflow_run_id=run_stable_id,
        finding_id=str(finding.id),
    )


async def _sandbox_on_poc_unexpected(
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    slack: SlackClient,
    log: Any,
    image_tag: str,
    e: Exception,
) -> None:
    run = await _require_run(session, run_stable_id, missing_message="run missing after sandbox exec failure")
    run.state = AdvisoryWorkflowState.error_sandbox.value
    run.error_message = _truncate_log(str(e), 4000)
    run.completed_at = _now_utc()
    await session.commit()
    log.exception(
        "workflow_unexpected_sandbox_error",
        metric_name="workflow_error_total",
        phase="executing_sandbox",
        workflow_run_id=str(run_stable_id),
        err=str(e),
    )
    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="sandbox_executor",
        tool_name="execute_poc",
        tool_inputs={"image": image_tag},
        tool_output=str(e),
    )
    await session.commit()
    await _best_effort_error_slack(
        slack,
        repo.slack_channel,
        title="Sandbox execution failed",
        detail=_safe_exc_detail(e),
        workflow_run_id=run_stable_id,
        finding_id=str(finding.id),
    )


async def _run_sandbox_phase(
    *,
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    scm: SCMProvider,
    slack: SlackClient,
    log: Any,
    work_dir: Path | None,
    container_socket: str,
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None,
) -> ExecutionResult | None:
    """Run env build → sandbox execute → sandbox_complete.

    Returns ``ExecutionResult`` on success, ``None`` on error (run state already
    updated to ``error_sandbox`` or ``error_unrecoverable``).
    """
    owns_work_dir = work_dir is None
    effective_work_dir = work_dir or Path(tempfile.mkdtemp(prefix="scout-"))
    try:
        return await _run_sandbox_phase_inner(
            session=session,
            run_stable_id=run_stable_id,
            finding=finding,
            repo=repo,
            scm=scm,
            slack=slack,
            log=log,
            effective_work_dir=effective_work_dir,
            container_socket=container_socket,
            schedule_retry=schedule_retry,
        )
    finally:
        if owns_work_dir:
            shutil.rmtree(effective_work_dir, ignore_errors=True)


async def _run_sandbox_phase_inner(
    *,
    session: AsyncSession,
    run_stable_id: uuid.UUID,
    finding: Finding,
    repo: RepoConfig,
    scm: SCMProvider,
    slack: SlackClient,
    log: Any,
    effective_work_dir: Path,
    container_socket: str,
    schedule_retry: Callable[[ScheduleRetryParams], Awaitable[None]] | None,
) -> ExecutionResult | None:
    repo_slug = f"{repo.github_org}/{repo.github_repo}".lower()

    await _require_run(session, run_stable_id, missing_message="run missing in sandbox phase")

    clone_ref = _sandbox_resolved_git_ref(finding, repo.default_git_ref)

    try:
        env_result: EnvBuildResult = await build_environment(
            scm,
            repo_slug=repo_slug,
            ref=clone_ref,
            work_dir=effective_work_dir,
            container_socket=container_socket,
        )
    except SecurityScoutError as e:
        await _sandbox_on_env_scout_error(
            session=session,
            run_stable_id=run_stable_id,
            finding=finding,
            repo=repo,
            slack=slack,
            log=log,
            repo_slug=repo_slug,
            e=e,
            schedule_retry=schedule_retry,
        )
        return None
    except Exception as e:
        await _sandbox_on_env_unexpected(session, run_stable_id, finding, repo, slack, log, e)
        return None

    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="env_builder",
        tool_name="build_environment",
        tool_inputs={"repo": repo_slug, "stack": env_result.detected_stack.value},
        tool_output=_truncate_log(f"image={env_result.image_tag}"),
    )
    await session.commit()

    run = await _require_run(session, run_stable_id, missing_message="run missing after env build")
    run.state = AdvisoryWorkflowState.executing_sandbox.value
    await session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.building_env.value,
        to_state=AdvisoryWorkflowState.executing_sandbox.value,
        workflow_run_id=str(run_stable_id),
    )

    poc_content = finding.reproduction or ""
    poc_command = ["python", "-c", poc_content] if poc_content else ["echo", "no PoC"]

    try:
        exec_result = await execute_poc(
            image=env_result.image_tag,
            poc_command=poc_command,
            poc_type=PocType.RESEARCHER_SUBMITTED,
            repo_path=env_result.repo_path,
            container_socket=container_socket,
        )
    except (NotImplementedError, SandboxError) as e:
        await _sandbox_on_poc_expected_failure(
            session,
            run_stable_id,
            finding,
            repo,
            slack,
            log,
            env_result.image_tag,
            e,
        )
        return None
    except Exception as e:
        await _sandbox_on_poc_unexpected(
            session,
            run_stable_id,
            finding,
            repo,
            slack,
            log,
            env_result.image_tag,
            e,
        )
        return None

    await _append_action_log(
        session,
        workflow_run_id=run_stable_id,
        agent="sandbox_executor",
        tool_name="execute_poc",
        tool_inputs={
            "image": env_result.image_tag,
            "poc_type": exec_result.poc_type.value,
        },
        tool_output=_truncate_log(
            f"tier={exec_result.confidence_tier.value} "
            f"exit={exec_result.exit_code} "
            f"elapsed={exec_result.elapsed_seconds:.1f}s"
        ),
    )
    await session.commit()

    log.info(
        "sandbox_execution_complete",
        metric_name="sandbox_execution_seconds",
        duration_seconds=exec_result.elapsed_seconds,
        confidence_tier=exec_result.confidence_tier.value,
        workflow_run_id=str(run_stable_id),
    )

    finding.status = exec_result.confidence_tier
    finding.poc_executed = True
    merged_evidence = dict(finding.evidence or {})
    merged_evidence["execution"] = {
        "excerpt": exec_result.evidence_excerpt,
        "exit_code": exec_result.exit_code,
        "elapsed_seconds": exec_result.elapsed_seconds,
        "poc_type": exec_result.poc_type.value,
        "timed_out": exec_result.timed_out,
    }
    finding.evidence = merged_evidence
    await session.commit()

    run = await _require_run(session, run_stable_id, missing_message="run missing after sandbox exec")
    run.state = AdvisoryWorkflowState.sandbox_complete.value
    await session.commit()
    log.info(
        "workflow_state_transition",
        metric_name="workflow_state_current",
        from_state=AdvisoryWorkflowState.executing_sandbox.value,
        to_state=AdvisoryWorkflowState.sandbox_complete.value,
        workflow_run_id=str(run_stable_id),
    )

    return exec_result

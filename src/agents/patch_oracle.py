# SPDX-License-Identifier: Apache-2.0
"""Human-triggered patch oracle: dual checkout (vulnerable vs patched) PoC run."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agents.env_builder import build_environment
from agents.sandbox_executor import (
    ExecutionResult,
    PocType,
    assign_confidence_tier,
    execute_poc,
    output_matches_success_patterns,
)
from exceptions import SecurityScoutError
from models import AgentActionLog, Finding, FindingStatus, WorkflowRun
from tools.docker_sandbox import SandboxBuildError
from tools.scm.protocol import SCMProvider

_LOG = structlog.get_logger(__name__)

_MAX_LOG = 500


def _truncate_log(text: str | None, max_chars: int = _MAX_LOG) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


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


def _oracle_vulnerable_and_patched_candidates(
    finding: Finding,
    *,
    default_git_ref: str,
) -> tuple[str, list[str]] | None:
    ev = finding.evidence or {}
    oracle = ev.get("oracle")
    if not isinstance(oracle, dict):
        return None
    fall_back = default_git_ref.strip() if default_git_ref.strip() else "main"
    vuln_raw = oracle.get("vulnerable_ref")
    vulnerable_ref = vuln_raw.strip() if isinstance(vuln_raw, str) and vuln_raw.strip() else fall_back
    candidates_raw = oracle.get("patched_ref_candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        return None
    candidates: list[str] = []
    for item in candidates_raw:
        if isinstance(item, str) and item.strip():
            s = item.strip()
            if s not in candidates:
                candidates.append(s)
    if not candidates:
        return None
    return vulnerable_ref, candidates


def _poc_command_and_type(finding: Finding) -> tuple[list[str], PocType]:
    poc_raw = finding.reproduction or ""
    poc_command = ["python", "-c", poc_raw] if poc_raw else ["echo", "no PoC"]
    poc_type = PocType.RESEARCHER_SUBMITTED
    ev = finding.evidence or {}
    ex = ev.get("execution")
    if not isinstance(ex, dict):
        return poc_command, poc_type
    pt = ex.get("poc_type")
    if pt == PocType.NUCLEI_TEMPLATE.value:
        poc_type = PocType.NUCLEI_TEMPLATE
    elif pt == PocType.LLM_GENERATED.value:
        poc_type = PocType.LLM_GENERATED
    return poc_command, poc_type


def _assert_eligible_for_patch_oracle(finding: Finding) -> None:
    if finding.status != FindingStatus.confirmed_low:
        msg = f"patch oracle requires confirmed_low status, got {finding.status.value}"
        raise RuntimeError(msg)
    if not finding.patch_available:
        msg = "patch oracle requires patch_available"
        raise RuntimeError(msg)


async def _load_finding_for_oracle(session: AsyncSession, finding_id: uuid.UUID) -> Finding:
    finding = await session.get(Finding, finding_id)
    if finding is None:
        msg = "finding not found for patch oracle"
        raise RuntimeError(msg)
    return finding


async def _assert_workflow_run_matches_finding(
    session: AsyncSession,
    workflow_run_id: uuid.UUID,
    finding_id: uuid.UUID,
) -> None:
    run = await session.get(WorkflowRun, workflow_run_id)
    if run is None:
        msg = "workflow run not found for patch oracle"
        raise RuntimeError(msg)
    if run.finding_id != finding_id:
        msg = "workflow run does not match finding for patch oracle"
        raise RuntimeError(msg)


async def _run_vulnerable_and_patched_executions(
    scm: SCMProvider,
    *,
    repo_slug: str,
    vulnerable_ref: str,
    patched_candidates: list[str],
    poc_command: list[str],
    poc_type: PocType,
    container_socket: str,
    log: structlog.stdlib.BoundLogger,
) -> tuple[ExecutionResult, str, ExecutionResult]:
    work_root = Path(tempfile.mkdtemp(prefix="scout-oracle-"))
    try:
        vuln_dir = work_root / "vuln"
        vuln_dir.mkdir()
        env_v = await build_environment(
            scm,
            repo_slug=repo_slug,
            ref=vulnerable_ref,
            work_dir=vuln_dir,
            container_socket=container_socket,
        )
        vuln_exec = await execute_poc(
            image=env_v.image_tag,
            poc_command=poc_command,
            poc_type=poc_type,
            repo_path=env_v.repo_path,
            container_socket=container_socket,
        )

        patch_dir = work_root / "patched"
        patched_ref: str | None = None
        pat_exec: ExecutionResult | None = None
        last_attempt_err: str | None = None

        for cand in patched_candidates:
            if patch_dir.exists():
                shutil.rmtree(patch_dir, ignore_errors=True)
            patch_dir.mkdir(parents=True, exist_ok=True)
            try:
                env_p = await build_environment(
                    scm,
                    repo_slug=repo_slug,
                    ref=cand,
                    work_dir=patch_dir,
                    container_socket=container_socket,
                )
            except (SecurityScoutError, SandboxBuildError) as e:
                last_attempt_err = str(e)
                log.warning(
                    "patch_oracle_patched_ref_try_failed",
                    patched_ref=cand,
                    err=last_attempt_err,
                )
                continue
            patched_ref = cand
            pat_exec = await execute_poc(
                image=env_p.image_tag,
                poc_command=poc_command,
                poc_type=poc_type,
                repo_path=env_p.repo_path,
                container_socket=container_socket,
            )
            break

        if pat_exec is None or patched_ref is None:
            msg = f"patch oracle could not build a patched environment at any candidate ref {patched_candidates!r}"
            if last_attempt_err:
                msg += f" (last error: {last_attempt_err})"
            raise RuntimeError(msg)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    assert pat_exec is not None
    assert patched_ref is not None
    return vuln_exec, patched_ref, pat_exec


async def run_patch_oracle_job(
    session: AsyncSession,
    scm: SCMProvider,
    *,
    repo_slug: str,
    finding_id: uuid.UUID,
    workflow_run_id: uuid.UUID,
    container_socket: str,
    default_git_ref: str = "main",
) -> tuple[FindingStatus, str]:
    """Execute patch oracle; persist upgraded tier and merged evidence. Returns (tier, summary)."""
    finding = await _load_finding_for_oracle(session, finding_id)
    await _assert_workflow_run_matches_finding(session, workflow_run_id, finding_id)

    refs = _oracle_vulnerable_and_patched_candidates(finding, default_git_ref=default_git_ref)
    if refs is None:
        msg = "patch oracle metadata missing (patched ref candidates)"
        raise RuntimeError(msg)
    vulnerable_ref, patched_candidates = refs

    _assert_eligible_for_patch_oracle(finding)
    poc_command, poc_type = _poc_command_and_type(finding)

    log = _LOG.bind(
        agent="patch_oracle",
        finding_id=str(finding_id),
        workflow_run_id=str(workflow_run_id),
    )
    log.info(
        "patch_oracle_start",
        vulnerable_ref=vulnerable_ref,
        patched_candidates=patched_candidates,
    )

    vuln_exec, patched_ref, pat_exec = await _run_vulnerable_and_patched_executions(
        scm,
        repo_slug=repo_slug,
        vulnerable_ref=vulnerable_ref,
        patched_candidates=patched_candidates,
        poc_command=poc_command,
        poc_type=poc_type,
        container_socket=container_socket,
        log=log,
    )

    vuln_hit = vuln_exec.exit_code == 0 and output_matches_success_patterns(
        vuln_exec.raw_stdout,
        vuln_exec.raw_stderr,
    )
    pat_hit = pat_exec.exit_code == 0 and output_matches_success_patterns(
        pat_exec.raw_stdout,
        pat_exec.raw_stderr,
    )
    oracle_passed = vuln_hit and not pat_hit

    tier = assign_confidence_tier(
        exit_code=vuln_exec.exit_code,
        stdout=vuln_exec.raw_stdout,
        stderr=vuln_exec.raw_stderr,
        timed_out=vuln_exec.timed_out,
        has_patch_oracle=True,
        patch_oracle_passed=oracle_passed,
        poc_type=poc_type,
    )

    merged = dict(finding.evidence or {})
    merged["patch_oracle"] = {
        "vulnerable_ref": vulnerable_ref,
        "patched_ref": patched_ref,
        "vulnerable_triggers": vuln_hit,
        "patched_triggers": pat_hit,
        "oracle_passed": oracle_passed,
        "vulnerable_exit_code": vuln_exec.exit_code,
        "patched_exit_code": pat_exec.exit_code,
    }
    execution = dict(merged["execution"]) if isinstance(merged.get("execution"), dict) else {}
    execution["excerpt"] = vuln_exec.evidence_excerpt
    execution["exit_code"] = vuln_exec.exit_code
    execution["elapsed_seconds"] = vuln_exec.elapsed_seconds
    execution["poc_type"] = poc_type.value
    execution["timed_out"] = vuln_exec.timed_out
    merged["execution"] = execution

    finding.status = tier
    finding.evidence = merged
    await session.flush()

    await _append_action_log(
        session,
        workflow_run_id=workflow_run_id,
        agent="patch_oracle",
        tool_name="patch_oracle.dual_run",
        tool_inputs={
            "vulnerable_ref": vulnerable_ref,
            "patched_ref": patched_ref,
        },
        tool_output=_truncate_log(
            f"tier={tier.value} vuln_hit={vuln_hit} pat_hit={pat_hit} passed={oracle_passed}",
            max_chars=2000,
        ),
    )
    await session.commit()

    summary = (
        f"Patch oracle complete: tier `{tier.value}` — "
        f"vulnerable output matched={vuln_hit}, patched matched={pat_hit}, "
        f"differential_ok={oracle_passed}."
    )
    log.info("patch_oracle_done", tier=tier.value, oracle_passed=oracle_passed)
    return tier, summary


__all__ = ["run_patch_oracle_job"]

# SPDX-License-Identifier: Apache-2.0
"""Sandbox executor agent.

Runs PoC artifacts inside a hardened container and assigns a confidence tier
to the result.  All execution output is sanitised via ``input_sanitiser``
before it reaches any LLM context.

ALLOWED tools: docker_sandbox.run, docker_sandbox.destroy, nuclei.run,
               input_sanitiser.sanitise
NOT ALLOWED: SCM write, Slack, GitHub API, triage, rate_limiter
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import structlog

from models import FindingStatus
from tools.docker_sandbox import SandboxConfig, SandboxError, run_container
from tools.input_sanitiser import sanitize_text
from tools.nuclei import NUCLEI_WALL_CLOCK_SECONDS, NucleiError, run_nuclei

_LOG = structlog.get_logger(__name__)

_MAX_EVIDENCE_CHARS = 500
_MAX_OUTPUT_BYTES = 50 * 1024
_POC_EXECUTION_WALL_CLOCK_S = 60


class PocType(StrEnum):
    RESEARCHER_SUBMITTED = "researcher-submitted"
    NUCLEI_TEMPLATE = "nuclei-template"
    LLM_GENERATED = "llm-generated"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Outcome of PoC execution with confidence tier."""

    confidence_tier: FindingStatus
    evidence_excerpt: str
    raw_stdout: str
    raw_stderr: str
    elapsed_seconds: float
    poc_type: PocType
    exit_code: int = 0
    timed_out: bool = False
    nuclei_matches: int = 0


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def _sanitise_output(raw: str) -> str:
    return sanitize_text(raw, max_chars=_MAX_OUTPUT_BYTES)


def _extract_evidence(stdout: str, stderr: str) -> str:
    """Extract the most relevant evidence snippet from execution output."""
    combined = stdout + "\n" + stderr if stderr else stdout
    return _truncate(combined.strip(), _MAX_EVIDENCE_CHARS)


_SUCCESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(vulnerable|exploitable|injection\s+successful)", re.IGNORECASE),
    re.compile(r"(rce|remote\s+code\s+execution)", re.IGNORECASE),
    re.compile(r"(sql\s+injection|xss|ssrf)", re.IGNORECASE),
    re.compile(r"(pwned|shell|access\s+granted)", re.IGNORECASE),
    re.compile(r"HTTP/[12]\.\d\s+200", re.IGNORECASE),
]


def _has_expected_pattern(stdout: str, stderr: str) -> bool:
    """Check if output contains patterns indicating successful exploitation."""
    combined = stdout + "\n" + stderr
    return any(p.search(combined) for p in _SUCCESS_PATTERNS)


def output_matches_success_patterns(stdout: str, stderr: str) -> bool:
    """Public helper for patch-oracle differential checks (same patterns as tier assignment)."""
    return _has_expected_pattern(stdout, stderr)


def assign_confidence_tier(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    has_patch_oracle: bool = False,
    patch_oracle_passed: bool = False,
    poc_type: PocType = PocType.RESEARCHER_SUBMITTED,
) -> FindingStatus:
    """Assign a confidence tier based on execution outcome.

    - exit 0 + expected output pattern → CONFIRMED_LOW
    - patch oracle confirms differential → CONFIRMED_HIGH
    - LLM-generated PoCs capped at CONFIRMED_LOW
    - inconclusive → UNCONFIRMED
    - build/timeout/OOM → ERROR
    """
    if timed_out:
        return FindingStatus.error

    if exit_code != 0 and not _has_expected_pattern(stdout, stderr):
        if _is_build_or_infra_error(stderr):
            return FindingStatus.error
        return FindingStatus.unconfirmed

    if exit_code == 0 and _has_expected_pattern(stdout, stderr):
        if has_patch_oracle and patch_oracle_passed:
            if poc_type == PocType.LLM_GENERATED:
                return FindingStatus.confirmed_low
            return FindingStatus.confirmed_high
        return FindingStatus.confirmed_low

    if exit_code == 0:
        return FindingStatus.unconfirmed

    return FindingStatus.unconfirmed


def _is_build_or_infra_error(stderr: str) -> bool:
    """Detect infrastructure failures vs PoC failures."""
    infra_markers = [
        "OOMKilled",
        "oom-kill",
        "out of memory",
        "cannot allocate memory",
        "no space left on device",
        "docker",
        "podman",
        "container",
        "permission denied",
        "exec format error",
    ]
    lower = stderr.lower()
    return any(m.lower() in lower for m in infra_markers)


async def execute_poc(
    *,
    image: str,
    poc_command: list[str],
    poc_type: PocType = PocType.RESEARCHER_SUBMITTED,
    repo_path: Path | None = None,
    env: dict[str, str] | None = None,
    container_socket: str = "unix:///var/run/docker.sock",
) -> ExecutionResult:
    """Execute a PoC script inside a hardened sandbox container.

    Returns an ``ExecutionResult`` with confidence tier and sanitised evidence.
    """
    log = _LOG.bind(agent="sandbox_executor", poc_type=poc_type.value)
    log.info("poc_execution_start", image=image)

    volumes: dict[str, str] = {}
    if repo_path is not None:
        volumes[str(repo_path)] = "/workspace:ro"

    wall_clock_s = _POC_EXECUTION_WALL_CLOCK_S
    config = SandboxConfig(
        image=image,
        command=poc_command,
        env=env or {},
        read_only_volumes=volumes,
        max_run_seconds=wall_clock_s,
        working_dir="/workspace" if repo_path else None,
    )

    try:
        async with asyncio.timeout(wall_clock_s):
            result = await run_container(config, socket=container_socket)
    except TimeoutError:
        log.warning("poc_execution_timeout", wall_clock_seconds=wall_clock_s)
        return ExecutionResult(
            confidence_tier=FindingStatus.error,
            evidence_excerpt="Execution timed out",
            raw_stdout="",
            raw_stderr="",
            elapsed_seconds=float(wall_clock_s),
            poc_type=poc_type,
            exit_code=-1,
            timed_out=True,
        )
    except SandboxError as exc:
        log.warning("poc_execution_sandbox_error", err=str(exc))
        return ExecutionResult(
            confidence_tier=FindingStatus.error,
            evidence_excerpt=f"Sandbox error: {_truncate(str(exc), 200)}",
            raw_stdout="",
            raw_stderr=str(exc),
            elapsed_seconds=0.0,
            poc_type=poc_type,
            exit_code=-1,
        )

    sanitised_stdout = _sanitise_output(result.stdout)
    sanitised_stderr = _sanitise_output(result.stderr)

    tier = assign_confidence_tier(
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        poc_type=poc_type,
    )

    evidence = _extract_evidence(sanitised_stdout, sanitised_stderr)

    log.info(
        "poc_execution_complete",
        confidence_tier=tier.value,
        exit_code=result.exit_code,
        elapsed_seconds=result.elapsed_seconds,
    )

    return ExecutionResult(
        confidence_tier=tier,
        evidence_excerpt=evidence,
        raw_stdout=sanitised_stdout,
        raw_stderr=sanitised_stderr,
        elapsed_seconds=result.elapsed_seconds,
        poc_type=poc_type,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
    )


async def execute_nuclei(
    *,
    target: str,
    cve_id: str | None = None,
    template_paths: list[Path] | None = None,
) -> ExecutionResult:
    """Run Nuclei templates against a target and return structured results."""
    log = _LOG.bind(agent="sandbox_executor", poc_type=PocType.NUCLEI_TEMPLATE.value)
    log.info("nuclei_execution_start", target=target, cve_id=cve_id)

    template_ids = [cve_id] if cve_id else None

    try:
        async with asyncio.timeout(NUCLEI_WALL_CLOCK_SECONDS):
            result = await run_nuclei(
                target=target,
                template_ids=template_ids,
                template_paths=template_paths,
            )
    except TimeoutError:
        log.warning("nuclei_execution_timeout", wall_clock_seconds=NUCLEI_WALL_CLOCK_SECONDS)
        return ExecutionResult(
            confidence_tier=FindingStatus.error,
            evidence_excerpt="Nuclei scan timed out",
            raw_stdout="",
            raw_stderr="",
            elapsed_seconds=float(NUCLEI_WALL_CLOCK_SECONDS),
            poc_type=PocType.NUCLEI_TEMPLATE,
            exit_code=-1,
            timed_out=True,
        )
    except NucleiError as exc:
        log.warning("nuclei_execution_error", err=str(exc))
        return ExecutionResult(
            confidence_tier=FindingStatus.error,
            evidence_excerpt=f"Nuclei error: {_truncate(str(exc), 200)}",
            raw_stdout="",
            raw_stderr=str(exc),
            elapsed_seconds=0.0,
            poc_type=PocType.NUCLEI_TEMPLATE,
            exit_code=-1,
        )
    except Exception as exc:
        log.exception("nuclei_execution_unexpected_error", err=str(exc))
        return ExecutionResult(
            confidence_tier=FindingStatus.error,
            evidence_excerpt=f"Nuclei error: {_truncate(str(exc), 200)}",
            raw_stdout="",
            raw_stderr=str(exc),
            elapsed_seconds=0.0,
            poc_type=PocType.NUCLEI_TEMPLATE,
            exit_code=-1,
        )

    if result.timed_out:
        return ExecutionResult(
            confidence_tier=FindingStatus.error,
            evidence_excerpt="Nuclei scan timed out",
            raw_stdout=result.stdout_raw,
            raw_stderr=result.stderr_raw,
            elapsed_seconds=result.elapsed_seconds,
            poc_type=PocType.NUCLEI_TEMPLATE,
            exit_code=result.exit_code,
            timed_out=True,
        )

    stdout_sanitised = _sanitise_output(result.stdout_raw)
    stderr_sanitised = _sanitise_output(result.stderr_raw)

    if result.matches:
        tier = FindingStatus.confirmed_low
        match_details = [f"{m.template_id} at {m.matched_at}" for m in result.matches[:5]]
        evidence = f"Nuclei matches: {'; '.join(match_details)}"
    else:
        tier = FindingStatus.unconfirmed
        evidence = "No Nuclei matches found"

    evidence = _truncate(evidence, _MAX_EVIDENCE_CHARS)

    log.info(
        "nuclei_execution_complete",
        confidence_tier=tier.value,
        match_count=len(result.matches),
        elapsed_seconds=result.elapsed_seconds,
    )

    return ExecutionResult(
        confidence_tier=tier,
        evidence_excerpt=evidence,
        raw_stdout=stdout_sanitised,
        raw_stderr=stderr_sanitised,
        elapsed_seconds=result.elapsed_seconds,
        poc_type=PocType.NUCLEI_TEMPLATE,
        exit_code=result.exit_code,
        timed_out=False,
        nuclei_matches=len(result.matches),
    )


__all__ = [
    "ExecutionResult",
    "PocType",
    "assign_confidence_tier",
    "execute_nuclei",
    "execute_poc",
    "output_matches_success_patterns",
]

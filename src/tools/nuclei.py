# SPDX-License-Identifier: Apache-2.0
"""Nuclei template runner and result parser.

Runs Nuclei as a subprocess inside the sandbox environment, parses its
JSON-lines output, and sanitises results before they reach any LLM context.

Tool access: called by Sandbox Executor only.  No direct access to SCM, Slack,
or Docker lifecycle APIs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from tools.input_sanitiser import sanitize_text

_LOG = structlog.get_logger(__name__)

NUCLEI_WALL_CLOCK_SECONDS = 120
_TEMPLATE_LIST_WALL_CLOCK_SECONDS = 30
_MAX_OUTPUT_BYTES = 50 * 1024


class NucleiError(Exception):
    """Raised when Nuclei execution fails."""

    def __init__(self, message: str, *, is_transient: bool = False) -> None:
        super().__init__(message)
        self.is_transient = is_transient


@dataclass(frozen=True, slots=True)
class NucleiMatch:
    """A single Nuclei finding from JSON output."""

    template_id: str
    matched_at: str
    matcher_name: str
    severity: str
    extracted_results: list[str] = field(default_factory=list)
    curl_command: str = ""
    raw_json: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NucleiResult:
    """Aggregated result from a Nuclei scan."""

    matches: tuple[NucleiMatch, ...]
    exit_code: int
    stdout_raw: str
    stderr_raw: str
    timed_out: bool
    elapsed_seconds: float


def _truncate(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + "\n[truncated]"


def parse_nuclei_json(raw_output: str) -> tuple[NucleiMatch, ...]:
    """Parse Nuclei JSON-lines output into structured matches."""
    matches: list[NucleiMatch] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        info = obj.get("info", {})
        if not isinstance(info, dict):
            info = {}

        extracted = obj.get("extracted-results", [])
        if not isinstance(extracted, list):
            extracted = []

        matches.append(
            NucleiMatch(
                template_id=str(obj.get("template-id", "")),
                matched_at=str(obj.get("matched-at", "")),
                matcher_name=str(obj.get("matcher-name", "")),
                severity=str(info.get("severity", "unknown")),
                extracted_results=[str(r) for r in extracted],
                curl_command=str(obj.get("curl-command", "")),
                raw_json=obj,
            )
        )
    return tuple(matches)


def sanitise_nuclei_output(raw: str) -> str:
    """Sanitise Nuclei output before embedding in any LLM context."""
    return sanitize_text(raw, max_chars=_MAX_OUTPUT_BYTES)


async def check_template_exists(cve_id: str) -> bool:
    """Check whether a Nuclei template exists for a given CVE ID.

    Runs ``nuclei -tl`` and greps for the CVE ID.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nuclei",
            "-tl",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(_TEMPLATE_LIST_WALL_CLOCK_SECONDS):
                stdout, _ = await proc.communicate()
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            return False
        template_list = stdout.decode("utf-8", errors="replace")
        return cve_id.upper() in template_list.upper()
    except OSError:
        return False


async def run_nuclei(
    *,
    target: str,
    template_ids: list[str] | None = None,
    template_paths: list[Path] | None = None,
    extra_args: list[str] | None = None,
) -> NucleiResult:
    """Run Nuclei against *target* with specified templates.

    Returns structured results with all output sanitised.
    """
    cmd: list[str] = [
        "nuclei",
        "-target",
        target,
        "-jsonl",
        "-no-interactsh",
        "-rate-limit",
        "10",
        "-timeout",
        str(min(NUCLEI_WALL_CLOCK_SECONDS, 30)),
        "-silent",
    ]

    if template_ids:
        for tid in template_ids:
            cmd.extend(["-id", tid])

    if template_paths:
        for tp in template_paths:
            cmd.extend(["-t", str(tp)])

    if extra_args:
        cmd.extend(extra_args)

    _LOG.info("nuclei_start", target=target, template_ids=template_ids)

    import time

    start = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise NucleiError("nuclei binary not found on PATH") from exc

    try:
        stdout_bytes, stderr_bytes = await proc.communicate()
    except asyncio.CancelledError:
        proc.kill()
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(proc.wait(), timeout=5)
        raise

    elapsed = time.monotonic() - start
    timed_out = False
    exit_code = proc.returncode or 0

    stdout_raw = _truncate(stdout_bytes.decode("utf-8", errors="replace"))
    stderr_raw = _truncate(stderr_bytes.decode("utf-8", errors="replace"))

    matches = parse_nuclei_json(stdout_raw)

    _LOG.info(
        "nuclei_complete",
        target=target,
        match_count=len(matches),
        exit_code=exit_code,
        elapsed_seconds=round(elapsed, 2),
        timed_out=timed_out,
    )

    return NucleiResult(
        matches=matches,
        exit_code=exit_code,
        stdout_raw=stdout_raw,
        stderr_raw=stderr_raw,
        timed_out=timed_out,
        elapsed_seconds=round(elapsed, 2),
    )


__all__ = [
    "NUCLEI_WALL_CLOCK_SECONDS",
    "NucleiError",
    "NucleiMatch",
    "NucleiResult",
    "check_template_exists",
    "parse_nuclei_json",
    "run_nuclei",
    "sanitise_nuclei_output",
]

# SPDX-License-Identifier: Apache-2.0
"""Container sandbox lifecycle management.

Uses the ``docker`` Python SDK (``docker-py``) with a configurable socket path
(``CONTAINER_SOCKET`` env var) so the same code works against both Docker and
Podman.  All containers enforce defence-in-depth hardening: ``--cap-drop=all``,
``--network none``, ``--read-only``, ``--pids-limit``, seccomp, resource caps.

The Env Builder and Sandbox Executor agents call these functions via the
Orchestrator.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import structlog

_LOG = structlog.get_logger(__name__)

_DEFAULT_SOCKET = "unix:///var/run/docker.sock"
_BUILD_TIMEOUT_SECONDS = 300
_RUN_TIMEOUT_SECONDS = 60
_MAX_OUTPUT_BYTES = 50 * 1024


class SandboxError(Exception):
    """Base exception for sandbox operations."""

    def __init__(self, message: str, *, is_transient: bool = False) -> None:
        super().__init__(message)
        self.is_transient = is_transient


class SandboxBuildError(SandboxError):
    """Docker image build failed."""


class SandboxTimeoutError(SandboxError):
    """Container exceeded its time limit."""


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    network_mode: str = "none"
    memory_limit: str = "512m"
    cpu_quota: float = 0.5
    pids_limit: int = 50
    max_run_seconds: int = _RUN_TIMEOUT_SECONDS
    seccomp_profile: Path | None = None
    read_only_volumes: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None


@dataclass(frozen=True, slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Outcome of a Docker image build."""

    image_tag: str
    build_log: str


def _truncate_output(raw: bytes, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + "\n[truncated]"


async def build_image(
    dockerfile_path: Path,
    context_path: Path,
    tag: str,
    *,
    socket: str = _DEFAULT_SOCKET,
) -> BuildResult:
    """Build a Docker image from *dockerfile_path* in *context_path*.

    Not implemented yet; callers must fall back or surface a clear error.
    """
    _ = (dockerfile_path, context_path, tag, socket)
    async with asyncio.timeout(_BUILD_TIMEOUT_SECONDS):
        raise NotImplementedError("build_image is not implemented yet")


async def run_container(
    config: SandboxConfig,
    *,
    socket: str = _DEFAULT_SOCKET,
) -> SandboxResult:
    """Run a command inside a hardened, ephemeral container.

    Not implemented yet; callers must fall back or surface a clear error.
    """
    raise NotImplementedError("run_container is not implemented yet")


async def destroy_container(
    container_id: str,
    *,
    socket: str = _DEFAULT_SOCKET,
) -> None:
    """Force-remove a container by ID.

    Not implemented yet; callers must fall back or surface a clear error.
    """
    raise NotImplementedError("destroy_container is not implemented yet")


__all__ = [
    "BuildResult",
    "SandboxBuildError",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "SandboxTimeoutError",
    "build_image",
    "destroy_container",
    "run_container",
]

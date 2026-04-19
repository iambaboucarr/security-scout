# SPDX-License-Identifier: Apache-2.0
"""Container sandbox lifecycle management.

Uses the ``docker`` Python SDK (``docker-py``) with a configurable socket path
(``CONTAINER_SOCKET`` env var) so the same code works against both Docker and
Podman.  All containers enforce defence-in-depth hardening: ``--cap-drop=all``,
``--network none``, ``--read-only``, ``--pids-limit``, seccomp, resource caps.

**Trust boundary:** access to the container API socket is equivalent to high
privilege on the host for most deployments. Callers must only pass
``read_only_volumes`` host paths that are trusted (e.g. orchestrator-controlled
clone directories). The Env Builder and Sandbox Executor call these functions
via the Orchestrator.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import docker
import structlog
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

_LOG = structlog.get_logger(__name__)

_DEFAULT_SOCKET = "unix:///var/run/docker.sock"
_BUILD_TIMEOUT_SECONDS = 300
_RUN_TIMEOUT_SECONDS = 60
_MAX_OUTPUT_BYTES = 50 * 1024
# Writable scratch inside read-only containers (matches Docker/Podman tmpfs mount target).
_TMPFS_MOUNT = "/tmp"  # noqa: S108

_ALLOWED_NETWORK_MODES = frozenset({"none"})


class SandboxError(Exception):
    """Base exception for sandbox operations."""

    def __init__(self, message: str, *, is_transient: bool = False) -> None:
        super().__init__(message)
        self.is_transient = is_transient


class SandboxBuildError(SandboxError):
    """Docker image build failed."""


class SandboxTimeoutError(SandboxError):
    """Reserved for future deadline wrappers; not raised by ``run_container``.

    Inner wall-clock limits return ``SandboxResult`` with ``timed_out=True``.
    """


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    network_mode: str = "none"  # only "none" is allowed (see verify_runtime_hardening)
    memory_limit: str = "512m"
    cpu_quota: float = 0.5
    pids_limit: int = 50
    max_run_seconds: int = _RUN_TIMEOUT_SECONDS
    seccomp_profile: Path | None = None
    # Host path -> container bind (optional ":ro" / ":rw"); host paths must be trusted.
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


def default_seccomp_path() -> Path | None:
    """Return bundled ``sandbox/seccomp.json`` when running from a source checkout."""
    candidate = Path(__file__).resolve().parents[2] / "sandbox" / "seccomp.json"
    return candidate if candidate.is_file() else None


def _resolve_seccomp_path(profile: Path | None) -> Path:
    if profile is not None:
        resolved = profile.expanduser().resolve()
        if not resolved.is_file():
            msg = f"seccomp profile not found: {resolved}"
            raise SandboxError(msg)
        return resolved
    env = os.environ.get("SECURITY_SCOUT_SECCOMP")
    if env:
        resolved = Path(env).expanduser().resolve()
        if resolved.is_file():
            return resolved
        msg = f"SECURITY_SCOUT_SECCOMP does not point to a file: {resolved}"
        raise SandboxError(msg)
    bundled = default_seccomp_path()
    if bundled is not None:
        return bundled
    msg = "seccomp profile required: set SECURITY_SCOUT_SECCOMP or SandboxConfig.seccomp_profile"
    raise SandboxError(msg)


def verify_runtime_hardening(config: SandboxConfig, *, seccomp_path: Path) -> None:
    """Validate hardening policy before creating a container (call site tests)."""
    if config.network_mode not in _ALLOWED_NETWORK_MODES:
        msg = f"network_mode must be one of {_ALLOWED_NETWORK_MODES}, got {config.network_mode!r}"
        raise SandboxError(msg)
    if not config.memory_limit:
        msg = "memory_limit must be set"
        raise SandboxError(msg)
    if config.cpu_quota <= 0:
        msg = "cpu_quota must be positive"
        raise SandboxError(msg)
    if config.pids_limit <= 0:
        msg = "pids_limit must be positive"
        raise SandboxError(msg)
    seccomp_opt = f"seccomp={seccomp_path}"
    security_opt = ["no-new-privileges", seccomp_opt]
    planned = _planned_host_config(config, security_opt)
    if planned["cap_drop"] != ["ALL"]:
        msg = "cap_drop must be ALL"
        raise SandboxError(msg)
    if planned["read_only"] is not True:
        msg = "read_only must be True"
        raise SandboxError(msg)
    if not any(s.startswith("seccomp=") for s in security_opt):
        msg = "seccomp security_opt missing"
        raise SandboxError(msg)


def _planned_host_config(config: SandboxConfig, security_opt: list[str]) -> dict[str, object]:
    """Return the kwargs we will pass to ``containers.create`` (host_config side)."""
    nano_cpus = int(config.cpu_quota * 1_000_000_000)
    return {
        "cap_drop": ["ALL"],
        "network_mode": config.network_mode,
        "mem_limit": config.memory_limit,
        "nano_cpus": nano_cpus,
        "pids_limit": config.pids_limit,
        "read_only": True,
        "security_opt": security_opt,
        "tmpfs": {_TMPFS_MOUNT: "rw,size=64m"},
    }


def _parse_bind_spec(spec: str) -> tuple[str, str]:
    part = spec.rsplit(":", 1)
    if len(part) == 2 and part[1] in ("ro", "rw"):
        return part[0], part[1]
    return spec, "ro"


def _volume_bindings(read_only_volumes: dict[str, str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for host, spec in read_only_volumes.items():
        resolved_host = str(Path(host).resolve())
        bind, mode = _parse_bind_spec(spec)
        out[resolved_host] = {"bind": bind, "mode": mode}
    return out


def _docker_client(socket: str) -> docker.DockerClient:
    try:
        return docker.DockerClient(base_url=socket)
    except DockerException as exc:
        msg = f"docker client init failed: {exc}"
        raise SandboxError(msg, is_transient=True) from exc


def _is_transient_docker_error(exc: BaseException) -> bool:
    if isinstance(exc, (RequestsConnectionError, ReadTimeout)):
        return True
    if isinstance(exc, APIError):
        code = exc.status_code
        return code is not None and code in (500, 502, 503, 504)
    return False


async def build_image(
    dockerfile_path: Path,
    context_path: Path,
    tag: str,
    *,
    socket: str = _DEFAULT_SOCKET,
) -> BuildResult:
    """Build a Docker image from *dockerfile_path* in *context_path*."""
    try:
        async with asyncio.timeout(_BUILD_TIMEOUT_SECONDS):
            return await asyncio.to_thread(_build_image_sync, dockerfile_path, context_path, tag, socket)
    except TimeoutError as exc:
        msg = f"image build exceeded {_BUILD_TIMEOUT_SECONDS}s"
        raise SandboxBuildError(msg, is_transient=True) from exc


def _build_image_sync(
    dockerfile_path: Path,
    context_path: Path,
    tag: str,
    socket: str,
) -> BuildResult:
    ctx = context_path.resolve()
    df = dockerfile_path.resolve()
    try:
        rel_dockerfile = df.relative_to(ctx)
    except ValueError as exc:
        msg = "Dockerfile must be inside the build context directory tree"
        raise SandboxBuildError(msg) from exc

    try:
        client = _docker_client(socket)
    except SandboxError as exc:
        raise SandboxBuildError(str(exc), is_transient=exc.is_transient) from exc
    log_lines: list[str] = []
    try:
        build_stream = client.api.build(
            path=str(ctx),
            dockerfile=str(rel_dockerfile).replace("\\", "/"),
            tag=tag,
            rm=True,
            forcerm=True,
            timeout=_BUILD_TIMEOUT_SECONDS,
            decode=True,
        )
        for chunk in build_stream:
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk:
                msg = str(chunk["error"])
                raise SandboxBuildError(msg)
            chunk_stream = chunk.get("stream")
            if chunk_stream:
                log_lines.append(str(chunk_stream).rstrip())
            message = chunk.get("message")
            if message:
                log_lines.append(str(message).rstrip())
    except APIError as exc:
        msg = f"docker build API error: {exc}"
        raise SandboxBuildError(msg, is_transient=_is_transient_docker_error(exc)) from exc
    except (RequestsConnectionError, ReadTimeout) as exc:
        msg = f"docker build connection error: {exc}"
        raise SandboxBuildError(msg, is_transient=True) from exc
    finally:
        client.close()

    raw_log = "\n".join(log_lines)
    if len(raw_log) > _MAX_OUTPUT_BYTES:
        raw_log = raw_log[:_MAX_OUTPUT_BYTES] + "\n[truncated]"
    return BuildResult(image_tag=tag, build_log=raw_log)


async def run_container(
    config: SandboxConfig,
    *,
    socket: str = _DEFAULT_SOCKET,
) -> SandboxResult:
    """Run a command inside a hardened, ephemeral container."""
    seccomp_path = _resolve_seccomp_path(config.seccomp_profile)
    verify_runtime_hardening(config, seccomp_path=seccomp_path)
    return await asyncio.to_thread(_run_container_sync, config, socket, seccomp_path)


def _run_container_sync(config: SandboxConfig, socket: str, seccomp_path: Path) -> SandboxResult:
    seccomp_opt = f"seccomp={seccomp_path}"
    security_opt = ["no-new-privileges", seccomp_opt]

    volumes = _volume_bindings(config.read_only_volumes)
    nano_cpus = int(config.cpu_quota * 1_000_000_000)

    client = _docker_client(socket)
    started = time.perf_counter()
    container = None
    timed_out = False
    exit_code = -1
    try:
        try:
            container = client.containers.create(
                image=config.image,
                command=config.command,
                environment=config.env,
                user="1000:1000",
                working_dir=config.working_dir,
                network_mode=config.network_mode,
                mem_limit=config.memory_limit,
                nano_cpus=nano_cpus,
                pids_limit=config.pids_limit,
                cap_drop=["ALL"],
                security_opt=security_opt,
                read_only=True,
                tmpfs={_TMPFS_MOUNT: "rw,size=64m"},
                volumes=volumes,
            )
        except ImageNotFound as exc:
            msg = f"image not found: {config.image}"
            raise SandboxError(msg) from exc
        except APIError as exc:
            msg = f"docker create failed: {exc}"
            raise SandboxError(msg, is_transient=_is_transient_docker_error(exc)) from exc

        container.start()
        try:
            wait_result = container.wait(timeout=config.max_run_seconds)
        except ReadTimeout:
            timed_out = True
            try:
                container.kill()
            except APIError:
                _LOG.warning("sandbox_kill_after_timeout_failed", container=container.id)
            wait_result = {"StatusCode": 137}
        except APIError as exc:
            msg = f"docker wait failed: {exc}"
            raise SandboxError(msg, is_transient=_is_transient_docker_error(exc)) from exc

        exit_code = int(wait_result.get("StatusCode", -1))
        out_d: bytes
        err_d: bytes
        out_d, err_d = container.logs(stdout=True, stderr=True, tail="all", demux=True)
        stdout_s = _truncate_output(out_d or b"")
        stderr_s = _truncate_output(err_d or b"")
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except APIError as exc:
                _LOG.warning("sandbox_container_remove_failed", err=str(exc), container=getattr(container, "id", None))
        client.close()

    elapsed = time.perf_counter() - started
    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout_s,
        stderr=stderr_s,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
    )


async def destroy_container(
    container_id: str,
    *,
    socket: str = _DEFAULT_SOCKET,
) -> None:
    """Force-remove a container by ID."""

    def _sync() -> None:
        client = _docker_client(socket)
        try:
            try:
                c = client.containers.get(container_id)
            except NotFound:
                return
            c.remove(force=True)
        except APIError as exc:
            msg = f"docker remove failed: {exc}"
            raise SandboxError(msg, is_transient=_is_transient_docker_error(exc)) from exc
        finally:
            client.close()

    await asyncio.to_thread(_sync)


__all__ = [
    "BuildResult",
    "SandboxBuildError",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "SandboxTimeoutError",
    "build_image",
    "default_seccomp_path",
    "destroy_container",
    "run_container",
    "verify_runtime_hardening",
]

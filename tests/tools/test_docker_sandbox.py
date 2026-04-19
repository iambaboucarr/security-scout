# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from docker.errors import APIError, ImageNotFound, NotFound
from requests.exceptions import ReadTimeout

from tools import docker_sandbox as docker_sandbox_module
from tools.docker_sandbox import (
    SandboxBuildError,
    SandboxConfig,
    SandboxError,
    _is_transient_docker_error,
    build_image,
    default_seccomp_path,
    destroy_container,
    run_container,
    verify_runtime_hardening,
)


def _minimal_config(**kwargs: Any) -> SandboxConfig:
    defaults: dict[str, Any] = {
        "image": "alpine:3",
        "command": ["echo", "hi"],
    }
    defaults.update(kwargs)
    return SandboxConfig(**defaults)


def test_default_seccomp_path_when_present() -> None:
    p = default_seccomp_path()
    assert p is not None
    assert p.name == "seccomp.json"


def test_verify_runtime_hardening_rejects_bad_network(tmp_path: Path) -> None:
    cfg = _minimal_config(network_mode="bridge")
    sec = tmp_path / "seccomp.json"
    sec.write_text("{}")
    with pytest.raises(SandboxError, match="network_mode"):
        verify_runtime_hardening(cfg, seccomp_path=sec)


def test_verify_runtime_hardening_accepts_none_network(tmp_path: Path) -> None:
    cfg = _minimal_config()
    sec = tmp_path / "seccomp.json"
    sec.write_text("{}")
    verify_runtime_hardening(cfg, seccomp_path=sec)


def test_verify_runtime_hardening_rejects_zero_cpu(tmp_path: Path) -> None:
    sec = tmp_path / "seccomp.json"
    sec.write_text("{}")
    cfg = _minimal_config(cpu_quota=0.0)
    with pytest.raises(SandboxError, match="cpu_quota"):
        verify_runtime_hardening(cfg, seccomp_path=sec)


def test_verify_runtime_hardening_rejects_zero_pids(tmp_path: Path) -> None:
    sec = tmp_path / "seccomp.json"
    sec.write_text("{}")
    cfg = _minimal_config(pids_limit=0)
    with pytest.raises(SandboxError, match="pids_limit"):
        verify_runtime_hardening(cfg, seccomp_path=sec)


def test_verify_runtime_hardening_rejects_empty_memory_limit(tmp_path: Path) -> None:
    sec = tmp_path / "seccomp.json"
    sec.write_text("{}")
    cfg = _minimal_config(memory_limit="")
    with pytest.raises(SandboxError, match="memory_limit"):
        verify_runtime_hardening(cfg, seccomp_path=sec)


def test_truncate_output_limits_length() -> None:
    raw = b"a" * (_docker_sandbox_max_bytes() + 100)
    out = docker_sandbox_module._truncate_output(raw)
    assert "[truncated]" in out
    assert len(out) <= _docker_sandbox_max_bytes() + 20


def _docker_sandbox_max_bytes() -> int:
    return docker_sandbox_module._MAX_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_volume_mount_without_explicit_mode_defaults_to_ro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sec = tmp_path / "sec.json"
    sec.write_text("{}")

    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = (b"", b"")

    mock_cc = MagicMock()
    mock_cc.create.return_value = mock_container
    mock_client = MagicMock()
    mock_client.containers = mock_cc
    mock_client.close = MagicMock()
    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    cfg = _minimal_config(seccomp_profile=sec, read_only_volumes={str(tmp_path): "/data"})
    await run_container(cfg, socket="unix:///var/run/docker.sock")
    vols = mock_cc.create.call_args.kwargs["volumes"]
    host_key = str(Path(tmp_path).resolve())
    assert vols[host_key]["bind"] == "/data"
    assert vols[host_key]["mode"] == "ro"


@pytest.mark.asyncio
async def test_run_container_create_includes_hardening(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sec = tmp_path / "sec.json"
    sec.write_text("{}")

    created_kwargs: dict = {}

    mock_container = MagicMock()
    mock_container.id = "abc123"
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = (b"out\n", b"err\n")

    mock_cc = MagicMock()
    mock_cc.create = MagicMock(return_value=mock_container)

    mock_client = MagicMock()
    mock_client.containers = mock_cc
    mock_client.close = MagicMock()

    def _fake_client(_socket: str) -> MagicMock:
        return mock_client

    def _capture_create(*args: object, **kwargs: object) -> MagicMock:
        created_kwargs.update(kwargs)
        return mock_container

    mock_cc.create.side_effect = _capture_create

    monkeypatch.setattr("tools.docker_sandbox._docker_client", _fake_client)

    cfg = _minimal_config(seccomp_profile=sec, read_only_volumes={str(tmp_path): "/workspace:ro"})
    result = await run_container(cfg, socket="unix:///var/run/docker.sock")

    assert result.exit_code == 0
    assert result.timed_out is False
    assert "out" in result.stdout
    assert mock_cc.create.call_count == 1
    assert created_kwargs["cap_drop"] == ["ALL"]
    assert created_kwargs["network_mode"] == "none"
    assert created_kwargs["read_only"] is True
    assert created_kwargs["pids_limit"] == 50
    assert created_kwargs["mem_limit"] == "512m"
    assert any(str(s).startswith("seccomp=") for s in created_kwargs["security_opt"])
    assert "no-new-privileges" in created_kwargs["security_opt"]
    mock_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_run_container_image_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sec = tmp_path / "sec.json"
    sec.write_text("{}")

    mock_cc = MagicMock()
    mock_cc.create.side_effect = ImageNotFound("nope")

    mock_client = MagicMock()
    mock_client.containers = mock_cc
    mock_client.close = MagicMock()

    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    cfg = _minimal_config(seccomp_profile=sec)
    with pytest.raises(SandboxError, match="image not found"):
        await run_container(cfg, socket="unix:///var/run/docker.sock")


@pytest.mark.asyncio
async def test_run_container_wait_timeout_kills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sec = tmp_path / "sec.json"
    sec.write_text("{}")

    mock_container = MagicMock()
    mock_container.wait.side_effect = ReadTimeout("timeout")
    mock_container.logs.return_value = (b"", b"")

    mock_cc = MagicMock()
    mock_cc.create.return_value = mock_container

    mock_client = MagicMock()
    mock_client.containers = mock_cc
    mock_client.close = MagicMock()

    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    cfg = _minimal_config(seccomp_profile=sec, max_run_seconds=30)
    result = await run_container(cfg, socket="unix:///var/run/docker.sock")

    assert result.timed_out is True
    mock_container.kill.assert_called_once()


@pytest.mark.asyncio
async def test_destroy_container_noop_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.containers.get.side_effect = NotFound("missing")

    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    await destroy_container("deadbeef", socket="unix:///var/run/docker.sock")
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_build_image_success_collects_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    ctx = tmp_path

    mock_client = MagicMock()
    mock_client.api.build.return_value = iter([{"stream": "Step 1\n"}, {"message": "done"}])
    mock_client.close = MagicMock()

    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    result = await build_image(df, ctx, "mytag:1", socket="unix:///var/run/docker.sock")
    assert result.image_tag == "mytag:1"
    assert "Step 1" in result.build_log
    assert "done" in result.build_log


@pytest.mark.asyncio
async def test_run_container_raises_when_explicit_seccomp_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    cfg = _minimal_config(seccomp_profile=missing)
    with pytest.raises(SandboxError, match="seccomp profile not found"):
        await run_container(cfg, socket="unix:///var/run/docker.sock")


@pytest.mark.asyncio
async def test_run_container_raises_when_env_seccomp_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECURITY_SCOUT_SECCOMP", str(tmp_path / "does_not_exist.json"))
    monkeypatch.setattr("tools.docker_sandbox.default_seccomp_path", lambda: None)
    cfg = _minimal_config(seccomp_profile=None)
    with pytest.raises(SandboxError, match="SECURITY_SCOUT_SECCOMP"):
        await run_container(cfg, socket="unix:///var/run/docker.sock")


@pytest.mark.asyncio
async def test_run_container_requires_seccomp_when_unbundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SECURITY_SCOUT_SECCOMP", raising=False)
    monkeypatch.setattr("tools.docker_sandbox.default_seccomp_path", lambda: None)
    cfg = _minimal_config(seccomp_profile=None)
    with pytest.raises(SandboxError, match="seccomp profile required"):
        await run_container(cfg, socket="unix:///var/run/docker.sock")


@pytest.mark.asyncio
async def test_run_container_uses_security_scout_seccomp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sec = tmp_path / "profile.json"
    sec.write_text("{}")
    monkeypatch.setenv("SECURITY_SCOUT_SECCOMP", str(sec))
    monkeypatch.setattr("tools.docker_sandbox.default_seccomp_path", lambda: None)

    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = (b"", b"")

    mock_cc = MagicMock()
    mock_cc.create.return_value = mock_container
    mock_client = MagicMock()
    mock_client.containers = mock_cc
    mock_client.close = MagicMock()
    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    cfg = _minimal_config(seccomp_profile=None)
    await run_container(cfg, socket="unix:///var/run/docker.sock")
    assert mock_cc.create.called


@pytest.mark.asyncio
async def test_destroy_container_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_c = MagicMock()
    mock_c.remove.side_effect = APIError("fail", response=MagicMock(status_code=500))
    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_c
    mock_client.close = MagicMock()
    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    with pytest.raises(SandboxError, match="docker remove failed"):
        await destroy_container("abc", socket="unix:///var/run/docker.sock")


@pytest.mark.asyncio
async def test_destroy_container_removes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_c = MagicMock()
    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_c
    mock_client.close = MagicMock()
    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    await destroy_container("abc", socket="unix:///var/run/docker.sock")
    mock_c.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_build_image_streams_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM scratch\n")
    ctx = tmp_path

    mock_client = MagicMock()
    mock_client.api.build.return_value = iter([{"error": "boom"}])
    mock_client.close = MagicMock()

    monkeypatch.setattr("tools.docker_sandbox._docker_client", lambda _s: mock_client)

    with pytest.raises(SandboxBuildError, match="boom"):
        await build_image(df, ctx, "t:1", socket="unix:///var/run/docker.sock")


@pytest.mark.asyncio
async def test_build_image_rejects_dockerfile_outside_context(tmp_path: Path) -> None:
    outer = tmp_path / "outside"
    outer.mkdir()
    df = outer / "Dockerfile"
    df.write_text("FROM scratch\n")
    ctx = tmp_path / "ctx"
    ctx.mkdir()

    with pytest.raises(SandboxBuildError, match="inside the build context"):
        await build_image(df, ctx, "t:1", socket="unix:///var/run/docker.sock")


def test_api_error_transient_classification() -> None:
    resp = MagicMock()
    resp.status_code = 503
    exc = APIError("x", response=resp)
    assert _is_transient_docker_error(exc) is True

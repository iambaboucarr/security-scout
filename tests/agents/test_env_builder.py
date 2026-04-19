# SPDX-License-Identifier: Apache-2.0
"""Tests for agents/env_builder.py — environment build agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.env_builder import (
    DetectedStack,
    EnvBuildResult,
    build_environment,
    detect_stack,
)
from exceptions import TransientError
from tools.scm.protocol import SCMProvider

# ---------------------------------------------------------------------------
# detect_stack
# ---------------------------------------------------------------------------


class TestDetectStack:
    def test_dockerfile(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").touch()
        assert detect_stack(tmp_path) == DetectedStack.DOCKERFILE

    def test_docker_compose_yml(self, tmp_path: Path) -> None:
        (tmp_path / "docker-compose.yml").touch()
        assert detect_stack(tmp_path) == DetectedStack.DOCKER_COMPOSE

    def test_docker_compose_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "docker-compose.yaml").touch()
        assert detect_stack(tmp_path) == DetectedStack.DOCKER_COMPOSE

    def test_compose_yml(self, tmp_path: Path) -> None:
        (tmp_path / "compose.yml").touch()
        assert detect_stack(tmp_path) == DetectedStack.DOCKER_COMPOSE

    def test_python_requirements(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").touch()
        assert detect_stack(tmp_path) == DetectedStack.PYTHON

    def test_python_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        assert detect_stack(tmp_path) == DetectedStack.PYTHON

    def test_node_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").touch()
        assert detect_stack(tmp_path) == DetectedStack.NODE

    def test_go_mod(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").touch()
        assert detect_stack(tmp_path) == DetectedStack.GO

    def test_java_maven(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").touch()
        assert detect_stack(tmp_path) == DetectedStack.JAVA_MAVEN

    def test_java_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").touch()
        assert detect_stack(tmp_path) == DetectedStack.JAVA_GRADLE

    def test_java_gradle_kts(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle.kts").touch()
        assert detect_stack(tmp_path) == DetectedStack.JAVA_GRADLE

    def test_ruby_gemfile(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").touch()
        assert detect_stack(tmp_path) == DetectedStack.RUBY

    def test_unknown_empty_dir(self, tmp_path: Path) -> None:
        assert detect_stack(tmp_path) == DetectedStack.UNKNOWN

    def test_priority_docker_compose_over_dockerfile(self, tmp_path: Path) -> None:
        (tmp_path / "docker-compose.yml").touch()
        (tmp_path / "Dockerfile").touch()
        assert detect_stack(tmp_path) == DetectedStack.DOCKER_COMPOSE

    def test_priority_dockerfile_over_python(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").touch()
        (tmp_path / "requirements.txt").touch()
        assert detect_stack(tmp_path) == DetectedStack.DOCKERFILE


# ---------------------------------------------------------------------------
# build_environment
# ---------------------------------------------------------------------------


def _make_scm(clone_result: Path | Exception) -> SCMProvider:
    scm = MagicMock(spec=SCMProvider)
    if isinstance(clone_result, Exception):
        scm.clone_repo = AsyncMock(side_effect=clone_result)
    else:
        scm.clone_repo = AsyncMock(return_value=clone_result)
    return scm


@pytest.mark.asyncio
async def test_build_environment_no_dockerfile_uses_sandbox_image(tmp_path: Path) -> None:
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    (repo_dir / "requirements.txt").touch()

    scm = _make_scm(repo_dir)
    result = await build_environment(
        scm,
        repo_slug="org/myrepo",
        ref="main",
        work_dir=tmp_path,
    )

    assert result.image_tag == "securityscout/sandbox:latest"
    assert result.detected_stack == DetectedStack.PYTHON
    assert result.repo_path == repo_dir
    assert "no Dockerfile" in result.build_log


@pytest.mark.asyncio
async def test_build_environment_with_dockerfile_falls_back_to_sandbox(tmp_path: Path) -> None:
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    (repo_dir / "Dockerfile").write_text("FROM python:3.14")

    scm = _make_scm(repo_dir)
    result = await build_environment(
        scm,
        repo_slug="org/myrepo",
        ref="v1.0.0",
        work_dir=tmp_path,
    )
    assert result.image_tag == "securityscout/sandbox:latest"
    assert "not yet implemented" in result.build_log


@pytest.mark.asyncio
async def test_build_environment_clone_transient_error(tmp_path: Path) -> None:
    scm = _make_scm(TransientError("network timeout"))
    with pytest.raises(TransientError, match="network timeout"):
        await build_environment(
            scm,
            repo_slug="org/repo",
            ref="main",
            work_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_build_environment_clone_generic_error_wraps_transient(tmp_path: Path) -> None:
    scm = _make_scm(OSError("disk full"))
    with pytest.raises(TransientError, match="clone failed"):
        await build_environment(
            scm,
            repo_slug="org/repo",
            ref="main",
            work_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_build_environment_custom_sandbox_image(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    scm = _make_scm(repo_dir)
    result = await build_environment(
        scm,
        repo_slug="org/repo",
        ref="main",
        work_dir=tmp_path,
        sandbox_image="custom:v1",
    )
    assert result.image_tag == "custom:v1"


@pytest.mark.asyncio
async def test_env_build_result_frozen() -> None:
    r = EnvBuildResult(
        image_tag="test:latest",
        repo_path=Path("/tmp/test"),
        detected_stack=DetectedStack.PYTHON,
        build_log="ok",
    )
    with pytest.raises(AttributeError):
        r.image_tag = "changed"  # type: ignore[misc]


class TestDetectedStackValues:
    def test_all_stacks_are_strings(self) -> None:
        for s in DetectedStack:
            assert isinstance(s.value, str)

    def test_unknown_is_last(self) -> None:
        assert DetectedStack.UNKNOWN.value == "unknown"

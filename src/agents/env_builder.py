# SPDX-License-Identifier: Apache-2.0
"""Environment builder agent.

Clones the target repository at the reported vulnerable version, detects the
project stack deterministically, and builds a Docker image for the target
service.  Uses Haiku model tier only when deterministic stack detection is
insufficient (not yet implemented — deterministic detection covers common
cases).

ALLOWED tools: scm.clone_repo, docker_sandbox.build_image
NOT ALLOWED: Slack, Nuclei, input_sanitiser write, GitHub write, rate_limiter
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import structlog

from exceptions import PermanentError, SecurityScoutError, TransientError
from tools.docker_sandbox import SandboxBuildError, build_image
from tools.scm.protocol import SCMProvider

_LOG = structlog.get_logger(__name__)


class DetectedStack(StrEnum):
    DOCKER_COMPOSE = "docker-compose"
    DOCKERFILE = "dockerfile"
    PYTHON = "python"
    NODE = "node"
    GO = "go"
    JAVA_MAVEN = "java-maven"
    JAVA_GRADLE = "java-gradle"
    RUBY = "ruby"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EnvBuildResult:
    """Outcome of environment preparation."""

    image_tag: str
    repo_path: Path
    detected_stack: DetectedStack
    build_log: str


_STACK_MARKERS: list[tuple[str, DetectedStack]] = [
    ("docker-compose.yml", DetectedStack.DOCKER_COMPOSE),
    ("docker-compose.yaml", DetectedStack.DOCKER_COMPOSE),
    ("compose.yml", DetectedStack.DOCKER_COMPOSE),
    ("compose.yaml", DetectedStack.DOCKER_COMPOSE),
    ("Dockerfile", DetectedStack.DOCKERFILE),
    ("requirements.txt", DetectedStack.PYTHON),
    ("pyproject.toml", DetectedStack.PYTHON),
    ("setup.py", DetectedStack.PYTHON),
    ("Pipfile", DetectedStack.PYTHON),
    ("package.json", DetectedStack.NODE),
    ("go.mod", DetectedStack.GO),
    ("pom.xml", DetectedStack.JAVA_MAVEN),
    ("build.gradle", DetectedStack.JAVA_GRADLE),
    ("build.gradle.kts", DetectedStack.JAVA_GRADLE),
    ("Gemfile", DetectedStack.RUBY),
]


def detect_stack(repo_path: Path) -> DetectedStack:
    """Identify project stack by presence of marker files.

    Returns the first match in priority order (Docker-based stacks first).
    """
    for marker_file, stack in _STACK_MARKERS:
        if (repo_path / marker_file).exists():
            return stack
    return DetectedStack.UNKNOWN


def _find_dockerfile(repo_path: Path) -> Path | None:
    """Locate the most appropriate Dockerfile in the repo."""
    candidates = [
        repo_path / "Dockerfile",
        repo_path / "docker" / "Dockerfile",
        repo_path / ".docker" / "Dockerfile",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


async def build_environment(
    scm: SCMProvider,
    *,
    repo_slug: str,
    ref: str,
    work_dir: Path,
    container_socket: str = "unix:///var/run/docker.sock",
    sandbox_image: str = "securityscout/sandbox:latest",
) -> EnvBuildResult:
    """Clone *repo_slug* at *ref* and build a Docker image for it.

    If the repo has its own Dockerfile, use that.  Otherwise, use the
    default sandbox image (no custom build needed — the sandbox executor
    runs the PoC inside the standard sandbox image with the repo mounted).
    """
    log = _LOG.bind(agent="env_builder", repo=repo_slug, ref=ref)

    log.info("env_build_start")
    try:
        repo_path = await scm.clone_repo(repo_slug, ref, work_dir)
    except SecurityScoutError:
        raise
    except Exception as exc:
        msg = f"clone failed for {repo_slug}@{ref}: {exc}"
        raise TransientError(msg) from exc

    stack = detect_stack(repo_path)
    log.info("stack_detected", stack=stack.value)

    dockerfile = _find_dockerfile(repo_path)
    if dockerfile is not None:
        tag = f"scout-target-{repo_slug.replace('/', '-').lower()}:{ref[:12]}"
        try:
            result = await build_image(
                dockerfile,
                repo_path,
                tag,
                socket=container_socket,
            )
        except SandboxBuildError:
            raise
        except Exception as exc:
            # Non-SandboxBuildError: unexpected docker-py or asyncio.to_thread failure.
            msg = f"image build failed for {repo_slug}@{ref}: {exc}"
            raise PermanentError(msg) from exc

        log.info("image_built", tag=result.image_tag)
        return EnvBuildResult(
            image_tag=result.image_tag,
            repo_path=repo_path,
            detected_stack=stack,
            build_log=result.build_log,
        )

    log.info("no_dockerfile_using_sandbox_image", sandbox_image=sandbox_image)
    return EnvBuildResult(
        image_tag=sandbox_image,
        repo_path=repo_path,
        detected_stack=stack,
        build_log="no Dockerfile found; using default sandbox image",
    )


__all__ = [
    "DetectedStack",
    "EnvBuildResult",
    "build_environment",
    "detect_stack",
]

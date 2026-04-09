from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REPOS_PATH = _REPO_ROOT / "repos.yaml"


class RepoMode(StrEnum):
    observe = "observe"
    comment = "comment"
    enforce = "enforce"


class RateLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pr_comments_per_hour: int = 20
    check_runs_per_hour: int = 10
    workflow_triggers_per_hour: int = 5
    slack_findings_per_hour: int = 30


class DockerBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: str = "."
    file: str = "Dockerfile"
    compose_file: str | None = None


class GitHubIssuesTrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["github_issues"] = "github_issues"
    security_label: str = "security"
    search_closed: bool = True


class JiraTrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["jira"] = "jira"
    project_key: str
    base_url: str


class LinearTrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["linear"] = "linear"
    team_id: str


IssueTrackerEntry = Annotated[
    GitHubIssuesTrackerConfig | JiraTrackerConfig | LinearTrackerConfig,
    Field(discriminator="type"),
]


class RepoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    github_org: str
    github_repo: str
    mode: RepoMode = RepoMode.observe
    slack_channel: str
    allowed_workflows: list[str]
    semgrep_rulesets: list[str] = Field(default_factory=list)
    docker_build: DockerBuildConfig | None = None
    notify_on_severity: list[str]
    require_approval_for: list[str]
    rate_limits: RateLimits | None = None
    issue_trackers: list[IssueTrackerEntry] = Field(default_factory=list)
    dedup_semantic_search: bool = False


class ReposManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repos: list[RepoConfig]

    @model_validator(mode="after")
    def unique_repo_names(self) -> Self:
        names = [r.name for r in self.repos]
        if len(names) != len(set(names)):
            msg = "repos.yaml: duplicate repo name"
            raise ValueError(msg)
        keys = {(r.github_org, r.github_repo) for r in self.repos}
        if len(keys) != len(self.repos):
            msg = "repos.yaml: duplicate github_org/github_repo pair"
            raise ValueError(msg)
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    github_webhook_secret: str
    github_pat: str
    slack_bot_token: str
    slack_signing_secret: str

    database_url: str = "sqlite+aiosqlite:///./security_scout.db"
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"

    repos_config_path: Path = Field(default=_DEFAULT_REPOS_PATH)

    mechanical_model: str = "claude-haiku-4-5"
    reasoning_model: str = "claude-sonnet-4-6"
    high_stakes_model: str = "claude-opus-4-6"


@dataclass(frozen=True, slots=True)
class AppConfig:
    settings: Settings
    repos: ReposManifest
    repos_yaml_sha256: str
    repos_yaml_path: Path


def compute_repos_yaml_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _coerce_manifest_payload(data: Any) -> Mapping[str, Any]:
    if data is None:
        msg = "repos.yaml is empty or not a mapping"
        raise ValueError(msg)
    if not isinstance(data, Mapping):
        msg = "repos.yaml root must be a mapping with a 'repos' key"
        raise TypeError(msg)
    return data


def load_repos_manifest(path: Path) -> tuple[ReposManifest, str]:
    raw = path.read_bytes()
    digest = compute_repos_yaml_sha256(raw)
    data = yaml.safe_load(raw)
    payload = _coerce_manifest_payload(data)
    manifest = ReposManifest.model_validate(payload)
    return manifest, digest


def load_app_config(settings: Settings | None = None) -> AppConfig:
    cfg = settings or Settings()
    path = cfg.repos_config_path
    if not path.is_file():
        msg = f"repos manifest not found: {path}"
        raise FileNotFoundError(msg)
    manifest, digest = load_repos_manifest(path)
    return AppConfig(
        settings=cfg,
        repos=manifest,
        repos_yaml_sha256=digest,
        repos_yaml_path=path.resolve(),
    )


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def log_config_loaded(app: AppConfig) -> None:
    # Pair with `db.log_and_persist_config_loaded` when a DB session exists (ADR-015 Guardrail 6).
    log = structlog.get_logger(__name__)
    log.info(
        "config_loaded",
        metric_name="config_loaded",
        repos_yaml_sha256=app.repos_yaml_sha256,
        repos_config_path=str(app.repos_yaml_path),
        repo_count=len(app.repos.repos),
    )

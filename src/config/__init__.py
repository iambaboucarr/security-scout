# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from models import Severity, SSVCAction

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_REPOS_PATH = _REPO_ROOT / "repos.yaml"


class RepoMode(StrEnum):
    observe = "observe"
    comment = "comment"
    enforce = "enforce"


class AdvisoryPollInterval(StrEnum):
    disabled = "disabled"
    every_5_min = "every_5_min"
    every_15_min = "every_15_min"
    hourly = "hourly"
    every_4_hours = "every_4_hours"
    daily = "daily"


_ADVISORY_POLL_INTERVAL_SECONDS: dict[AdvisoryPollInterval, int] = {
    AdvisoryPollInterval.every_5_min: 300,
    AdvisoryPollInterval.every_15_min: 900,
    AdvisoryPollInterval.hourly: 3600,
    AdvisoryPollInterval.every_4_hours: 14_400,
    AdvisoryPollInterval.daily: 86_400,
}

# ``arq.cron`` uses ``int`` (single), ``set[int]`` (allowlist), or ``None`` (any) for
# ``minute=`` / ``hour=`` (see arq `CronJob` / `next_cron`).


def advisory_poll_cron_minute_and_hour(
    preset: AdvisoryPollInterval,
) -> tuple[int | set[int] | None, int | set[int] | None] | None:
    """Return ``(minute, hour)`` for :func:`arq.cron.cron`, or ``None`` when not scheduled.

    All presets are evaluated in the worker's configured timezone (use ``WorkerSettings`` with
    :obj:`datetime.UTC` for the advisory sync tick).
    """
    if preset == AdvisoryPollInterval.disabled:
        return None
    if preset == AdvisoryPollInterval.every_5_min:
        return (set(range(0, 60, 5)), None)
    if preset == AdvisoryPollInterval.every_15_min:
        return ({0, 15, 30, 45}, None)
    if preset == AdvisoryPollInterval.hourly:
        return (0, None)
    if preset == AdvisoryPollInterval.every_4_hours:
        return (0, {0, 4, 8, 12, 16, 20})
    if preset == AdvisoryPollInterval.daily:
        return (0, 0)
    return None  # pragma: no cover - StrEnum is exhaustive for known values


def advisory_poll_interval_from_env() -> AdvisoryPollInterval:
    """Read ``ADVISORY_POLL_INTERVAL`` for ARQ cron scheduling without instantiating :class:`Settings`.

    Uses only ``os.environ`` so :func:`worker.configure_worker_cron_jobs` can run after ``.env`` is
    loaded. Invalid values map to ``disabled``. Tests and imports that never call that helper see no
    scheduled advisory sync until the worker entrypoint configures cron jobs.
    """
    raw = os.environ.get("ADVISORY_POLL_INTERVAL", AdvisoryPollInterval.disabled.value)
    try:
        return AdvisoryPollInterval(raw)
    except ValueError:
        return AdvisoryPollInterval.disabled


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
    label_name: str = "security"


IssueTrackerEntry = Annotated[
    GitHubIssuesTrackerConfig | JiraTrackerConfig | LinearTrackerConfig,
    Field(discriminator="type"),
]


class GovernanceRule(BaseModel):
    """A single governance rule; all specified criteria must match a finding for it to apply."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: list[Severity] | None = None
    ssvc_action: list[SSVCAction] | None = None
    duplicate: bool | None = None
    patch_available: bool | None = None
    poc_execution: bool | None = None

    @model_validator(mode="after")
    def _at_least_one_criterion(self) -> Self:
        if all(getattr(self, f) is None for f in type(self).model_fields):
            msg = "governance rule must specify at least one criterion"
            raise ValueError(msg)
        return self


class GovernanceApprover(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slack_user: str = Field(pattern=r"^U[A-Z0-9]{6,}$")


class GovernanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    auto_resolve: list[GovernanceRule] = Field(default_factory=list)
    notify: list[GovernanceRule] = Field(default_factory=list)
    approve: list[GovernanceRule] = Field(default_factory=list)


class RepoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    github_org: str
    github_repo: str
    # Git ref for sandbox / patch-oracle when finding evidence has no oracle.vulnerable_ref (e.g. master).
    default_git_ref: str = "main"
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
    # Days a previously-accepted risk remains valid before re-detection re-enters the pipeline.
    # ``0`` disables expiry (acceptances are permanent until manually cleared).
    accepted_risk_ttl_days: int = Field(default=90, ge=0)
    governance: GovernanceConfig | None = None
    # Notified by the interactive Slack approval handler on escalation.
    approvers: list[GovernanceApprover] = Field(default_factory=list)
    # Override with [] to disable REST polling for this repo. Validated: never ``closed``.
    advisory_poll_states: list[str] = Field(default_factory=lambda: ["triage"])
    advisory_poll_max_enqueues_per_tick: int = Field(default=25, ge=0)
    advisory_poll_max_pages: int = Field(default=5, ge=1)
    advisory_poll_per_page: int = Field(default=100, ge=1, le=100)
    advisory_poll_seed_without_enqueue: bool = False

    @field_validator("advisory_poll_states", mode="before")
    @classmethod
    def _normalize_advisory_poll_states(cls, v: object) -> list[str] | object:
        if v is None:
            return []
        if v == "":
            return []
        if not isinstance(v, list):
            return v
        return [str(x).strip().lower() for x in v]

    @model_validator(mode="after")
    def _advisory_poll_states_reject_closed(self) -> Self:
        if "closed" in self.advisory_poll_states:
            msg = "advisory_poll_states must not include 'closed'"
            raise ValueError(msg)
        return self


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


def _env_file_path() -> Path | None:
    p = _REPO_ROOT / ".env"
    return p if p.is_file() else None


def _secrets_dir() -> str | None:
    p = Path("/run/secrets")
    return str(p) if p.is_dir() else None


_DEV_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "dev-local-github-webhook-secret",
        "dev-local-github-pat",
        "xoxb-dev-local-placeholder",
        "dev-local-slack-signing-secret",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore",
        secrets_dir=_secrets_dir(),
    )

    # Dev placeholders so `make run` works without a full `.env`; override for real GitHub/Slack.
    github_webhook_secret: str = Field(default="dev-local-github-webhook-secret")
    github_pat: str = Field(default="dev-local-github-pat")
    slack_bot_token: str = Field(default="xoxb-dev-local-placeholder")
    slack_signing_secret: str = Field(default="dev-local-slack-signing-secret")

    database_url: str = "sqlite+aiosqlite:///./security_scout.db"
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"

    anthropic_api_key: str | None = None

    # Issue tracker credentials (per-tracker; only required for trackers actually configured in repos.yaml).
    # JIRA Cloud Basic auth uses email + token; for self-hosted Server PATs, leave email unset to send Bearer.
    jira_api_email: str | None = None
    jira_api_token: str | None = None
    linear_api_key: str | None = None

    repos_config_path: Path = Field(default=_DEFAULT_REPOS_PATH)

    scm_provider: str = "github"

    mechanical_model: str = "claude-haiku-4-5"
    reasoning_model: str = "claude-sonnet-4-6"
    high_stakes_model: str = "claude-opus-4-6"

    # Operational alert thresholds
    ops_slack_channel: str | None = None
    alert_stuck_workflow_minutes: int = 10
    alert_error_rate_threshold: float = 0.20
    alert_error_rate_window_minutes: int = 60
    alert_latency_p95_seconds: float = 60.0

    # Host header validation
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])

    # MCP read-only server
    mcp_client_allowlist: list[str] = Field(default_factory=list)

    # Container runtime (Docker or Podman API socket for PoC sandbox execution)
    container_socket: str = "unix:///var/run/docker.sock"

    # Advisory list polling: caps and shared sliding-window for ``advisory_poll`` (sync job only).
    advisory_poll_max_enqueues_per_tick_global: int = Field(default=100, ge=0)
    advisory_poll_rate_per_hour: int = Field(default=500, ge=1)
    # Preset cadence for scheduled repository-advisory sync; ``disabled`` = no schedule.
    # When turning polling on, ``hourly`` is a sensible first choice.
    advisory_poll_interval: AdvisoryPollInterval = Field(default=AdvisoryPollInterval.disabled)

    def advisory_poll_interval_seconds_for_dedup(self) -> int | None:
        """Seconds for Redis advisory dedupe TTL; ``None`` when polling is disabled (min TTL applies)."""
        if self.advisory_poll_interval == AdvisoryPollInterval.disabled:
            return None
        return _ADVISORY_POLL_INTERVAL_SECONDS[self.advisory_poll_interval]

    @model_validator(mode="after")
    def _reject_dev_placeholders_in_production(self) -> Self:
        if self.database_url.startswith("sqlite"):
            return self
        offending = [
            name
            for name in ("github_webhook_secret", "github_pat", "slack_bot_token", "slack_signing_secret")
            if getattr(self, name) in _DEV_PLACEHOLDER_SECRETS
        ]
        if offending:
            msg = (
                f"Production database detected ({self.database_url.split('@')[-1] if '@' in self.database_url else '...'}) "
                f"but the following secrets still have dev placeholder values: {', '.join(offending)}. "
                "Set real values in .env or environment variables before deploying."
            )
            raise ValueError(msg)
        return self


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
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        msg = f"{path}: invalid YAML{location}"
        raise ValueError(msg) from exc
    payload = _coerce_manifest_payload(data)
    manifest = ReposManifest.model_validate(payload)
    return manifest, digest


def advisory_polling_schedule_requested(settings: Settings, repos: ReposManifest) -> bool:
    """True when the operator chose a non-disabled interval and at least one repo enables poll states."""
    if settings.advisory_poll_interval == AdvisoryPollInterval.disabled:
        return False
    return any(repo.advisory_poll_states for repo in repos.repos)


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
    # Pair with `db.log_and_persist_config_loaded` when a DB session exists.
    log = structlog.get_logger(__name__)
    log.info(
        "config_loaded",
        metric_name="config_loaded",
        repos_yaml_sha256=app.repos_yaml_sha256,
        repos_config_path=str(app.repos_yaml_path),
        repo_count=len(app.repos.repos),
    )

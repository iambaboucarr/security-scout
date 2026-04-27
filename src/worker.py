# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
import structlog
from arq.connections import RedisSettings
from arq.typing import StartupShutdown, WorkerSettingsBase

from agents.orchestrator import AdvisoryWorkflowParams, ScheduleRetryParams, run_advisory_workflow
from agents.patch_oracle import run_patch_oracle_job
from ai.anthropic_provider import create_provider
from ai.provider import LLMProvider
from config import Settings, configure_logging, load_app_config
from db import create_engine, create_session_factory, session_scope
from exceptions import SecurityScoutError
from tools.advisory_polling import has_active_workflow_run, has_existing_advisory_finding
from tools.docker_sandbox import SandboxBuildError
from tools.issue_tracker import IssueTrackerCredentials
from tools.rate_limiter import SlidingWindowRateLimiter
from tools.scm import normalise_ghsa_id
from tools.scm.github import GitHubSCMProvider
from tools.slack import SlackAPIError, SlackClient

_LOG = structlog.get_logger(__name__)

_MAX_PATCH_ORACLE_FAILURE_DETAIL = 400


def _patch_oracle_failure_reply_text(exc: Exception) -> str:
    name = type(exc).__name__
    detail = str(exc).strip()
    if not detail:
        return f"Patch oracle failed: {name}"
    if len(detail) > _MAX_PATCH_ORACLE_FAILURE_DETAIL:
        detail = detail[: _MAX_PATCH_ORACLE_FAILURE_DETAIL - 1] + "…"
    return f"Patch oracle failed ({name}): {detail}"


async def startup(ctx: dict[Any, Any]) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    app_config = load_app_config(settings)
    ctx["settings"] = settings
    ctx["app_config"] = app_config
    ctx["engine"] = create_engine(settings.database_url)
    ctx["session_factory"] = create_session_factory(ctx["engine"])
    ctx["http_client"] = httpx.AsyncClient(timeout=30.0)
    ctx["llm"] = None
    if settings.anthropic_api_key:
        ctx["llm"] = create_provider(settings.anthropic_api_key)
    redis = ctx.get("redis")
    ctx["rate_limiter"] = SlidingWindowRateLimiter(redis) if redis is not None else None


async def shutdown(ctx: dict[Any, Any]) -> None:
    await ctx["http_client"].aclose()
    llm = ctx.get("llm")
    if llm is not None:
        await llm.close()
    await ctx["engine"].dispose()


async def process_advisory_workflow_job(
    ctx: dict[str, Any],
    *,
    repo_name: str,
    ghsa_id: str,
    advisory_source: str = "repository",
    resume_workflow_run_id: str | None = None,
) -> None:
    settings: Settings = ctx["settings"]
    app_config = ctx["app_config"]
    session_factory = ctx["session_factory"]
    http: httpx.AsyncClient = ctx["http_client"]
    llm: LLMProvider | None = ctx.get("llm")
    rate_limiter: SlidingWindowRateLimiter | None = ctx.get("rate_limiter")

    repo = next((r for r in app_config.repos.repos if r.name == repo_name), None)
    if repo is None:
        _LOG.error("worker_unknown_repo", repo_name=repo_name)
        return

    if advisory_source not in ("repository", "global"):
        _LOG.error("worker_invalid_advisory_source", advisory_source=advisory_source)
        return

    src: Literal["repository", "global"] = "repository" if advisory_source == "repository" else "global"

    resume_uuid: uuid.UUID | None = None
    if resume_workflow_run_id is not None:
        resume_uuid = uuid.UUID(resume_workflow_run_id)

    repo_slug = f"{repo.github_org}/{repo.github_repo}".lower()

    async def schedule_retry(params: ScheduleRetryParams) -> None:
        await ctx["redis"].enqueue_job(
            process_advisory_workflow_job,
            _defer_by=params.delay_seconds,
            repo_name=repo_name,
            ghsa_id=ghsa_id,
            advisory_source=advisory_source,
            resume_workflow_run_id=str(params.workflow_run_id),
        )

    tracker_credentials = IssueTrackerCredentials(
        jira_email=settings.jira_api_email,
        jira_api_token=settings.jira_api_token,
        linear_api_key=settings.linear_api_key,
    )

    async with (
        GitHubSCMProvider(settings.github_pat) as scm,
        SlackClient(settings.slack_bot_token) as slack,
        session_scope(session_factory) as session,
    ):
        if resume_workflow_run_id is None:
            try:
                g_norm = normalise_ghsa_id(ghsa_id)
            except ValueError:
                _LOG.error("worker_invalid_ghsa", ghsa_id=ghsa_id, repo_name=repo_name)
                return
            if await has_existing_advisory_finding(
                session,
                repo_slug=repo_slug,
                ghsa_id=g_norm,
            ):
                _LOG.info(
                    "advisory_workflow_dedupe_skip",
                    metric_name="advisory_poll_skipped_dedupe_total",
                    reason="finding_exists",
                    repo=repo_name,
                    repo_slug=repo_slug,
                    ghsa_id=g_norm,
                )
                return
            if await has_active_workflow_run(
                session,
                repo_slug=repo_slug,
                ghsa_id=g_norm,
                now=datetime.now(UTC),
            ):
                _LOG.info(
                    "advisory_workflow_dedupe_skip",
                    metric_name="advisory_poll_skipped_dedupe_total",
                    reason="run_exists",
                    repo=repo_name,
                    repo_slug=repo_slug,
                    ghsa_id=g_norm,
                )
                return

        await run_advisory_workflow(
            session,
            repo,
            scm,
            http,
            slack,
            AdvisoryWorkflowParams(
                ghsa_id=ghsa_id,
                advisory_source=src,
                run_id=uuid.uuid4(),
                llm=llm,
                reasoning_model=settings.reasoning_model,
                schedule_retry=schedule_retry,
                resume_workflow_run_id=resume_uuid,
                rate_limiter=rate_limiter,
                tracker_credentials=tracker_credentials,
                container_socket=settings.container_socket,
            ),
        )


async def process_patch_oracle_job(
    ctx: dict[str, Any],
    *,
    repo_name: str,
    finding_id: str,
    workflow_run_id: str,
    slack_channel_id: str,
    slack_message_ts: str,
) -> None:
    settings: Settings = ctx["settings"]
    app_config = ctx["app_config"]
    session_factory = ctx["session_factory"]

    repo = next((r for r in app_config.repos.repos if r.name == repo_name), None)
    if repo is None:
        _LOG.error("patch_oracle_unknown_repo", repo_name=repo_name)
        return

    fid = uuid.UUID(finding_id)
    wid = uuid.UUID(workflow_run_id)

    async with (
        GitHubSCMProvider(settings.github_pat) as scm,
        SlackClient(settings.slack_bot_token) as slack,
        session_scope(session_factory) as session,
    ):
        try:
            _tier, summary = await run_patch_oracle_job(
                session,
                scm,
                repo_slug=f"{repo.github_org}/{repo.github_repo}",
                finding_id=fid,
                workflow_run_id=wid,
                container_socket=settings.container_socket,
                default_git_ref=repo.default_git_ref,
            )
        except (RuntimeError, SecurityScoutError, SandboxBuildError) as exc:
            _LOG.exception(
                "patch_oracle_job_failed",
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
                err=str(exc),
            )
            try:
                await slack.post_thread_reply(
                    slack_channel_id,
                    thread_ts=slack_message_ts,
                    text=_patch_oracle_failure_reply_text(exc),
                    finding_id=finding_id,
                    workflow_run_id=wid,
                )
            except SlackAPIError as reply_exc:
                _LOG.warning(
                    "patch_oracle_failure_reply_failed",
                    err=str(reply_exc),
                    finding_id=finding_id,
                )
            return

        try:
            await slack.post_thread_reply(
                slack_channel_id,
                thread_ts=slack_message_ts,
                text=summary,
                finding_id=finding_id,
                workflow_run_id=wid,
            )
        except SlackAPIError as reply_exc:
            _LOG.warning(
                "patch_oracle_success_reply_failed",
                err=str(reply_exc),
                finding_id=finding_id,
            )


class WorkerSettings(WorkerSettingsBase):
    # Use REDIS_URL env (same as Settings.redis_url) so importing this module does not require all app secrets.
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    on_startup: StartupShutdown | None = startup
    on_shutdown: StartupShutdown | None = shutdown
    functions: Sequence[Any] = [process_advisory_workflow_job, process_patch_oracle_job]

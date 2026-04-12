from __future__ import annotations

import os
import uuid
from typing import Any, ClassVar, Literal

import httpx
import structlog
from arq.connections import RedisSettings

from agents.orchestrator import ScheduleRetryParams, run_advisory_workflow
from ai.anthropic_provider import create_provider
from ai.provider import LLMProvider
from config import Settings, configure_logging, load_app_config
from db import create_engine, create_session_factory, session_scope
from tools.github import GitHubClient
from tools.slack import SlackClient

_LOG = structlog.get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
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


async def shutdown(ctx: dict[str, Any]) -> None:
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

    async def schedule_retry(params: ScheduleRetryParams) -> None:
        await ctx["redis"].enqueue_job(
            process_advisory_workflow_job,
            _defer_by=params.delay_seconds,
            repo_name=repo_name,
            ghsa_id=ghsa_id,
            advisory_source=advisory_source,
            resume_workflow_run_id=str(params.workflow_run_id),
        )

    async with (
        GitHubClient(settings.github_pat) as gh,
        SlackClient(settings.slack_bot_token) as slack,
        session_scope(session_factory) as session,
    ):
        await run_advisory_workflow(
            session,
            repo,
            gh,
            http,
            slack,
            ghsa_id=ghsa_id,
            advisory_source=src,
            run_id=uuid.uuid4(),
            llm=llm,
            reasoning_model=settings.reasoning_model,
            schedule_retry=schedule_retry,
            resume_workflow_run_id=resume_uuid,
        )


class WorkerSettings:
    # Use REDIS_URL env (same as Settings.redis_url) so importing this module does not require all app secrets.
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    on_startup = startup
    on_shutdown = shutdown
    functions: ClassVar[list[Any]] = [process_advisory_workflow_job]

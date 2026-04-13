from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from config import Settings, configure_logging, load_app_config
from db import create_engine, create_session_factory, log_and_persist_config_loaded, session_scope
from models import Base
from webhooks import create_github_webhook_router
from webhooks.slack import create_slack_webhook_router

_LOG = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    settings = Settings()
    app_config = load_app_config(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        engine = create_engine(settings.database_url)
        if settings.database_url.startswith("sqlite"):
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        session_factory = create_session_factory(engine)
        redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))

        app.state.settings = settings
        app.state.app_config = app_config
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.redis_pool = redis_pool

        async def enqueue_advisory(
            *,
            repo_name: str,
            ghsa_id: str,
            advisory_source: str = "repository",
            resume_workflow_run_id: str | None = None,
        ) -> str | None:
            job = await redis_pool.enqueue_job(
                "worker.process_advisory_workflow_job",
                repo_name=repo_name,
                ghsa_id=ghsa_id,
                advisory_source=advisory_source,
                resume_workflow_run_id=resume_workflow_run_id,
            )
            if job is None:
                return None
            if isinstance(job, str):
                return job
            jid = getattr(job, "job_id", None)
            return str(jid) if jid is not None else None

        app.state.enqueue_advisory = enqueue_advisory

        async with session_scope(session_factory) as session:
            await log_and_persist_config_loaded(session, app_config)

        _LOG.info("app_startup_complete", metric_name="app_startup_complete")
        yield
        await redis_pool.close()
        await engine.dispose()
        _LOG.info("app_shutdown_complete", metric_name="app_shutdown_complete")

    app = FastAPI(title="Security Scout", lifespan=lifespan)
    app.include_router(create_github_webhook_router())
    app.include_router(create_slack_webhook_router())
    return app


app = create_app()

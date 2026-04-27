# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from config import Settings, configure_logging, load_app_config
from db import create_engine, create_session_factory, log_and_persist_config_loaded, session_scope
from models import Base
from tools.advisory_polling import try_enqueue_advisory
from webhooks import create_github_webhook_router
from webhooks.slack import create_slack_webhook_router

_LOG = structlog.get_logger(__name__)

_MAX_BODY_BYTES = 2_097_152  # 2 MB


class ContentSizeLimitMiddleware:
    """Reject requests whose body exceeds *max_bytes* with 413."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = _MAX_BODY_BYTES) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        content_length = self._header_value(scope, b"content-length")
        if content_length is not None and int(content_length) > self._max_bytes:
            await self._send_413(send)
            return

        received = 0

        async def _limited_receive() -> dict[str, object]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received += len(body) if isinstance(body, bytes) else 0
                if received > self._max_bytes:
                    raise _BodyTooLarge
            return dict(message)

        try:
            await self._app(scope, _limited_receive, send)
        except _BodyTooLarge:
            await self._send_413(send)

    @staticmethod
    def _header_value(scope: Scope, name: bytes) -> bytes | None:
        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        for key, value in headers:
            if key.lower() == name:
                return value
        return None

    @staticmethod
    async def _send_413(send: Send) -> None:
        body = b'{"detail":"Request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    pass


async def _run_readiness_checks(engine: Any, redis_pool: Any) -> tuple[dict[str, Any], int]:
    checks: dict[str, str] = {}
    ok = True

    if engine is None:
        checks["db"] = "uninitialised"
        ok = False
    else:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["db"] = "ok"
        except Exception:
            _LOG.warning("readyz_db_check_failed", exc_info=True)
            checks["db"] = "error"
            ok = False

    if redis_pool is None:
        checks["redis"] = "uninitialised"
        ok = False
    else:
        try:
            await redis_pool.ping()
            checks["redis"] = "ok"
        except Exception:
            _LOG.warning("readyz_redis_check_failed", exc_info=True)
            checks["redis"] = "error"
            ok = False

    return {"status": "ok" if ok else "degraded", "checks": checks}, 200 if ok else 503


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
            cfg = next((r for r in app_config.repos.repos if r.name == repo_name), None)
            if cfg is None:
                return None
            repo_slug = f"{cfg.github_org}/{cfg.github_repo}".lower()
            return await try_enqueue_advisory(
                redis_pool,
                repo_config_name=repo_name,
                repo_slug=repo_slug,
                ghsa_id=ghsa_id,
                advisory_source=advisory_source,
                resume_workflow_run_id=resume_workflow_run_id,
            )

        app.state.enqueue_advisory = enqueue_advisory

        async def enqueue_patch_oracle(
            *,
            repo_name: str,
            finding_id: str,
            workflow_run_id: str,
            slack_channel_id: str,
            slack_message_ts: str,
        ) -> str | None:
            job = await redis_pool.enqueue_job(
                "process_patch_oracle_job",
                repo_name=repo_name,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
                slack_channel_id=slack_channel_id,
                slack_message_ts=slack_message_ts,
            )
            if job is None:
                return None
            if isinstance(job, str):
                return job
            jid = getattr(job, "job_id", None)
            return str(jid) if jid is not None else None

        app.state.enqueue_patch_oracle = enqueue_patch_oracle

        async with session_scope(session_factory) as session:
            await log_and_persist_config_loaded(session, app_config)

        _LOG.info("app_startup_complete", metric_name="app_startup_complete")
        yield
        await redis_pool.close()
        await engine.dispose()
        _LOG.info("app_shutdown_complete", metric_name="app_shutdown_complete")

    is_dev = settings.database_url.startswith("sqlite")
    app = FastAPI(
        title="Security Scout",
        lifespan=lifespan,
        docs_url="/docs" if is_dev else None,
        redoc_url="/redoc" if is_dev else None,
        openapi_url="/openapi.json" if is_dev else None,
    )

    app.add_middleware(ContentSizeLimitMiddleware)

    if settings.trusted_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        engine = getattr(app.state, "engine", None)
        redis_pool = getattr(app.state, "redis_pool", None)
        body, status = await _run_readiness_checks(engine, redis_pool)
        return JSONResponse(body, status_code=status)

    app.include_router(create_github_webhook_router())
    app.include_router(create_slack_webhook_router())
    return app


app = create_app()

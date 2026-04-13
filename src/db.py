from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config import AppConfig, log_config_loaded
from models import AgentActionLog


def create_engine(database_url: str) -> AsyncEngine:
    engine = create_async_engine(
        database_url,
        echo=False,
        future=True,
    )

    if database_url.startswith("sqlite"):
        sync_engine = engine.sync_engine

        @event.listens_for(sync_engine, "connect")
        def _sqlite_fk_pragma(dbapi_connection: Any, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def session_scope(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        # Do not catch BaseException: KeyboardInterrupt/SystemExit should not run rollback+re-raise here.
        await session.rollback()
        raise
    finally:
        await session.close()


async def persist_config_loaded_audit(session: AsyncSession, app: AppConfig) -> None:
    entry = AgentActionLog(
        agent="system",
        tool_name="config_loaded",
        tool_inputs={
            "repos_yaml_sha256": app.repos_yaml_sha256,
            "repos_config_path": str(app.repos_yaml_path),
            "repo_count": len(app.repos.repos),
        },
        tool_output=None,
        workflow_run_id=None,
    )
    session.add(entry)
    await session.flush()


async def log_and_persist_config_loaded(session: AsyncSession, app: AppConfig) -> None:
    # Structured log + queryable `AgentActionLog` row for config-load audit trail.
    log_config_loaded(app)
    await persist_config_loaded_audit(session, app)


__all__ = [
    "create_engine",
    "create_session_factory",
    "log_and_persist_config_loaded",
    "persist_config_loaded_audit",
    "session_scope",
]

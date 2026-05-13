# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.auth import require_scope
from db import create_engine, create_session_factory
from models import ApiToken, ApiTokenScope, Base


@pytest.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'auth_test.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()


def _build_app(
    factory: async_sessionmaker[AsyncSession],
    required_scope: ApiTokenScope,
) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    dep = Depends(require_scope(required_scope))

    @app.get("/protected")
    async def protected(token: ApiToken = dep) -> JSONResponse:
        return JSONResponse({"token_id": str(token.id), "name": token.name})

    return app


@pytest.fixture
async def make_client(session_factory):
    async def _factory(scope: ApiTokenScope) -> AsyncClient:
        app = _build_app(session_factory, scope)
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    return _factory

# SPDX-License-Identifier: Apache-2.0
"""HTTP rate-limit dependency: shared sliding-window enforced per token."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api.rate_limit import (
    API_RATE_LIMIT_CIRCUIT_UNAVAILABLE_DETAIL,
    API_TOKEN_RATE_LIMIT,
    API_TOKEN_RATE_WINDOW_SECONDS,
)
from api.tokens import create_token
from api.v1 import create_v1_router
from db import create_engine, create_session_factory, session_scope
from models import ApiTokenScope, Base
from tests.tools.test_rate_limiter import _Clock, _FakeRedis


@pytest.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'rl_api.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
def fake_redis() -> _FakeRedis:
    return _FakeRedis(_Clock())


def _build_app(
    factory: async_sessionmaker[AsyncSession],
    *,
    redis: _FakeRedis | None,
) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    if redis is not None:
        app.state.redis_pool = redis
    app.include_router(create_v1_router())
    return app


async def _bearer_for(
    factory: async_sessionmaker[AsyncSession],
    *,
    scopes: list[ApiTokenScope],
) -> str:
    async with session_scope(factory) as session:
        issued = await create_token(session, name="rl-test", scopes=scopes, owner_slack_id="U_RL")
    return issued.raw_token


async def _make_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_under_the_limit_requests_succeed(
    session_factory: async_sessionmaker[AsyncSession],
    fake_redis: _FakeRedis,
) -> None:
    token = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])
    app = _build_app(session_factory, redis=fake_redis)
    headers = {"Authorization": f"Bearer {token}"}

    async with await _make_client(app) as client:
        for _ in range(API_TOKEN_RATE_LIMIT):
            resp = await client.get(
                "/api/v1/findings",
                params={"repo": "acme/app"},
                headers=headers,
            )
            assert resp.status_code == 200


async def test_request_above_limit_returns_429(
    session_factory: async_sessionmaker[AsyncSession],
    fake_redis: _FakeRedis,
) -> None:
    token = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])
    app = _build_app(session_factory, redis=fake_redis)
    headers = {"Authorization": f"Bearer {token}"}

    async with await _make_client(app) as client:
        for _ in range(API_TOKEN_RATE_LIMIT):
            resp = await client.get(
                "/api/v1/findings",
                params={"repo": "acme/app"},
                headers=headers,
            )
            assert resp.status_code == 200

        resp = await client.get(
            "/api/v1/findings",
            params={"repo": "acme/app"},
            headers=headers,
        )
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == str(API_TOKEN_RATE_WINDOW_SECONDS)


async def test_rate_limit_is_per_token_not_global(
    session_factory: async_sessionmaker[AsyncSession],
    fake_redis: _FakeRedis,
) -> None:
    token_a = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])
    token_b = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])
    app = _build_app(session_factory, redis=fake_redis)

    async with await _make_client(app) as client:
        for _ in range(API_TOKEN_RATE_LIMIT):
            resp = await client.get(
                "/api/v1/findings",
                params={"repo": "acme/app"},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert resp.status_code == 200

        # Second token should still be under its own budget.
        resp = await client.get(
            "/api/v1/findings",
            params={"repo": "acme/app"},
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp.status_code == 200


async def test_missing_redis_pool_fails_open(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    token = await _bearer_for(session_factory, scopes=[ApiTokenScope.findings_read])
    app = _build_app(session_factory, redis=None)

    async with await _make_client(app) as client:
        for _ in range(API_TOKEN_RATE_LIMIT + 5):
            resp = await client.get(
                "/api/v1/findings",
                params={"repo": "acme/app"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200


async def test_unauthenticated_request_does_not_consume_budget(
    session_factory: async_sessionmaker[AsyncSession],
    fake_redis: _FakeRedis,
) -> None:
    app = _build_app(session_factory, redis=fake_redis)
    async with await _make_client(app) as client:
        for _ in range(API_TOKEN_RATE_LIMIT + 10):
            resp = await client.get(
                "/api/v1/findings",
                params={"repo": "acme/app"},
                headers={"Authorization": "Bearer never-issued"},
            )
            assert resp.status_code == 401

    # No keys should have been written to the rate limiter store.
    assert all(not k.startswith("rl:api_token:") for k in fake_redis.sorted_sets)


async def test_circuit_open_returns_503(
    session_factory: async_sessionmaker[AsyncSession],
    fake_redis: _FakeRedis,
) -> None:
    async with session_scope(session_factory) as session:
        issued = await create_token(
            session,
            name="rl-circuit",
            scopes=[ApiTokenScope.findings_read],
            owner_slack_id="U_RL",
        )
    token_id = issued.record.id
    await fake_redis.setex(f"rl:circuit:{token_id}", 120, "1")

    app = _build_app(session_factory, redis=fake_redis)
    headers = {"Authorization": f"Bearer {issued.raw_token}"}

    async with await _make_client(app) as client:
        resp = await client.get(
            "/api/v1/findings",
            params={"repo": "acme/app"},
            headers=headers,
        )
    assert resp.status_code == 503
    assert resp.json()["detail"] == API_RATE_LIMIT_CIRCUIT_UNAVAILABLE_DETAIL
    assert resp.headers.get("retry-after") == "120"

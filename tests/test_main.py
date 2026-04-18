# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from main import _MAX_BODY_BYTES, ContentSizeLimitMiddleware, _run_readiness_checks


def _tiny_app() -> FastAPI:
    """Minimal app with the size-limit middleware for isolated testing."""
    app = FastAPI()
    app.add_middleware(ContentSizeLimitMiddleware)

    @app.post("/echo")
    async def echo(request: Request) -> JSONResponse:
        body = await request.body()
        return JSONResponse({"size": len(body)})

    return app


@pytest.fixture
def client() -> AsyncClient:
    transport = ASGITransport(app=_tiny_app())
    return AsyncClient(transport=transport, base_url="http://test")


async def test_content_size_limit_allows_normal_body(client: AsyncClient) -> None:
    payload = b"x" * 1024
    resp = await client.post("/echo", content=payload)
    assert resp.status_code == 200
    assert resp.json()["size"] == 1024


async def test_content_size_limit_rejects_oversized_body(client: AsyncClient) -> None:
    payload = b"x" * (_MAX_BODY_BYTES + 1)
    resp = await client.post("/echo", content=payload)
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


async def test_content_size_limit_rejects_via_content_length_header(client: AsyncClient) -> None:
    resp = await client.post(
        "/echo",
        content=b"small",
        headers={"content-length": str(_MAX_BODY_BYTES + 1)},
    )
    assert resp.status_code == 413


async def test_content_size_limit_allows_exactly_max(client: AsyncClient) -> None:
    payload = b"x" * _MAX_BODY_BYTES
    resp = await client.post("/echo", content=payload)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------


class _FakeConn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def execute(self, _stmt):
        return None


class _FakeEngine:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises

    def connect(self):
        if self._raises is not None:
            raise self._raises
        return _FakeConn()


async def test_readyz_reports_ok_when_deps_healthy() -> None:
    redis = MagicMock()
    redis.ping = AsyncMock(return_value=True)
    body, status = await _run_readiness_checks(_FakeEngine(), redis)
    assert status == 200
    assert body == {"status": "ok", "checks": {"db": "ok", "redis": "ok"}}


async def test_readyz_reports_degraded_when_deps_missing() -> None:
    body, status = await _run_readiness_checks(None, None)
    assert status == 503
    assert body["status"] == "degraded"
    assert body["checks"] == {"db": "uninitialised", "redis": "uninitialised"}


async def test_readyz_reports_degraded_when_db_fails() -> None:
    redis = MagicMock()
    redis.ping = AsyncMock(return_value=True)
    body, status = await _run_readiness_checks(_FakeEngine(raises=RuntimeError("db down")), redis)
    assert status == 503
    assert body["checks"]["db"] == "error"
    assert body["checks"]["redis"] == "ok"


async def test_readyz_reports_degraded_when_redis_fails() -> None:
    redis = MagicMock()
    redis.ping = AsyncMock(side_effect=RuntimeError("redis down"))
    body, status = await _run_readiness_checks(_FakeEngine(), redis)
    assert status == 503
    assert body["checks"]["db"] == "ok"
    assert body["checks"]["redis"] == "error"

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from db import create_engine

pytestmark = pytest.mark.postgres


def _postgres_test_url() -> str:
    url = os.environ.get("POSTGRES_TEST_URL", "").strip()
    if not url:
        pytest.fail(
            "POSTGRES_TEST_URL is not set. For local runs: docker compose up -d postgres "
            "and export POSTGRES_TEST_URL (see Makefile POSTGRES_TEST_URL default).",
        )
    return url


@pytest.mark.asyncio
async def test_postgres_async_engine_select_one() -> None:
    url = _postgres_test_url()
    engine = create_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
    finally:
        await engine.dispose()

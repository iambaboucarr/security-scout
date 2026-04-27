# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from db import create_engine, create_session_factory
from models import Base

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Run postgres-marked tests on one xdist worker to avoid DDL races on a shared DB."""
    _ = config
    for item in items:
        if item.get_closest_marker("postgres") is None:
            continue
        item.add_marker(pytest.mark.xdist_group("postgres_shared_schema"))


@pytest.fixture
async def db_session(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    async with factory() as session:
        yield session
    await engine.dispose()

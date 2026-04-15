# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from db import create_engine, create_session_factory
from models import Base


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

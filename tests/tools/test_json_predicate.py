# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import JSON, String, column, delete, func, or_, select, table
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import CompileError

from db import create_engine, create_session_factory
from models import Base, Finding, FindingStatus, Severity, WorkflowKind
from tools.json_predicate import json_text_at, json_text_at_upper_trimmed


def test_json_text_at_compile_sqlite_uses_json_extract() -> None:
    m = table("f", column("j", type_=JSON), column("i", type_=String))
    stmt = select(json_text_at(m.c.j, "ghsa_id"))
    s = str(stmt.compile(dialect=sqlite.dialect()))
    assert "json_extract" in s

    m2 = table("f2", column("j2", type_=JSON), column("i2", type_=String))
    stmt2 = select(json_text_at(m2.c.j2, "oracle", "vuln"))
    s2 = str(stmt2.compile(dialect=sqlite.dialect()))
    assert "json_extract" in s2
    assert "$.oracle.vuln" in set(stmt2.compile(dialect=sqlite.dialect()).params.values())


def test_json_text_at_compile_postgresql_uses_text_form() -> None:
    m = table("f", column("j", type_=JSON), column("i", type_=String))
    stmt = select(json_text_at(m.c.j, "k"))
    s = str(stmt.compile(dialect=postgresql.dialect()))
    assert "->>" in s


def test_json_text_at_invalid_path_rejected() -> None:
    c = column("j", type_=JSON)
    with pytest.raises(ValueError, match="invalid json path key"):
        json_text_at(c, "a.b")
    with pytest.raises(ValueError, match="requires at least one path key"):
        json_text_at(c)


def test_json_text_at_unsupported_dialect() -> None:
    from sqlalchemy.dialects import mysql

    m = table("f", column("j", type_=JSON), column("i", type_=String))
    stmt = select(json_text_at(m.c.j, "k"))
    with pytest.raises(CompileError, match="only supported for sqlite and postgresql"):
        stmt.compile(dialect=mysql.dialect())


@pytest.mark.asyncio
async def test_json_text_at_sqlite_finding_evidence_ghsa(db_session) -> None:
    ghsa = "GHSA-ABCD-ABCD-ABCD"
    fid = uuid.uuid4()
    db_session.add(
        Finding(
            id=fid,
            workflow=WorkflowKind.advisory,
            repo_name="o/r",
            source_ref="https://example.com",
            severity=Severity.high,
            title="t",
            status=FindingStatus.unconfirmed,
            evidence={"ghsa_id": ghsa, "n": 1},
        ),
    )
    await db_session.commit()
    ghsa_in_evidence = json_text_at_upper_trimmed(Finding.evidence, "ghsa_id") == ghsa
    ghsa_in_source = func.upper(Finding.source_ref).contains(ghsa)
    r = await db_session.execute(
        select(Finding.id).where(
            or_(ghsa_in_evidence, ghsa_in_source),
        ),
    )
    assert r.scalar_one() == fid


@pytest.mark.asyncio
async def test_json_text_at_sqlite_evidence_ghsa_padded_still_matches(db_session) -> None:
    ghsa = "GHSA-ABCD-ABCD-ABCD"
    fid = uuid.uuid4()
    db_session.add(
        Finding(
            id=fid,
            workflow=WorkflowKind.advisory,
            repo_name="o/r",
            source_ref="https://example.com/x",
            severity=Severity.high,
            title="t",
            status=FindingStatus.unconfirmed,
            evidence={"ghsa_id": f"  {ghsa}  ", "n": 1},
        ),
    )
    await db_session.commit()
    ghsa_in_evidence = json_text_at_upper_trimmed(Finding.evidence, "ghsa_id") == ghsa
    ghsa_in_source = func.upper(Finding.source_ref).contains(ghsa)
    r = await db_session.execute(
        select(Finding.id).where(or_(ghsa_in_evidence, ghsa_in_source)),
    )
    assert r.scalar_one() == fid


def _require_postgres_url() -> str:
    url = os.environ.get("POSTGRES_TEST_URL", "").strip()
    if not url:
        pytest.fail(
            "POSTGRES_TEST_URL is not set. For local runs: docker compose up -d postgres "
            "and export POSTGRES_TEST_URL (see Makefile POSTGRES_TEST_URL default).",
        )
    return url


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_json_text_at_postgres_finding_table() -> None:
    url = _require_postgres_url()
    engine = create_engine(url)
    ghsa = "GHSA-WWWW-XXXX-YYYY"
    fid = uuid.uuid4()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = create_session_factory(engine)
        ghsa_in_evidence = json_text_at_upper_trimmed(Finding.evidence, "ghsa_id") == ghsa
        ghsa_in_source = func.upper(Finding.source_ref).contains(ghsa)
        async with factory() as session:
            session.add(
                Finding(
                    id=fid,
                    workflow=WorkflowKind.advisory,
                    repo_name="o/r",
                    source_ref="https://example.com",
                    severity=Severity.high,
                    title="t",
                    status=FindingStatus.unconfirmed,
                    evidence={"ghsa_id": ghsa},
                ),
            )
            await session.commit()
        async with factory() as session:
            r = await session.execute(
                select(Finding.id).where(or_(ghsa_in_evidence, ghsa_in_source)),
            )
            assert r.scalar_one() == fid
            await session.execute(delete(Finding).where(Finding.id == fid))
            await session.commit()

        fid2 = uuid.uuid4()
        ghsa2 = "GHSA-PPAD-PPAD-PPAD"
        async with factory() as session:
            session.add(
                Finding(
                    id=fid2,
                    workflow=WorkflowKind.advisory,
                    repo_name="o/r",
                    source_ref="https://example.com/y",
                    severity=Severity.high,
                    title="t2",
                    status=FindingStatus.unconfirmed,
                    evidence={"ghsa_id": f"  {ghsa2}  "},
                ),
            )
            await session.commit()
        async with factory() as session:
            ge = json_text_at_upper_trimmed(Finding.evidence, "ghsa_id") == ghsa2
            gs = func.upper(Finding.source_ref).contains(ghsa2)
            r2 = await session.execute(select(Finding.id).where(or_(ge, gs)))
            assert r2.scalar_one() == fid2
            await session.execute(delete(Finding).where(Finding.id == fid2))
            await session.commit()
    finally:
        await engine.dispose()

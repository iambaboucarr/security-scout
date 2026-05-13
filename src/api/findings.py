# SPDX-License-Identifier: Apache-2.0
"""Authenticated HTTP routes for read-only finding and triage queries."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import status as http_status
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import require_scope
from models import ApiToken, ApiTokenScope
from tools import queries


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession]:
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session factory not initialised",
        )
    async with factory() as session:
        yield session


def _reraise_query_value_error(exc: ValueError) -> NoReturn:
    msg = str(exc)
    if msg.startswith("finding not found"):
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=msg) from exc
    raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=msg) from exc


def create_findings_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["findings"])

    @router.get("/findings", response_model=list[queries.FindingSummary])
    async def list_findings(
        _token: Annotated[ApiToken, Depends(require_scope(ApiTokenScope.findings_read))],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        repo: str,
        severity: str | None = None,
        finding_status: str | None = None,
        limit: int = 50,
    ) -> list[queries.FindingSummary]:
        try:
            return await queries.query_findings(
                session,
                repo=repo,
                severity=severity,
                status=finding_status,
                limit=limit,
            )
        except ValueError as e:
            _reraise_query_value_error(e)

    @router.get("/findings/{finding_id}", response_model=queries.FindingDetail)
    async def read_finding(
        _token: Annotated[ApiToken, Depends(require_scope(ApiTokenScope.findings_read))],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        finding_id: UUID,
    ) -> queries.FindingDetail:
        try:
            return await queries.get_finding_detail(session, finding_id=str(finding_id))
        except ValueError as e:
            _reraise_query_value_error(e)

    @router.get("/dependencies/check", response_model=queries.DependencyRisk)
    async def check_dependency_endpoint(
        _token: Annotated[ApiToken, Depends(require_scope(ApiTokenScope.findings_read))],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        package: str,
        version: str = "",
        ecosystem: str = "",
    ) -> queries.DependencyRisk:
        try:
            return await queries.check_dependency(
                session,
                package=package,
                version=version,
                ecosystem=ecosystem,
            )
        except ValueError as e:
            _reraise_query_value_error(e)

    @router.get("/triage/{advisory_id}", response_model=queries.TriageStatus)
    async def triage_status(
        _token: Annotated[ApiToken, Depends(require_scope(ApiTokenScope.findings_read))],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        advisory_id: str,
    ) -> queries.TriageStatus:
        try:
            return await queries.get_triage_status(session, advisory_id=advisory_id)
        except ValueError as e:
            _reraise_query_value_error(e)

    return router


__all__ = ["create_findings_router"]

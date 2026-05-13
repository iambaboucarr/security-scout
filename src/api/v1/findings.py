# SPDX-License-Identifier: Apache-2.0
"""Authenticated HTTP routes for finding queries and review decisions.

Read endpoints (``GET /findings``, ``GET /findings/{id}``, ``GET /check``,
``GET /triage/{advisory_id}``) require ``findings:read``. The review
endpoint (``POST /findings/{id}/review``) requires ``findings:write`` and
delegates to :func:`agents.api_review.apply_api_review_decision`, which
consults the orchestrator's governance gate before applying any state
change. Every authenticated request is counted against a per-token
sliding window (60 requests / 60 seconds) and rejected with ``429`` when
the limit is hit.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agents.api_review import (
    ReviewDecision,
    ReviewError,
    ReviewRejectionReason,
    apply_api_review_decision,
)
from api.auth import require_scope
from api.rate_limit import require_token_rate_limit
from config import AppConfig
from models import ApiToken, ApiTokenScope
from tools import queries
from tools.queries import FINDINGS_LIST_DEFAULT_LIMIT, FINDINGS_LIST_MAX_LIMIT

_REVIEW_REJECTION_STATUS: dict[ReviewRejectionReason, int] = {
    ReviewRejectionReason.finding_not_found: http_status.HTTP_404_NOT_FOUND,
    ReviewRejectionReason.no_pending_run: http_status.HTTP_409_CONFLICT,
    ReviewRejectionReason.governance_does_not_require_approval: http_status.HTTP_409_CONFLICT,
}


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ReviewDecision
    context: str | None = None


class ReviewResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: ReviewDecision
    finding_id: str
    workflow_run_id: str
    reviewer_id: str
    finding_status: str
    workflow_state: str


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] | None = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session factory not initialised",
        )
    async with factory() as session:
        yield session


def _require_app_config(request: Request) -> AppConfig:
    cfg: AppConfig | None = getattr(request.app.state, "app_config", None)
    if cfg is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application config not initialised",
        )
    return cfg


def _reraise_query_value_error(exc: ValueError) -> NoReturn:
    msg = str(exc)
    if msg.startswith("finding not found"):
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=msg) from exc
    raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=msg) from exc


_require_read = require_token_rate_limit(require_scope(ApiTokenScope.findings_read))
_require_write = require_token_rate_limit(require_scope(ApiTokenScope.findings_write))


def create_findings_router() -> APIRouter:
    router = APIRouter(tags=["findings"])

    @router.get("/findings", response_model=list[queries.FindingSummary])
    async def list_findings(
        _token: Annotated[ApiToken, Depends(_require_read)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        repo: str,
        severity: str | None = None,
        finding_status: str | None = None,
        limit: Annotated[
            int,
            Query(
                ge=1,
                le=FINDINGS_LIST_MAX_LIMIT,
                description=f"Max rows to return (1-{FINDINGS_LIST_MAX_LIMIT}).",
            ),
        ] = FINDINGS_LIST_DEFAULT_LIMIT,
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
        _token: Annotated[ApiToken, Depends(_require_read)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        finding_id: UUID,
    ) -> queries.FindingDetail:
        try:
            return await queries.get_finding_detail(session, finding_id=str(finding_id))
        except ValueError as e:
            _reraise_query_value_error(e)

    @router.get("/check", response_model=queries.DependencyRisk)
    async def check_dependency_endpoint(
        _token: Annotated[ApiToken, Depends(_require_read)],
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
        _token: Annotated[ApiToken, Depends(_require_read)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        advisory_id: str,
    ) -> queries.TriageStatus:
        try:
            return await queries.get_triage_status(session, advisory_id=advisory_id)
        except ValueError as e:
            _reraise_query_value_error(e)

    @router.post(
        "/findings/{finding_id}/review",
        response_model=ReviewResponse,
        status_code=http_status.HTTP_200_OK,
    )
    async def review_finding(
        request: Request,
        token: Annotated[ApiToken, Depends(_require_write)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        finding_id: UUID,
        body: ReviewRequest,
    ) -> ReviewResponse:
        app_config = _require_app_config(request)
        try:
            result = await apply_api_review_decision(
                session,
                app_config,
                finding_id=finding_id,
                decision=body.decision,
                token_id=token.id,
                context=body.context,
            )
        except ReviewError as exc:
            raise HTTPException(
                status_code=_REVIEW_REJECTION_STATUS[exc.reason],
                detail=str(exc),
            ) from exc
        return ReviewResponse(
            decision=result.decision,
            finding_id=str(result.finding_id),
            workflow_run_id=str(result.workflow_run_id),
            reviewer_id=result.reviewer_id,
            finding_status=result.finding_status.value,
            workflow_state=result.workflow_state.value,
        )

    return router


__all__ = ["ReviewRequest", "ReviewResponse", "create_findings_router"]

"""Read-only MCP server exposing finding and dependency queries.

Exposes four tools to MCP clients (Claude Code, Cursor, etc.):
  - query_findings       — list findings for a repo, optionally filtered
  - get_finding_detail   — full detail for a single finding
  - check_dependency     — check known advisories for a package version
  - get_triage_status    — triage outcome for a specific advisory

All responses are validated via Pydantic and sanitised through
``tools.input_sanitiser`` before reaching the model. An optional client
allowlist restricts which MCP clients may connect.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import mcp.types as mcp_types
import structlog
from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models import Finding, FindingStatus, Severity
from tools.input_sanitiser import sanitize_text

_LOG = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class FindingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    severity: str
    ssvc_action: str | None
    status: str
    triage_confidence: float | None
    source_ref: str
    created_at: datetime


class FindingDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    description: str | None
    severity: str
    ssvc_action: str | None
    status: str
    triage_confidence: float | None
    source_ref: str
    cve_id: str | None
    cwe_ids: list[str] | None
    cvss_score: float | None
    cvss_vector: str | None
    known_status: str | None
    duplicate_of: str | None
    duplicate_url: str | None
    reproduction: str | None
    evidence: dict[str, Any] | None
    approved_by: str | None
    approved_at: datetime | None
    created_at: datetime


class DependencyAdvisory(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    severity: str
    ssvc_action: str | None
    source_ref: str


class DependencyRisk(BaseModel):
    model_config = ConfigDict(frozen=True)

    package: str
    version: str
    ecosystem: str
    advisory_count: int
    advisories: list[DependencyAdvisory]


class TriageStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    advisory_id: str
    found: bool
    finding_id: str | None = None
    severity: str | None = None
    ssvc_action: str | None = None
    triage_confidence: float | None = None
    status: str | None = None
    known_status: str | None = None


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------


class _ClientAllowlistMiddleware(Middleware):
    """Reject tool calls from clients not in the allowlist."""

    def __init__(self, allowlist: frozenset[str]) -> None:
        self._allowlist = allowlist

    async def on_call_tool(
        self,
        context: MiddlewareContext[mcp_types.CallToolRequestParams],
        call_next: CallNext[mcp_types.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        # Defensive traversal: FastMCP's context structure varies across
        # versions; not all attributes are guaranteed to exist.
        client_name: str | None = None
        fmcp_ctx = getattr(context, "fastmcp_context", None)
        if fmcp_ctx is not None:
            client_info = getattr(fmcp_ctx, "client_info", None)
            if client_info is not None:
                client_name = getattr(client_info, "name", None)
        if client_name not in self._allowlist:
            _LOG.warning("mcp_client_rejected", client_name=client_name)
            msg = "client not in allowlist"
            raise PermissionError(msg)
        return await call_next(context)


def _sanitize_optional(text: str | None, *, max_chars: int = 2000) -> str | None:
    if text is None:
        return None
    return sanitize_text(text, max_chars=max_chars)


def _sanitize_evidence(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {k: sanitize_text(v, max_chars=2000) if isinstance(v, str) else v for k, v in evidence.items()}


def _parse_finding_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        msg = f"invalid finding id: {raw!r}"
        raise ValueError(msg) from None


def create_mcp_server(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    client_allowlist: list[str] | None = None,
) -> FastMCP:
    """Build the read-only MCP server with tools bound to *session_factory*.

    Parameters
    ----------
    session_factory:
        Async SQLAlchemy session factory for DB queries.
    client_allowlist:
        If non-empty, only clients whose ``client_info.name`` appears in
        this list are permitted.  An empty or ``None`` list disables
        filtering (all clients allowed).
    """
    mcp = FastMCP(
        "Security Scout (read-only)",
        instructions=(
            "Security Scout read-only server. "
            "Query vulnerability findings, check dependency risk, "
            "and inspect triage status. No write operations."
        ),
    )

    allowlist: frozenset[str] = frozenset(client_allowlist) if client_allowlist else frozenset()

    if allowlist:
        mcp.add_middleware(_ClientAllowlistMiddleware(allowlist))

    # ------------------------------------------------------------------
    # Tool: query_findings
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def query_findings(
        repo: str,
        severity: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[FindingSummary]:
        """List findings for a repository, optionally filtered by severity or status.

        Args:
            repo: Source reference prefix to match (e.g. "owner/repo").
            severity: Filter by severity level (critical, high, medium, low, informational).
            status: Filter by finding status (confirmed_high, confirmed_low, unconfirmed, false_positive, accepted_risk).
            limit: Maximum number of results (1-200, default 50).
        """
        limit = max(1, min(limit, 200))

        if severity is not None:
            try:
                Severity(severity.lower())
            except ValueError:
                msg = f"invalid severity: {severity!r} — use one of: critical, high, medium, low, informational"
                raise ValueError(msg) from None

        if status is not None:
            try:
                FindingStatus(status.lower())
            except ValueError:
                valid = ", ".join(s.value for s in FindingStatus)
                msg = f"invalid status: {status!r} — use one of: {valid}"
                raise ValueError(msg) from None

        stmt = select(Finding).where(Finding.source_ref.contains(repo))

        if severity is not None:
            stmt = stmt.where(Finding.severity == Severity(severity.lower()))
        if status is not None:
            stmt = stmt.where(Finding.status == FindingStatus(status.lower()))

        stmt = stmt.order_by(Finding.created_at.desc()).limit(limit)

        async with session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        _LOG.info(
            "mcp_query_findings",
            metric_name="mcp_query_findings",
            repo=repo,
            severity=severity,
            result_count=len(rows),
        )

        return [
            FindingSummary(
                id=str(row.id),
                title=sanitize_text(row.title, max_chars=500),
                severity=row.severity.value,
                ssvc_action=row.ssvc_action.value if row.ssvc_action else None,
                status=row.status.value,
                triage_confidence=row.triage_confidence,
                source_ref=row.source_ref,
                created_at=row.created_at,
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Tool: get_finding_detail
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def get_finding_detail(finding_id: str) -> FindingDetail:
        """Get full detail for a single finding by its UUID.

        Args:
            finding_id: The UUID of the finding.
        """
        fid = _parse_finding_id(finding_id)

        async with session_factory() as session:
            row = await session.get(Finding, fid)

        if row is None:
            msg = f"finding not found: {finding_id}"
            raise ValueError(msg)

        _LOG.info("mcp_get_finding_detail", metric_name="mcp_get_finding_detail", finding_id=finding_id)

        return FindingDetail(
            id=str(row.id),
            title=sanitize_text(row.title, max_chars=500),
            description=_sanitize_optional(row.description),
            severity=row.severity.value,
            ssvc_action=row.ssvc_action.value if row.ssvc_action else None,
            status=row.status.value,
            triage_confidence=row.triage_confidence,
            source_ref=row.source_ref,
            cve_id=row.cve_id,
            cwe_ids=row.cwe_ids,
            cvss_score=row.cvss_score,
            cvss_vector=row.cvss_vector,
            known_status=row.known_status.value if row.known_status else None,
            duplicate_of=row.duplicate_of,
            duplicate_url=row.duplicate_url,
            reproduction=_sanitize_optional(row.reproduction),
            evidence=_sanitize_evidence(row.evidence),
            approved_by=row.approved_by,
            approved_at=row.approved_at,
            created_at=row.created_at,
        )

    # ------------------------------------------------------------------
    # Tool: check_dependency
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def check_dependency(
        package: str,
        version: str,
        ecosystem: str,
    ) -> DependencyRisk:
        """Search for known advisories matching a package name.

        Performs a case-insensitive name match against finding source
        references.  The version and ecosystem are returned in the
        response for context but are **not** used as query filters.

        Args:
            package: Package name to search for (e.g. "lodash", "requests").
            version: Package version for advisory context (not used as a filter).
            ecosystem: Package ecosystem for advisory context (not used as a filter).
        """
        if not package.strip():
            msg = "package name is required"
            raise ValueError(msg)

        search_term = package.strip().lower()

        async with session_factory() as session:
            result = await session.execute(
                select(Finding).where(
                    Finding.source_ref.icontains(search_term),
                )
            )
            rows = result.scalars().all()

        advisories = [
            DependencyAdvisory(
                id=str(row.id),
                title=sanitize_text(row.title, max_chars=200),
                severity=row.severity.value,
                ssvc_action=row.ssvc_action.value if row.ssvc_action else None,
                source_ref=row.source_ref,
            )
            for row in rows
        ]

        _LOG.info(
            "mcp_check_dependency",
            metric_name="mcp_check_dependency",
            package=package,
            version=version,
            ecosystem=ecosystem,
            advisory_count=len(advisories),
        )

        return DependencyRisk(
            package=package,
            version=version,
            ecosystem=ecosystem,
            advisory_count=len(advisories),
            advisories=advisories,
        )

    # ------------------------------------------------------------------
    # Tool: get_triage_status
    # ------------------------------------------------------------------

    @mcp.tool(
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def get_triage_status(advisory_id: str) -> TriageStatus:
        """Check if an advisory has been triaged and what the outcome was.

        Searches by GHSA ID, CVE ID, or source reference.

        Args:
            advisory_id: Advisory identifier (GHSA-xxxx-xxxx-xxxx or CVE-YYYY-NNNNN).
        """
        if not advisory_id.strip():
            msg = "advisory_id is required"
            raise ValueError(msg)

        normalised = advisory_id.strip().upper()

        async with session_factory() as session:
            stmt = select(Finding).where((Finding.source_ref.icontains(normalised)) | (Finding.cve_id == normalised))
            result = await session.execute(stmt)
            row = result.scalars().first()

        _LOG.info(
            "mcp_get_triage_status",
            metric_name="mcp_get_triage_status",
            advisory_id=advisory_id,
            found=row is not None,
        )

        if row is None:
            return TriageStatus(advisory_id=advisory_id, found=False)

        return TriageStatus(
            advisory_id=advisory_id,
            found=True,
            finding_id=str(row.id),
            severity=row.severity.value,
            ssvc_action=row.ssvc_action.value if row.ssvc_action else None,
            triage_confidence=row.triage_confidence,
            status=row.status.value,
            known_status=row.known_status.value if row.known_status else None,
        )

    return mcp

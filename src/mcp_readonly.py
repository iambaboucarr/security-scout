# SPDX-License-Identifier: Apache-2.0
"""Read-only MCP server exposing finding and dependency queries.

Exposes four tools to MCP clients (Claude Code, Cursor, etc.):
  - query_findings       — list findings for a repo, optionally filtered
  - get_finding_detail   — full detail for a single finding
  - check_dependency     — check known advisories for a package version
  - get_triage_status    — triage outcome for a specific advisory

The query logic, response models, and sanitisation helpers live in
``tools.queries`` so the HTTP API can share the same code path. This
module is a thin protocol adapter: per-call session lifecycle, MCP tool
annotations, client-allowlist middleware.
"""

from __future__ import annotations

import mcp.types as mcp_types
import structlog
from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tools import queries
from tools.queries import (
    DependencyRisk,
    FindingDetail,
    FindingSummary,
    TriageStatus,
)

_LOG = structlog.get_logger(__name__)


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


def create_mcp_server(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    client_allowlist: list[str] | None = None,
) -> FastMCP:
    """Build the read-only MCP server with tools bound to *session_factory*.

    Parameters
    ----------
    session_factory:
        Async SQLAlchemy session factory. A fresh session is opened per
        tool invocation so each call gets a clean transaction scope.
    client_allowlist:
        If non-empty, only clients whose ``client_info.name`` appears in
        this list are permitted. An empty or ``None`` list disables
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
            repo: GitHub ``owner/repo`` slug (compared case-insensitively; stored canonical lowercase).
            severity: Filter by severity level (critical, high, medium, low, informational).
            status: Filter by finding status (confirmed_high, confirmed_low, unconfirmed, false_positive, accepted_risk).
            limit: Maximum number of results (1-200, default 50).
        """
        async with session_factory() as session:
            return await queries.query_findings(
                session,
                repo=repo,
                severity=severity,
                status=status,
                limit=limit,
            )

    @mcp.tool(
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def get_finding_detail(finding_id: str) -> FindingDetail:
        """Get full detail for a single finding by its UUID.

        Args:
            finding_id: The UUID of the finding.
        """
        async with session_factory() as session:
            return await queries.get_finding_detail(session, finding_id=finding_id)

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
        references. The version and ecosystem are returned in the
        response for context but are **not** used as query filters.

        Args:
            package: Package name to search for (e.g. "lodash", "requests").
            version: Package version for advisory context (not used as a filter).
            ecosystem: Package ecosystem for advisory context (not used as a filter).
        """
        async with session_factory() as session:
            return await queries.check_dependency(
                session,
                package=package,
                version=version,
                ecosystem=ecosystem,
            )

    @mcp.tool(
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def get_triage_status(advisory_id: str) -> TriageStatus:
        """Check if an advisory has been triaged and what the outcome was.

        Searches by GHSA ID, CVE ID, or source reference.

        Args:
            advisory_id: Advisory identifier (GHSA-xxxx-xxxx-xxxx or CVE-YYYY-NNNNN).
        """
        async with session_factory() as session:
            return await queries.get_triage_status(session, advisory_id=advisory_id)

    return mcp

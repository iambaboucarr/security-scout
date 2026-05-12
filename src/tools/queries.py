# SPDX-License-Identifier: Apache-2.0
"""Pure async query functions over the finding data model.

This module is the single source of read-side query logic for both the
read-only MCP server (``mcp_readonly``) and HTTP handlers that expose the
same reads. Each function takes an :class:`AsyncSession` so the caller
owns transaction scope — MCP tools open a per-call session, FastAPI routes
inject a per-request session.

Every query:

* Validates input and raises :class:`ValueError` with an
  operator-readable message on rejection.
* Sanitises every string placed in a response model (values loaded from
  the database, user-supplied echo fields such as package coordinates in
  ``check_dependency``, and structured evidence blobs) via
  ``tools.input_sanitiser.sanitize_text`` so tool output is safe to treat
  as untrusted data in downstream prompts.
* Logs a single ``metric_name=...`` structlog event for observability.

Response models are frozen Pydantic instances; mutate by constructing a
new one.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Finding, FindingStatus, Severity
from tools.input_sanitiser import sanitize_text

_LOG = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response models
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
# Sanitisation + parsing helpers (public — used by callers that build their
# own response shapes alongside the query functions)
# ---------------------------------------------------------------------------


def sanitize_optional(text: str | None, *, max_chars: int = 2000) -> str | None:
    if text is None:
        return None
    return sanitize_text(text, max_chars=max_chars)


def sanitize_evidence_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, max_chars=2000)
    if isinstance(value, dict):
        return {k: sanitize_evidence_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_evidence_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_evidence_value(item) for item in value)
    return value


def sanitize_evidence(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {k: sanitize_evidence_value(v) for k, v in evidence.items()}


def parse_finding_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        msg = f"invalid finding id: {raw!r}"
        raise ValueError(msg) from None


def _validate_severity(severity: str) -> Severity:
    try:
        return Severity(severity.lower())
    except ValueError:
        msg = f"invalid severity: {severity!r} — use one of: critical, high, medium, low, informational"
        raise ValueError(msg) from None


def _validate_status(status: str) -> FindingStatus:
    try:
        return FindingStatus(status.lower())
    except ValueError:
        valid = ", ".join(s.value for s in FindingStatus)
        msg = f"invalid status: {status!r} — use one of: {valid}"
        raise ValueError(msg) from None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200
_ECHO_PACKAGE_MAX = 500
_ECHO_VERSION_MAX = 200
_ECHO_ECOSYSTEM_MAX = 64


async def query_findings(
    session: AsyncSession,
    *,
    repo: str,
    severity: str | None = None,
    status: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[FindingSummary]:
    """List findings for *repo*, filtered by severity / status.

    *repo* matches ``Finding.repo_name`` case-insensitively (stored
    canonical lowercase). *limit* is clamped to ``[1, 200]``.
    """
    clamped_limit = max(1, min(limit, _MAX_LIMIT))

    severity_enum = _validate_severity(severity) if severity is not None else None
    status_enum = _validate_status(status) if status is not None else None

    repo_key = repo.strip().lower()
    stmt = select(Finding).where(Finding.repo_name == repo_key)
    if severity_enum is not None:
        stmt = stmt.where(Finding.severity == severity_enum)
    if status_enum is not None:
        stmt = stmt.where(Finding.status == status_enum)
    stmt = stmt.order_by(Finding.created_at.desc()).limit(clamped_limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    _LOG.info(
        "query_findings",
        metric_name="query_findings",
        repo=repo_key,
        severity=severity,
        status=status,
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


async def get_finding_detail(
    session: AsyncSession,
    *,
    finding_id: str,
) -> FindingDetail:
    """Full detail for a single finding. Raises ``ValueError`` on bad UUID or missing row."""
    fid = parse_finding_id(finding_id)
    row = await session.get(Finding, fid)
    if row is None:
        msg = f"finding not found: {finding_id}"
        raise ValueError(msg)

    _LOG.info(
        "get_finding_detail",
        metric_name="get_finding_detail",
        finding_id=finding_id,
    )

    return FindingDetail(
        id=str(row.id),
        title=sanitize_text(row.title, max_chars=500),
        description=sanitize_optional(row.description),
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
        reproduction=sanitize_optional(row.reproduction),
        evidence=sanitize_evidence(row.evidence),
        approved_by=row.approved_by,
        approved_at=row.approved_at,
        created_at=row.created_at,
    )


async def check_dependency(
    session: AsyncSession,
    *,
    package: str,
    version: str,
    ecosystem: str,
) -> DependencyRisk:
    """Search advisories whose ``source_ref`` contains *package* (case-insensitive).

    *version* and *ecosystem* are passed through into the response for
    context but are **not** used as query filters.
    """
    if not package.strip():
        msg = "package name is required"
        raise ValueError(msg)

    search_term = package.strip().lower()
    result = await session.execute(
        select(Finding).where(Finding.source_ref.icontains(search_term)),
    )
    rows = result.scalars().all()

    package_out = sanitize_text(package.strip(), max_chars=_ECHO_PACKAGE_MAX)
    version_out = sanitize_text(version, max_chars=_ECHO_VERSION_MAX)
    ecosystem_out = sanitize_text(ecosystem, max_chars=_ECHO_ECOSYSTEM_MAX)

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
        "check_dependency",
        metric_name="check_dependency",
        package=package_out,
        version=version_out,
        ecosystem=ecosystem_out,
        advisory_count=len(advisories),
    )

    return DependencyRisk(
        package=package_out,
        version=version_out,
        ecosystem=ecosystem_out,
        advisory_count=len(advisories),
        advisories=advisories,
    )


async def get_triage_status(
    session: AsyncSession,
    *,
    advisory_id: str,
) -> TriageStatus:
    """Find the first finding referencing *advisory_id* (GHSA or CVE)."""
    if not advisory_id.strip():
        msg = "advisory_id is required"
        raise ValueError(msg)

    normalised = advisory_id.strip().upper()
    stmt = select(Finding).where(
        (Finding.source_ref.icontains(normalised)) | (Finding.cve_id == normalised),
    )
    result = await session.execute(stmt)
    row = result.scalars().first()

    _LOG.info(
        "get_triage_status",
        metric_name="get_triage_status",
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


__all__ = [
    "DependencyAdvisory",
    "DependencyRisk",
    "FindingDetail",
    "FindingSummary",
    "TriageStatus",
    "check_dependency",
    "get_finding_detail",
    "get_triage_status",
    "parse_finding_id",
    "query_findings",
    "sanitize_evidence",
    "sanitize_evidence_value",
    "sanitize_optional",
]

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import GitHubIssuesTrackerConfig
from models import Finding, FindingStatus, WorkflowKind
from tools.github import (
    GitHubClient,
    GitHubIssueSearchItem,
    normalise_ghsa_id,
    validate_github_repo_name,
    validate_github_repo_owner,
)

_LOG = structlog.get_logger(__name__)

_CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


def normalise_cve_id(raw: str) -> str:
    s = raw.strip()
    if not _CVE_PATTERN.match(s):
        msg = f"invalid CVE id: {raw!r}"
        raise ValueError(msg)
    parts = s.upper().split("-")
    return f"CVE-{parts[1]}-{parts[2]}"


def _try_normalise_cve(cve_id: str | None) -> str | None:
    if cve_id is None:
        return None
    try:
        return normalise_cve_id(cve_id)
    except ValueError:
        return None


def _try_normalise_ghsa(ghsa_id: str | None) -> str | None:
    if ghsa_id is None:
        return None
    try:
        return normalise_ghsa_id(ghsa_id)
    except ValueError:
        return None


def _normalise_cwe_token(raw: str) -> str:
    s = raw.strip().upper()
    if s.startswith("CWE-"):
        return s
    if s.isdigit():
        return f"CWE-{s}"
    return s


def _label_query_fragment(label: str) -> str:
    if label.isalnum() and label.isascii():
        return f"label:{label}"
    esc = label.replace('"', '\\"')
    return f'label:"{esc}"'


def _github_issue_match_status(state: str) -> str:
    s = state.lower()
    if s == "open":
        return "open"
    if s == "closed":
        return "resolved"
    return s


def _scout_finding_status_label(status: FindingStatus) -> str:
    match status:
        case FindingStatus.accepted_risk:
            return "accepted_risk"
        case FindingStatus.false_positive:
            return "resolved"
        case FindingStatus.confirmed_high | FindingStatus.confirmed_low:
            return "resolved"
        case FindingStatus.unconfirmed:
            return "open"
    return status.value


def _issue_text_contains_identifier(
    item: GitHubIssueSearchItem,
    *,
    cve: str | None,
    ghsa: str | None,
) -> bool:
    hay = f"{item.title}\n{item.body or ''}".upper()
    return (cve is not None and cve.upper() in hay) or (ghsa is not None and ghsa.upper() in hay)


def _finding_has_cwe(row: Finding, cwe: str) -> bool:
    ids = row.cwe_ids
    if not ids:
        return False
    want = _normalise_cwe_token(cwe)
    return any(_normalise_cwe_token(x) == want for x in ids)


class TrackerMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tracker: str
    issue_id: str
    issue_url: str | None
    title: str
    status: str
    match_tier: int = Field(ge=1, le=3)
    match_field: str
    matched_value: str
    last_updated: datetime | None = None


@runtime_checkable
class IssueTrackerAdapter(Protocol):
    async def search_known_vulnerability(
        self,
        cve_id: str | None,
        ghsa_id: str | None,
        cwe_id: str | None,
        affected_package: str | None,
        affected_versions: str | None,
        summary: str | None,
    ) -> list[TrackerMatch]: ...


def _github_should_skip_duplicate_url(url: str, seen_urls: set[str]) -> bool:
    if not url:
        return False
    if url in seen_urls:
        return True
    seen_urls.add(url)
    return False


def _github_issue_to_match(
    owner: str,
    repo: str,
    item: GitHubIssueSearchItem,
    *,
    match_field: str,
    matched_value: str,
) -> TrackerMatch:
    return TrackerMatch(
        tracker="github_issues",
        issue_id=f"{owner}/{repo}#{item.number}",
        issue_url=item.html_url or None,
        title=item.title,
        status=_github_issue_match_status(item.state),
        match_tier=1,
        match_field=match_field,
        matched_value=matched_value,
        last_updated=item.updated_at,
    )


async def _github_append_tier1_matches(
    client: GitHubClient,
    build_query: Callable[[str], str],
    owner: str,
    repo: str,
    search_token: str,
    seen_urls: set[str],
    matches: list[TrackerMatch],
    *,
    match_field: str,
    matched_value: str,
    id_cve: str | None,
    id_ghsa: str | None,
) -> None:
    q = build_query(search_token)
    items = await client.search_issues(q, per_page=30, page=1)
    for it in items:
        if not _issue_text_contains_identifier(it, cve=id_cve, ghsa=id_ghsa):
            continue
        url = it.html_url or ""
        if _github_should_skip_duplicate_url(url, seen_urls):
            continue
        matches.append(
            _github_issue_to_match(
                owner,
                repo,
                it,
                match_field=match_field,
                matched_value=matched_value,
            ),
        )


def _scout_match_from_finding(
    row: Finding,
    *,
    tier: int,
    match_field: str,
    matched_value: str,
) -> TrackerMatch:
    return TrackerMatch(
        tracker="scout_history",
        issue_id=str(row.id),
        issue_url=None,
        title=row.title,
        status=_scout_finding_status_label(row.status),
        match_tier=tier,
        match_field=match_field,
        matched_value=matched_value,
        last_updated=row.created_at,
    )


async def _scout_append_cve_matches(
    session: AsyncSession,
    exclude: uuid.UUID | None,
    cve: str,
    seen_ids: set[uuid.UUID],
    matches: list[TrackerMatch],
) -> None:
    stmt = select(Finding).where(Finding.cve_id == cve)
    if exclude is not None:
        stmt = stmt.where(Finding.id != exclude)
    result = await session.execute(stmt)
    for row in result.scalars():
        if row.id in seen_ids:
            continue
        seen_ids.add(row.id)
        matches.append(_scout_match_from_finding(row, tier=1, match_field="cve_id", matched_value=cve))


async def _scout_append_ghsa_matches(
    session: AsyncSession,
    exclude: uuid.UUID | None,
    ghsa: str,
    seen_ids: set[uuid.UUID],
    matches: list[TrackerMatch],
) -> None:
    stmt = select(Finding).where(Finding.workflow == WorkflowKind.advisory)
    if exclude is not None:
        stmt = stmt.where(Finding.id != exclude)
    result = await session.execute(stmt)
    for row in result.scalars():
        if ghsa.upper() not in row.source_ref.upper():
            continue
        if row.id in seen_ids:
            continue
        seen_ids.add(row.id)
        matches.append(_scout_match_from_finding(row, tier=1, match_field="ghsa_id", matched_value=ghsa))


async def _scout_append_cwe_matches(
    session: AsyncSession,
    exclude: uuid.UUID | None,
    cwe_token: str,
    seen_ids: set[uuid.UUID],
    matches: list[TrackerMatch],
) -> None:
    stmt = select(Finding).where(Finding.workflow == WorkflowKind.advisory)
    if exclude is not None:
        stmt = stmt.where(Finding.id != exclude)
    result = await session.execute(stmt)
    for row in result.scalars():
        if not _finding_has_cwe(row, cwe_token):
            continue
        if row.id in seen_ids:
            continue
        seen_ids.add(row.id)
        matches.append(_scout_match_from_finding(row, tier=2, match_field="cwe_id", matched_value=cwe_token))


class GitHubIssuesAdapter:
    def __init__(
        self,
        client: GitHubClient,
        owner: str,
        repo: str,
        tracker_cfg: GitHubIssuesTrackerConfig,
    ) -> None:
        self._client = client
        self._owner = validate_github_repo_owner(owner)
        self._repo = validate_github_repo_name(repo)
        self._cfg = tracker_cfg

    def _build_query(self, search_token: str) -> str:
        parts = [
            f"repo:{self._owner}/{self._repo}",
            "is:issue",
            _label_query_fragment(self._cfg.security_label),
        ]
        if not self._cfg.search_closed:
            parts.append("is:open")
        parts.append(search_token)
        return " ".join(parts)

    async def search_known_vulnerability(
        self,
        cve_id: str | None,
        ghsa_id: str | None,
        cwe_id: str | None,
        affected_package: str | None,
        affected_versions: str | None,
        summary: str | None,
    ) -> list[TrackerMatch]:
        _ = cwe_id, affected_package, affected_versions, summary
        matches: list[TrackerMatch] = []
        seen_urls: set[str] = set()
        cve = _try_normalise_cve(cve_id)
        ghsa = _try_normalise_ghsa(ghsa_id)
        if cve is not None:
            await _github_append_tier1_matches(
                self._client,
                self._build_query,
                self._owner,
                self._repo,
                cve,
                seen_urls,
                matches,
                match_field="cve_id",
                matched_value=cve,
                id_cve=cve,
                id_ghsa=None,
            )
        if ghsa is not None:
            await _github_append_tier1_matches(
                self._client,
                self._build_query,
                self._owner,
                self._repo,
                ghsa,
                seen_urls,
                matches,
                match_field="ghsa_id",
                matched_value=ghsa,
                id_cve=None,
                id_ghsa=ghsa,
            )
        return matches


class ScoutHistoricalAdapter:
    def __init__(
        self,
        session: AsyncSession,
        *,
        exclude_finding_id: uuid.UUID | None = None,
    ) -> None:
        self._session = session
        self._exclude = exclude_finding_id

    async def search_known_vulnerability(
        self,
        cve_id: str | None,
        ghsa_id: str | None,
        cwe_id: str | None,
        affected_package: str | None,
        affected_versions: str | None,
        summary: str | None,
    ) -> list[TrackerMatch]:
        _ = affected_package, affected_versions, summary
        matches: list[TrackerMatch] = []
        seen_ids: set[uuid.UUID] = set()
        cve = _try_normalise_cve(cve_id)
        ghsa = _try_normalise_ghsa(ghsa_id)
        if cve is not None:
            await _scout_append_cve_matches(self._session, self._exclude, cve, seen_ids, matches)
        if ghsa is not None:
            await _scout_append_ghsa_matches(self._session, self._exclude, ghsa, seen_ids, matches)
        if len(matches) == 0 and cwe_id is not None:
            token = _normalise_cwe_token(cwe_id)
            await _scout_append_cwe_matches(self._session, self._exclude, token, seen_ids, matches)
        return matches


def dedupe_tracker_matches(matches: Sequence[TrackerMatch]) -> list[TrackerMatch]:
    seen: set[str] = set()
    out: list[TrackerMatch] = []
    for m in matches:
        key = m.issue_url or f"{m.tracker}:{m.issue_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


async def run_dedup_checks(
    *,
    cve_id: str | None,
    ghsa_id: str | None,
    cwe_id: str | None,
    affected_package: str | None,
    affected_versions: str | None,
    summary: str | None,
    adapters: Sequence[IssueTrackerAdapter],
) -> list[TrackerMatch]:
    started = time.perf_counter()
    combined: list[TrackerMatch] = []
    for adapter in adapters:
        batch = await adapter.search_known_vulnerability(
            cve_id,
            ghsa_id,
            cwe_id,
            affected_package,
            affected_versions,
            summary,
        )
        combined.extend(batch)
        for m in batch:
            _LOG.info(
                "dedup_match_total",
                metric_name="dedup_match_total",
                tier=m.match_tier,
                tracker=m.tracker,
                match_field=m.match_field,
            )
    elapsed = time.perf_counter() - started
    _LOG.info(
        "dedup_latency_seconds",
        metric_name="dedup_latency_seconds",
        duration_seconds=elapsed,
        adapter_count=len(adapters),
        match_count=len(combined),
    )
    return dedupe_tracker_matches(combined)


__all__ = [
    "GitHubIssuesAdapter",
    "IssueTrackerAdapter",
    "ScoutHistoricalAdapter",
    "TrackerMatch",
    "dedupe_tracker_matches",
    "normalise_cve_id",
    "run_dedup_checks",
]

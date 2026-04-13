from __future__ import annotations

import base64
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import GitHubIssuesTrackerConfig, JiraTrackerConfig, LinearTrackerConfig
from models import Finding, FindingStatus, WorkflowKind
from tools.github import validate_github_repo_name, validate_github_repo_owner
from tools.scm import IssueSearchItem, SCMProvider, normalise_ghsa_id

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
    item: IssueSearchItem,
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
    item: IssueSearchItem,
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
    scm: SCMProvider,
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
    items = await scm.search_issues(q, per_page=30, page=1)
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
        scm: SCMProvider,
        owner: str,
        repo: str,
        tracker_cfg: GitHubIssuesTrackerConfig,
    ) -> None:
        self._scm = scm
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
                self._scm,
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
                self._scm,
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


@dataclass(frozen=True, slots=True)
class IssueTrackerCredentials:
    """Credentials for non-GitHub issue trackers, sourced from environment via Settings."""

    jira_email: str | None = None
    jira_api_token: str | None = None
    linear_api_key: str | None = None


_JIRA_RESERVED_CHARS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _jira_escape_text(value: str) -> str:
    """Escape JQL reserved characters and quotes for use inside a quoted text literal."""
    return _JIRA_RESERVED_CHARS.sub(lambda m: "\\" + m.group(0), value)


def _jira_basic_or_bearer_header(creds: IssueTrackerCredentials) -> dict[str, str]:
    if not creds.jira_api_token:
        msg = "JIRA adapter requires a token (set JIRA_API_TOKEN)"
        raise ValueError(msg)
    if creds.jira_email:
        userpass = f"{creds.jira_email}:{creds.jira_api_token}".encode()
        return {"Authorization": "Basic " + base64.b64encode(userpass).decode("ascii")}
    return {"Authorization": f"Bearer {creds.jira_api_token}"}


def _jira_status_to_match_status(category: str | None, name: str | None) -> str:
    cat = (category or "").lower()
    if cat == "done":
        return "resolved"
    if cat == "indeterminate":
        return "in_progress"
    if cat == "new":
        return "open"
    return (name or "open").lower()


def _jira_issue_to_match(
    issue: dict[str, Any],
    base_url: str,
    *,
    match_field: str,
    matched_value: str,
) -> TrackerMatch | None:
    key = issue.get("key")
    if not isinstance(key, str) or not key:
        return None
    fields = issue.get("fields") or {}
    if not isinstance(fields, dict):
        fields = {}
    summary_raw = fields.get("summary")
    summary = summary_raw if isinstance(summary_raw, str) else key
    status = fields.get("status") or {}
    status_name = status.get("name") if isinstance(status, dict) else None
    category = None
    if isinstance(status, dict):
        cat = status.get("statusCategory")
        if isinstance(cat, dict):
            category = cat.get("key")
    updated_raw = fields.get("updated")
    updated_at: datetime | None = None
    if isinstance(updated_raw, str):
        try:
            updated_at = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
        except ValueError:
            updated_at = None
    return TrackerMatch(
        tracker="jira",
        issue_id=key,
        issue_url=f"{base_url.rstrip('/')}/browse/{key}",
        title=summary,
        status=_jira_status_to_match_status(category, status_name if isinstance(status_name, str) else None),
        match_tier=1,
        match_field=match_field,
        matched_value=matched_value,
        last_updated=updated_at,
    )


class JiraIssuesAdapter:
    """JIRA Cloud / Server adapter using the v3 REST search API.

    Auth: Basic (email + API token) for Cloud, Bearer (PAT only) for Server. The token comes from the
    ``JIRA_API_TOKEN`` environment variable via :class:`IssueTrackerCredentials`; the adapter is
    skipped when no token is configured.
    """

    _SEARCH_PATH = "/rest/api/3/search"

    def __init__(
        self,
        client: httpx.AsyncClient,
        cfg: JiraTrackerConfig,
        creds: IssueTrackerCredentials,
    ) -> None:
        self._client = client
        self._cfg = cfg
        self._auth_headers = _jira_basic_or_bearer_header(creds)

    def _build_jql(self, search_token: str) -> str:
        escaped = _jira_escape_text(search_token)
        return f'project = "{self._cfg.project_key}" AND (summary ~ "{escaped}" OR labels = "{escaped}")'

    async def _search(self, jql: str) -> list[dict[str, Any]]:
        url = f"{self._cfg.base_url.rstrip('/')}{self._SEARCH_PATH}"
        response = await self._client.get(
            url,
            params={
                "jql": jql,
                "fields": "summary,status,updated",
                "maxResults": 30,
            },
            headers={**self._auth_headers, "Accept": "application/json"},
        )
        if response.status_code >= 400:
            _LOG.warning(
                "jira_search_http_error",
                metric_name="api_error_total",
                api="jira",
                http_status=response.status_code,
            )
            return []
        try:
            payload = response.json()
        except ValueError:
            _LOG.warning("jira_search_invalid_json", api="jira")
            return []
        if not isinstance(payload, dict):
            return []
        issues = payload.get("issues")
        return issues if isinstance(issues, list) else []

    async def _append_matches(
        self,
        token: str,
        match_field: str,
        seen: set[str],
        out: list[TrackerMatch],
    ) -> None:
        issues = await self._search(self._build_jql(token))
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            match = _jira_issue_to_match(
                issue,
                self._cfg.base_url,
                match_field=match_field,
                matched_value=token,
            )
            if match is None or match.issue_id in seen:
                continue
            seen.add(match.issue_id)
            out.append(match)

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
        seen: set[str] = set()
        cve = _try_normalise_cve(cve_id)
        ghsa = _try_normalise_ghsa(ghsa_id)
        if cve is not None:
            await self._append_matches(cve, "cve_id", seen, matches)
        if ghsa is not None:
            await self._append_matches(ghsa, "ghsa_id", seen, matches)
        return matches


_LINEAR_ENDPOINT = "https://api.linear.app/graphql"

_LINEAR_QUERY = """
query SecurityScoutDedup($filter: IssueFilter!) {
  issues(filter: $filter, first: 30) {
    nodes {
      id
      identifier
      title
      url
      updatedAt
      state { name type }
    }
  }
}
""".strip()


def _linear_state_to_match_status(state_type: str | None, state_name: str | None) -> str:
    t = (state_type or "").lower()
    if t == "completed":
        return "resolved"
    if t == "canceled":
        return "wont_fix"
    if t == "started":
        return "in_progress"
    if t in ("backlog", "unstarted", "triage"):
        return "open"
    return (state_name or "open").lower()


def _linear_node_to_match(
    node: dict[str, Any],
    *,
    match_field: str,
    matched_value: str,
) -> TrackerMatch | None:
    identifier = node.get("identifier")
    if not isinstance(identifier, str) or not identifier:
        return None
    title_raw = node.get("title")
    title = title_raw if isinstance(title_raw, str) else identifier
    url_raw = node.get("url")
    url = url_raw if isinstance(url_raw, str) else None
    updated_raw = node.get("updatedAt")
    updated_at: datetime | None = None
    if isinstance(updated_raw, str):
        try:
            updated_at = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
        except ValueError:
            updated_at = None
    state = node.get("state") or {}
    state_type = state.get("type") if isinstance(state, dict) else None
    state_name = state.get("name") if isinstance(state, dict) else None
    return TrackerMatch(
        tracker="linear",
        issue_id=identifier,
        issue_url=url,
        title=title,
        status=_linear_state_to_match_status(
            state_type if isinstance(state_type, str) else None,
            state_name if isinstance(state_name, str) else None,
        ),
        match_tier=1,
        match_field=match_field,
        matched_value=matched_value,
        last_updated=updated_at,
    )


def _linear_filter(team_id: str, label_name: str, token: str) -> dict[str, Any]:
    return {
        "team": {"id": {"eq": team_id}},
        "labels": {"name": {"eq": label_name}},
        "or": [
            {"title": {"containsIgnoreCase": token}},
            {"description": {"containsIgnoreCase": token}},
        ],
    }


class LinearIssuesAdapter:
    """Linear adapter using the GraphQL API.

    Filters by team + label, then matches CVE/GHSA tokens against issue title or description text.
    Auth uses the ``LINEAR_API_KEY`` environment variable via :class:`IssueTrackerCredentials`;
    the adapter is skipped when no key is configured.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        cfg: LinearTrackerConfig,
        creds: IssueTrackerCredentials,
    ) -> None:
        if not creds.linear_api_key:
            msg = "Linear adapter requires an API key (set LINEAR_API_KEY)"
            raise ValueError(msg)
        self._client = client
        self._cfg = cfg
        self._auth_header = creds.linear_api_key

    async def _search(self, token: str) -> list[dict[str, Any]]:
        body = {
            "query": _LINEAR_QUERY,
            "variables": {"filter": _linear_filter(self._cfg.team_id, self._cfg.label_name, token)},
        }
        response = await self._client.post(
            _LINEAR_ENDPOINT,
            json=body,
            headers={
                "Authorization": self._auth_header,
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            _LOG.warning(
                "linear_search_http_error",
                metric_name="api_error_total",
                api="linear",
                http_status=response.status_code,
            )
            return []
        try:
            payload = response.json()
        except ValueError:
            _LOG.warning("linear_search_invalid_json", api="linear")
            return []
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("errors"), list):
            _LOG.warning(
                "linear_graphql_error",
                metric_name="api_error_total",
                api="linear",
            )
            return []
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        issues = data.get("issues")
        if not isinstance(issues, dict):
            return []
        nodes = issues.get("nodes")
        return nodes if isinstance(nodes, list) else []

    async def _append_matches(
        self,
        token: str,
        match_field: str,
        seen: set[str],
        out: list[TrackerMatch],
    ) -> None:
        for node in await self._search(token):
            if not isinstance(node, dict):
                continue
            match = _linear_node_to_match(
                node,
                match_field=match_field,
                matched_value=token,
            )
            if match is None or match.issue_id in seen:
                continue
            seen.add(match.issue_id)
            out.append(match)

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
        seen: set[str] = set()
        cve = _try_normalise_cve(cve_id)
        ghsa = _try_normalise_ghsa(ghsa_id)
        if cve is not None:
            await self._append_matches(cve, "cve_id", seen, matches)
        if ghsa is not None:
            await self._append_matches(ghsa, "ghsa_id", seen, matches)
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
    "IssueTrackerCredentials",
    "JiraIssuesAdapter",
    "LinearIssuesAdapter",
    "ScoutHistoricalAdapter",
    "TrackerMatch",
    "dedupe_tracker_matches",
    "normalise_cve_id",
    "run_dedup_checks",
]

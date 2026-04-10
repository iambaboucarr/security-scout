"""GitHub REST read API: advisories, PR metadata, repo metadata.

Uses ``httpx`` (async I/O). PyGitHub is synchronous-only and unused here. Writes must
respect ``RepoConfig.mode`` (ADR-015); reads are allowed in all modes.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Literal, Self, cast

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from exceptions import SecurityScoutError

__all__ = [
    "AdvisoryData",
    "GitHubAPIError",
    "GitHubClient",
    "GitHubInvalidRepoSlugError",
    "GitHubMalformedResponseError",
    "PullRequestFileInfo",
    "PullRequestInfo",
    "RepositoryMetadata",
    "normalise_ghsa_id",
    "validate_github_repo_name",
    "validate_github_repo_owner",
]


_GHSA_ID_PATTERN = re.compile(r"^GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}$", re.IGNORECASE)
# GitHub login / org: alphanumeric and single internal hyphens; max 39 (docs).
_REPO_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,37}[a-zA-Z0-9])?$")
# Repository name: letters, digits, ., -, _ ; max 100.
_REPO_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")

_MAX_PR_FILES_PAGES = 500

_DEFAULT_API_VERSION = "2022-11-28"
_DEFAULT_TIMEOUT = httpx.Timeout(30.0)
_LOG = structlog.get_logger(__name__)


class GitHubAPIError(SecurityScoutError):
    def __init__(
        self,
        message: str | None = None,
        *,
        is_transient: bool,
        http_status: int | None = None,
        github_request_id: str | None = None,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=is_transient,
            is_resource_error=False,
        )
        self.http_status = http_status
        self.github_request_id = github_request_id

    @classmethod
    def from_status(cls, status: int, message: str) -> GitHubAPIError:
        transient = status in (408, 425, 429, 500, 502, 503, 504)
        return cls(message, is_transient=transient, http_status=status)

    @classmethod
    def from_httpx_response(
        cls,
        response: httpx.Response,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> GitHubAPIError:
        status = response.status_code
        transient = status in (408, 425, 429, 500, 502, 503, 504)
        if not transient and _looks_like_github_rate_limit(response):
            transient = True
        request_id = response.headers.get("x-github-request-id")
        message = _message_from_error_body(response)
        if not message:
            message = response.reason_phrase or f"HTTP {status}"
        return cls(
            message,
            is_transient=transient,
            http_status=status,
            github_request_id=request_id,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )


def _looks_like_github_rate_limit(response: httpx.Response) -> bool:
    """Treat 403 as retryable when GitHub signals rate limiting (primary or secondary)."""
    if response.status_code != 403:
        return False
    if response.headers.get("x-ratelimit-remaining") == "0":
        return True
    if response.headers.get("retry-after"):
        return True
    body = _message_from_error_body(response).lower()
    return "rate limit" in body or "secondary rate" in body


class GitHubMalformedResponseError(SecurityScoutError):
    """HTTP 200 with a JSON shape that does not match the expected GitHub contract."""

    def __init__(
        self,
        message: str | None = None,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=False,
            is_resource_error=False,
        )


class GitHubInvalidRepoSlugError(SecurityScoutError):
    """``owner`` or ``repo`` path segment is not a valid GitHub slug (config injection guard)."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message, is_transient=False, is_resource_error=False)


def _parse_github_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            _LOG.warning("github_datetime_parse_failed", raw_preview=value[:80] if len(value) > 80 else value)
            return None
    return None


def _message_from_error_body(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()
    if not isinstance(data, dict):
        return response.text.strip()
    msg = data.get("message")
    if isinstance(msg, str) and msg:
        return msg
    errors = data.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            maybe = first.get("message")
            if isinstance(maybe, str) and maybe:
                return maybe
    return response.text.strip()


def normalise_ghsa_id(raw: str) -> str:
    """Return canonical ``GHSA-xxxx-xxxx-xxxx`` (raises if shape is invalid)."""
    s = raw.strip()
    if not _GHSA_ID_PATTERN.match(s):
        msg = f"invalid GHSA id: {raw!r}"
        raise ValueError(msg)
    parts = s.split("-")
    return f"GHSA-{'-'.join(p.upper() for p in parts[1:])}"


def validate_github_repo_owner(owner: str) -> str:
    """Return stripped owner/org slug; raises ``GitHubInvalidRepoSlugError`` if invalid."""
    s = owner.strip()
    if not _REPO_OWNER_PATTERN.fullmatch(s):
        msg = f"invalid GitHub owner or org slug: {owner!r}"
        raise GitHubInvalidRepoSlugError(msg)
    return s


def validate_github_repo_name(repo: str) -> str:
    """Return stripped repository name; raises ``GitHubInvalidRepoSlugError`` if invalid."""
    s = repo.strip()
    if not _REPO_NAME_PATTERN.fullmatch(s):
        msg = f"invalid GitHub repository name: {repo!r}"
        raise GitHubInvalidRepoSlugError(msg)
    return s


def _require_pull_number(pull_number: int) -> int:
    if pull_number < 1:
        msg = f"pull_number must be >= 1, got {pull_number}"
        raise ValueError(msg)
    return pull_number


class AdvisoryData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ghsa_id: str
    source: Literal["repository", "global"]
    summary: str
    description: str
    severity: str | None = None
    cve_ids: tuple[str, ...] = ()
    cwe_ids: tuple[str, ...] = ()
    html_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None


class PullRequestInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    state: str
    head_sha: str
    base_sha: str
    user_login: str | None = None
    html_url: str
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


class PullRequestFileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    status: str
    patch: str | None = None
    additions: int = 0
    deletions: int = 0
    sha: str | None = None


class RepositoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    full_name: str
    description: str | None = None
    default_branch: str
    private: bool
    html_url: str
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    language: str | None = None
    pushed_at: datetime | None = None


class _GitHubClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    base_url: str = "https://api.github.com"
    api_version: str = _DEFAULT_API_VERSION


def _auth_headers(token: str, api_version: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": api_version,
    }


def _cve_ids_from_identifiers(identifiers: object) -> tuple[str, ...]:
    if not isinstance(identifiers, list):
        return ()
    out: list[str] = []
    for item in identifiers:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "CVE" and isinstance(item.get("value"), str):
            out.append(item["value"])
    return tuple(out)


def _cwe_ids_from_cwes(cwes: object) -> tuple[str, ...]:
    if not isinstance(cwes, list):
        return ()
    out: list[str] = []
    for item in cwes:
        if not isinstance(item, dict):
            continue
        cid = item.get("cwe_id")
        if isinstance(cid, str) and cid:
            out.append(cid)
    return tuple(out)


def _advisory_from_payload(
    data: dict[str, Any],
    *,
    source: Literal["repository", "global"],
    finding_id: str | None = None,
    workflow_run_id: uuid.UUID | str | None = None,
) -> AdvisoryData:
    ghsa = data.get("ghsa_id")
    if not isinstance(ghsa, str) or not ghsa:
        msg = "GitHub advisory payload missing ghsa_id"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    summary = data.get("summary")
    description = data.get("description")
    if not isinstance(summary, str):
        summary = ""
    if not isinstance(description, str):
        description = ""
    severity = data.get("severity")
    sev = severity if isinstance(severity, str) else None
    html = data.get("html_url")
    html_url = html if isinstance(html, str) else None

    return AdvisoryData(
        ghsa_id=ghsa,
        source=source,
        summary=summary,
        description=description,
        severity=sev,
        cve_ids=_cve_ids_from_identifiers(data.get("identifiers")),
        cwe_ids=_cwe_ids_from_cwes(data.get("cwes")),
        html_url=html_url,
        published_at=_parse_github_datetime(data.get("published_at")),
        updated_at=_parse_github_datetime(data.get("updated_at")),
    )


class GitHubClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        api_version: str = _DEFAULT_API_VERSION,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = _GitHubClientConfig(token=token, base_url=base_url, api_version=api_version)
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> Self:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._cfg.base_url.rstrip("/"),
                headers=_auth_headers(self._cfg.token, self._cfg.api_version),
                timeout=_DEFAULT_TIMEOUT,
            )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None if self._owns_client else self._client

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "GitHubClient must be used as a context manager or constructed with a client"
            raise RuntimeError(msg)
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> httpx.Response:
        client = self._client_or_raise()
        if self._owns_client:
            response = await client.request(method, path, params=params)
        else:
            response = await client.request(
                method,
                path,
                params=params,
                headers=_auth_headers(self._cfg.token, self._cfg.api_version),
            )
        if response.is_success:
            return response
        err = GitHubAPIError.from_httpx_response(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        _LOG.warning(
            "github_api_error",
            http_status=err.http_status,
            github_request_id=err.github_request_id,
            path=path,
        )
        raise err

    async def fetch_repository_security_advisory(
        self,
        owner: str,
        repo: str,
        ghsa_id: str,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> AdvisoryData:
        o = validate_github_repo_owner(owner)
        r = validate_github_repo_name(repo)
        ghsa = normalise_ghsa_id(ghsa_id)
        path = f"/repos/{o}/{r}/security-advisories/{ghsa}"
        response = await self._request(
            "GET",
            path,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        payload = _as_json_object(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        return _advisory_from_payload(
            payload,
            source="repository",
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def fetch_global_security_advisory(
        self,
        ghsa_id: str,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> AdvisoryData:
        ghsa = normalise_ghsa_id(ghsa_id)
        path = f"/advisories/{ghsa}"
        response = await self._request(
            "GET",
            path,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        payload = _as_json_object(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        return _advisory_from_payload(
            payload,
            source="global",
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def fetch_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> PullRequestInfo:
        o = validate_github_repo_owner(owner)
        r = validate_github_repo_name(repo)
        pn = _require_pull_number(pull_number)
        path = f"/repos/{o}/{r}/pulls/{pn}"
        response = await self._request(
            "GET",
            path,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        payload = _as_json_object(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        return _pull_request_from_payload(
            payload,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def fetch_pull_request_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[PullRequestFileInfo, ...]:
        o = validate_github_repo_owner(owner)
        r = validate_github_repo_name(repo)
        pn = _require_pull_number(pull_number)
        path = f"/repos/{o}/{r}/pulls/{pn}/files"
        collected: list[PullRequestFileInfo] = []
        page = 1
        while True:
            if page > _MAX_PR_FILES_PAGES:
                msg = f"pull request files exceeded {_MAX_PR_FILES_PAGES} pages"
                raise GitHubMalformedResponseError(
                    msg,
                    finding_id=finding_id,
                    workflow_run_id=workflow_run_id,
                )
            response = await self._request(
                "GET",
                path,
                params={"per_page": 100, "page": page},
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
            )
            batch = _as_json_array(
                response,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
            )
            for item in batch:
                if not isinstance(item, dict):
                    continue
                try:
                    collected.append(PullRequestFileInfo.model_validate(item))
                except ValidationError as exc:
                    msg = "pull request file entry failed validation"
                    raise GitHubMalformedResponseError(
                        msg,
                        finding_id=finding_id,
                        workflow_run_id=workflow_run_id,
                    ) from exc
            if len(batch) < 100:
                break
            page += 1
        return tuple(collected)

    async def fetch_repository_metadata(
        self,
        owner: str,
        repo: str,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> RepositoryMetadata:
        o = validate_github_repo_owner(owner)
        r = validate_github_repo_name(repo)
        path = f"/repos/{o}/{r}"
        response = await self._request(
            "GET",
            path,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        payload = _as_json_object(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        return _repository_metadata_from_payload(
            payload,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )


def _as_json_object(
    response: httpx.Response,
    *,
    finding_id: str | None = None,
    workflow_run_id: uuid.UUID | str | None = None,
) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        msg = "GitHub API returned non-JSON body"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        ) from exc
    if not isinstance(data, dict):
        msg = "expected JSON object from GitHub API"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    return cast(dict[str, Any], data)


def _as_json_array(
    response: httpx.Response,
    *,
    finding_id: str | None = None,
    workflow_run_id: uuid.UUID | str | None = None,
) -> list[Any]:
    try:
        data = response.json()
    except ValueError as exc:
        msg = "GitHub API returned non-JSON body"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        ) from exc
    if not isinstance(data, list):
        msg = "expected JSON array from GitHub API"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    return data


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, float) and value.is_integer():
        iv = int(value)
        if iv >= 1:
            return iv
    return None


def _pull_request_from_payload(
    data: dict[str, Any],
    *,
    finding_id: str | None = None,
    workflow_run_id: uuid.UUID | str | None = None,
) -> PullRequestInfo:
    num = _coerce_positive_int(data.get("number"))
    if num is None:
        msg = "pull request payload missing number"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    title = data.get("title")
    if not isinstance(title, str):
        title = ""
    state = data.get("state")
    if not isinstance(state, str):
        state = ""
    head = data.get("head")
    base = data.get("base")
    head_sha = ""
    base_sha = ""
    if isinstance(head, dict) and isinstance(head.get("sha"), str):
        head_sha = head["sha"]
    if isinstance(base, dict) and isinstance(base.get("sha"), str):
        base_sha = base["sha"]
    user = data.get("user")
    user_login: str | None = None
    if isinstance(user, dict) and isinstance(user.get("login"), str):
        user_login = user["login"]
    html_url = data.get("html_url")
    if not isinstance(html_url, str):
        html_url = ""
    additions = data.get("additions")
    deletions = data.get("deletions")
    changed = data.get("changed_files")
    return PullRequestInfo(
        number=num,
        title=title,
        state=state,
        head_sha=head_sha,
        base_sha=base_sha,
        user_login=user_login,
        html_url=html_url,
        additions=additions if isinstance(additions, int) else 0,
        deletions=deletions if isinstance(deletions, int) else 0,
        changed_files=changed if isinstance(changed, int) else 0,
    )


def _repository_metadata_from_payload(
    data: dict[str, Any],
    *,
    finding_id: str | None = None,
    workflow_run_id: uuid.UUID | str | None = None,
) -> RepositoryMetadata:
    full_name = data.get("full_name")
    if not isinstance(full_name, str) or not full_name:
        msg = "repository payload missing full_name"
        raise GitHubMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    desc = data.get("description")
    description = desc if isinstance(desc, str) else None
    default_branch = data.get("default_branch")
    if not isinstance(default_branch, str):
        default_branch = "main"
    private = bool(data.get("private"))
    html_url = data.get("html_url")
    if not isinstance(html_url, str):
        html_url = ""
    stars = data.get("stargazers_count")
    forks = data.get("forks_count")
    issues = data.get("open_issues_count")
    lang = data.get("language")
    language = lang if isinstance(lang, str) else None
    pushed_at = _parse_github_datetime(data.get("pushed_at"))
    return RepositoryMetadata(
        full_name=full_name,
        description=description,
        default_branch=default_branch,
        private=private,
        html_url=html_url,
        stargazers_count=stars if isinstance(stars, int) else 0,
        forks_count=forks if isinstance(forks, int) else 0,
        open_issues_count=issues if isinstance(issues, int) else 0,
        language=language,
        pushed_at=pushed_at,
    )

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest

from tools.github import (
    AdvisoryData,
    GitHubAPIError,
    GitHubClient,
    GitHubInvalidRepoSlugError,
    GitHubIssueSearchItem,
    GitHubMalformedResponseError,
    PullRequestInfo,
    RepositoryMetadata,
    normalise_ghsa_id,
    validate_github_repo_name,
    validate_github_repo_owner,
)


def _transport(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        transport=handler,
    )


def test_normalise_ghsa_id_uppercases() -> None:
    assert normalise_ghsa_id("ghsa-abcd-efgh-ijkl") == "GHSA-ABCD-EFGH-IJKL"


def test_normalise_ghsa_id_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="invalid GHSA"):
        normalise_ghsa_id("CVE-2024-0000")


def test_validate_repo_owner_and_name() -> None:
    assert validate_github_repo_owner("acme-corp") == "acme-corp"
    assert validate_github_repo_name("my.repo-name") == "my.repo-name"


def test_validate_repo_owner_rejects_path_injection() -> None:
    with pytest.raises(GitHubInvalidRepoSlugError, match="invalid GitHub owner"):
        validate_github_repo_owner("evil/corp")


def test_validate_repo_name_rejects_empty() -> None:
    with pytest.raises(GitHubInvalidRepoSlugError):
        validate_github_repo_name("")


@pytest.mark.asyncio
async def test_fetch_repository_security_advisory_success() -> None:
    payload = {
        "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
        "summary": "Test vuln",
        "description": "Details here",
        "severity": "high",
        "html_url": "https://github.com/o/r/security/advisories/GHSA-ABCD-EFGH-IJKL",
        "identifiers": [{"type": "CVE", "value": "CVE-2024-1234"}],
        "cwes": [{"cwe_id": "CWE-79", "name": "XSS"}],
        "published_at": "2024-01-02T12:00:00Z",
        "updated_at": "2024-01-03T00:00:00Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert "/repos/acme/app/security-advisories/GHSA-ABCD-EFGH-IJKL" in str(request.url)
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        adv = await gh.fetch_repository_security_advisory("acme", "app", "ghsa-abcd-efgh-ijkl")

    assert adv == AdvisoryData(
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        source="repository",
        summary="Test vuln",
        description="Details here",
        severity="high",
        cve_ids=("CVE-2024-1234",),
        cwe_ids=("CWE-79",),
        html_url="https://github.com/o/r/security/advisories/GHSA-ABCD-EFGH-IJKL",
        published_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        updated_at=datetime(2024, 1, 3, 0, 0, tzinfo=UTC),
        cvss_vector=None,
        cvss_score_api=None,
        affected_package_name=None,
        affected_package_ecosystem=None,
    )


@pytest.mark.asyncio
async def test_fetch_global_security_advisory_success() -> None:
    payload = {
        "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
        "summary": "Global",
        "description": "Desc",
        "severity": "medium",
        "identifiers": [],
        "cwes": [],
        "published_at": None,
        "updated_at": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/advisories/GHSA-ABCD-EFGH-IJKL" in str(request.url)
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        adv = await gh.fetch_global_security_advisory("GHSA-ABCD-EFGH-IJKL")

    assert adv.source == "global"
    assert adv.summary == "Global"


@pytest.mark.asyncio
async def test_fetch_pull_request_success() -> None:
    payload = {
        "number": 42,
        "title": "Fix thing",
        "state": "open",
        "head": {"sha": "abc"},
        "base": {"sha": "def"},
        "user": {"login": "dev"},
        "html_url": "https://github.com/acme/app/pull/42",
        "additions": 3,
        "deletions": 1,
        "changed_files": 2,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/repos/acme/app/pulls/42" in str(request.url)
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        pr = await gh.fetch_pull_request("acme", "app", 42)

    assert pr == PullRequestInfo(
        number=42,
        title="Fix thing",
        state="open",
        head_sha="abc",
        base_sha="def",
        user_login="dev",
        html_url="https://github.com/acme/app/pull/42",
        additions=3,
        deletions=1,
        changed_files=2,
    )


@pytest.mark.asyncio
async def test_fetch_pull_request_files_paginates() -> None:
    page1 = [{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "sha": "s1"}]
    page1.extend(
        {
            "filename": f"f{i}.py",
            "status": "added",
            "additions": 1,
            "deletions": 0,
            "sha": f"s{i}",
        }
        for i in range(99)
    )
    page2 = [{"filename": "z.py", "status": "added", "additions": 1, "deletions": 0, "sha": "sz"}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/repos/acme/app/pulls/7/files" in str(request.url)
        page = request.url.params.get("page")
        if page == "1":
            return httpx.Response(200, json=page1)
        if page == "2":
            return httpx.Response(200, json=page2)
        return httpx.Response(500, json={"message": "unexpected page"})

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        files = await gh.fetch_pull_request_files("acme", "app", 7)

    assert len(files) == 101
    assert files[0].filename == "a.py"
    assert files[-1].filename == "z.py"


@pytest.mark.asyncio
async def test_fetch_pull_request_files_stops_at_max_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.github._MAX_PR_FILES_PAGES", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        if page in ("1", "2"):
            batch = [
                {
                    "filename": f"p{i}.py",
                    "status": "added",
                    "additions": 1,
                    "deletions": 0,
                    "sha": "x",
                }
                for i in range(100)
            ]
            return httpx.Response(200, json=batch)
        return httpx.Response(500, json={"message": "unexpected page"})

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubMalformedResponseError, match="exceeded"):
            await gh.fetch_pull_request_files("acme", "app", 7)


@pytest.mark.asyncio
async def test_search_issues_success() -> None:
    payload = {
        "total_count": 1,
        "items": [
            {
                "number": 7,
                "title": "Security: CVE-2024-1",
                "html_url": "https://github.com/acme/app/issues/7",
                "state": "open",
                "updated_at": "2024-06-01T12:00:00Z",
                "body": "text",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url.path) == "/search/issues"
        assert request.url.params.get("q") == "repo:acme/app is:open"
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        items = await gh.search_issues("repo:acme/app is:open")

    assert items == (
        GitHubIssueSearchItem(
            number=7,
            title="Security: CVE-2024-1",
            html_url="https://github.com/acme/app/issues/7",
            state="open",
            updated_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
            body="text",
        ),
    )


@pytest.mark.asyncio
async def test_fetch_repository_metadata_success() -> None:
    payload = {
        "full_name": "acme/app",
        "description": "App",
        "default_branch": "main",
        "private": True,
        "html_url": "https://github.com/acme/app",
        "stargazers_count": 10,
        "forks_count": 2,
        "open_issues_count": 3,
        "language": "Python",
        "pushed_at": "2024-05-01T08:00:00Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url.path) == "/repos/acme/app"
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        meta = await gh.fetch_repository_metadata("acme", "app")

    assert meta == RepositoryMetadata(
        full_name="acme/app",
        description="App",
        default_branch="main",
        private=True,
        html_url="https://github.com/acme/app",
        stargazers_count=10,
        forks_count=2,
        open_issues_count=3,
        language="Python",
        pushed_at=datetime(2024, 5, 1, 8, 0, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_github_api_error_includes_request_id() -> None:
    wid = uuid.uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"message": "Not Found"},
            headers={"x-github-request-id": "req-123"},
        )

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubAPIError) as exc:
            await gh.fetch_repository_metadata(
                "acme",
                "missing",
                workflow_run_id=wid,
            )

    err = exc.value
    assert err.http_status == 404
    assert err.github_request_id == "req-123"
    assert err.workflow_run_id == wid
    assert err.is_transient is False
    assert "Not Found" in str(err)


@pytest.mark.asyncio
async def test_transient_error_on_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "Unavailable"})

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubAPIError) as exc:
            await gh.fetch_global_security_advisory("GHSA-ABCD-EFGH-IJKL")

    assert exc.value.is_transient is True
    assert exc.value.http_status == 503


@pytest.mark.asyncio
async def test_403_rate_limit_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"x-ratelimit-remaining": "0"},
        )

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubAPIError) as exc:
            await gh.fetch_repository_metadata("acme", "app")

    assert exc.value.http_status == 403
    assert exc.value.is_transient is True


@pytest.mark.asyncio
async def test_non_json_success_body_raises_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubMalformedResponseError, match="non-JSON"):
            await gh.fetch_repository_metadata("acme", "app")


@pytest.mark.asyncio
async def test_list_repository_security_advisories_success() -> None:
    items = [
        {
            "ghsa_id": "GHSA-AAAA-BBBB-CCCC",
            "summary": "SQLi in API",
            "description": "Details",
            "severity": "high",
            "identifiers": [{"type": "CVE", "value": "CVE-2024-0001"}],
            "cwes": [{"cwe_id": "CWE-89"}],
            "published_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        },
        {
            "ghsa_id": "GHSA-DDDD-EEEE-FFFF",
            "summary": "XSS in form",
            "description": "More details",
            "severity": "medium",
            "identifiers": [],
            "cwes": [],
            "published_at": None,
            "updated_at": None,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/repos/acme/app/security-advisories" in str(request.url)
        assert "per_page=100" in str(request.url)
        return httpx.Response(200, json=items)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        result = await gh.list_repository_security_advisories("acme", "app", per_page=100)

    assert len(result) == 2
    assert result[0].ghsa_id == "GHSA-AAAA-BBBB-CCCC"
    assert result[0].severity == "high"
    assert result[0].source == "repository"
    assert result[1].ghsa_id == "GHSA-DDDD-EEEE-FFFF"


@pytest.mark.asyncio
async def test_list_repository_security_advisories_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        result = await gh.list_repository_security_advisories("acme", "app")

    assert result == ()


@pytest.mark.asyncio
async def test_list_repository_security_advisories_with_filters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        assert "state=published" in url
        assert "severity=critical" in url
        assert "sort=published" in url
        assert "direction=asc" in url
        return httpx.Response(200, json=[])

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        await gh.list_repository_security_advisories(
            "acme",
            "app",
            state="published",
            severity="critical",
            sort="published",
            direction="asc",
        )


@pytest.mark.asyncio
async def test_list_repository_security_advisories_paginates_with_cursor() -> None:
    page1 = [{"ghsa_id": f"GHSA-{i:04d}-AAAA-BBBB", "summary": f"Adv {i}", "description": ""} for i in range(2)]
    page2 = [{"ghsa_id": "GHSA-0002-AAAA-BBBB", "summary": "Last", "description": ""}]
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        url = str(request.url)
        if "after=" not in url:
            link = '<https://api.github.com/repos/a/b?after=cursor123>; rel="next"'
            return httpx.Response(200, json=page1, headers={"link": link})
        assert "after=cursor123" in url
        return httpx.Response(200, json=page2)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        result = await gh.list_repository_security_advisories("acme", "app", per_page=2)

    assert call_count == 2
    assert len(result) == 3


@pytest.mark.asyncio
async def test_list_repository_security_advisories_rejects_bad_per_page() -> None:
    async with _transport(httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(ValueError, match="per_page"):
            await gh.list_repository_security_advisories("acme", "app", per_page=0)
        with pytest.raises(ValueError, match="per_page"):
            await gh.list_repository_security_advisories("acme", "app", per_page=101)


@pytest.mark.asyncio
async def test_list_repository_security_advisories_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Forbidden"})

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubAPIError) as exc:
            await gh.list_repository_security_advisories("acme", "app")

    assert exc.value.http_status == 403


@pytest.mark.asyncio
async def test_client_requires_context_or_injected_http_client() -> None:
    gh = GitHubClient("token")
    with pytest.raises(RuntimeError, match="context manager"):
        await gh.fetch_repository_metadata("acme", "app")

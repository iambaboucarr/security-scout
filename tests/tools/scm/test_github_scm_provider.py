from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tools.github import GitHubClient
from tools.scm import AdvisoryData, DiffData, RepositoryMetadata
from tools.scm.github import GitHubSCMProvider, _split_repo_slug


def test_split_repo_slug_valid() -> None:
    assert _split_repo_slug("acme/app") == ("acme", "app")


def test_split_repo_slug_rejects_bare_name() -> None:
    with pytest.raises(ValueError, match="owner/name"):
        _split_repo_slug("app")


def test_split_repo_slug_rejects_empty_parts() -> None:
    with pytest.raises(ValueError, match="owner/name"):
        _split_repo_slug("/app")
    with pytest.raises(ValueError, match="owner/name"):
        _split_repo_slug("acme/")


def _mock_transport(responses: dict[str, httpx.Response]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path
        if key in responses:
            return responses[key]
        return httpx.Response(404, json={"message": "Not Found"})

    return httpx.MockTransport(handler)


async def _make_provider(transport: httpx.MockTransport) -> GitHubSCMProvider:
    client = httpx.AsyncClient(base_url="https://api.github.com", transport=transport)
    gh = GitHubClient("fake-token", client=client)
    await gh.__aenter__()
    return GitHubSCMProvider.from_client(gh)


def _advisory_payload(ghsa_id: str = "GHSA-ABCD-EFGH-IJKL") -> dict[str, object]:
    return {
        "ghsa_id": ghsa_id,
        "summary": "Test advisory",
        "description": "A test advisory description",
        "severity": "high",
        "html_url": f"https://github.com/advisories/{ghsa_id}",
        "identifiers": [{"type": "CVE", "value": "CVE-2024-0001"}],
        "cwes": [{"cwe_id": "CWE-79"}],
    }


def _pr_payload() -> dict[str, object]:
    return {
        "number": 42,
        "title": "Fix XSS",
        "state": "open",
        "head": {"sha": "abc123"},
        "base": {"sha": "def456"},
        "user": {"login": "dev"},
        "html_url": "https://github.com/acme/app/pull/42",
        "additions": 10,
        "deletions": 3,
        "changed_files": 2,
    }


def _pr_files_payload() -> list[dict[str, object]]:
    return [
        {"filename": "src/main.py", "status": "modified", "additions": 5, "deletions": 2, "sha": "aaa"},
        {"filename": "tests/test_main.py", "status": "added", "additions": 5, "deletions": 1, "sha": "bbb"},
    ]


def _repo_metadata_payload() -> dict[str, object]:
    return {
        "full_name": "acme/app",
        "description": "A demo app",
        "default_branch": "main",
        "private": False,
        "html_url": "https://github.com/acme/app",
        "stargazers_count": 100,
        "forks_count": 20,
        "open_issues_count": 5,
        "language": "Python",
    }


@pytest.mark.asyncio
async def test_fetch_advisory_repository() -> None:
    transport = _mock_transport(
        {
            "/repos/acme/app/security-advisories/GHSA-ABCD-EFGH-IJKL": httpx.Response(
                200,
                json=_advisory_payload(),
            ),
        }
    )
    provider = await _make_provider(transport)

    advisory = await provider.fetch_advisory(
        "GHSA-ABCD-EFGH-IJKL",
        repo="acme/app",
        source="repository",
    )
    assert isinstance(advisory, AdvisoryData)
    assert advisory.ghsa_id == "GHSA-ABCD-EFGH-IJKL"
    assert advisory.severity == "high"


@pytest.mark.asyncio
async def test_fetch_advisory_global() -> None:
    transport = _mock_transport(
        {
            "/advisories/GHSA-ABCD-EFGH-IJKL": httpx.Response(
                200,
                json=_advisory_payload(),
            ),
        }
    )
    provider = await _make_provider(transport)

    advisory = await provider.fetch_advisory(
        "GHSA-ABCD-EFGH-IJKL",
        source="global",
    )
    assert isinstance(advisory, AdvisoryData)
    assert advisory.source == "global"


@pytest.mark.asyncio
async def test_fetch_advisory_repository_requires_repo() -> None:
    provider = GitHubSCMProvider("fake-token")
    with pytest.raises(ValueError, match="repo is required"):
        await provider.fetch_advisory("GHSA-ABCD-EFGH-IJKL", source="repository")


@pytest.mark.asyncio
async def test_fetch_pr_diff() -> None:
    transport = _mock_transport(
        {
            "/repos/acme/app/pulls/42": httpx.Response(200, json=_pr_payload()),
            "/repos/acme/app/pulls/42/files": httpx.Response(200, json=_pr_files_payload()),
        }
    )
    provider = await _make_provider(transport)

    diff = await provider.fetch_pr_diff("acme/app", 42)
    assert isinstance(diff, DiffData)
    assert diff.pull_request.number == 42
    assert len(diff.files) == 2
    assert diff.files[0].filename == "src/main.py"


@pytest.mark.asyncio
async def test_fetch_repository_metadata() -> None:
    transport = _mock_transport(
        {
            "/repos/acme/app": httpx.Response(200, json=_repo_metadata_payload()),
        }
    )
    provider = await _make_provider(transport)

    meta = await provider.fetch_repository_metadata("acme/app")
    assert isinstance(meta, RepositoryMetadata)
    assert meta.full_name == "acme/app"
    assert meta.language == "Python"


@pytest.mark.asyncio
async def test_fetch_repository_contributors_count() -> None:
    contributors = [{"login": f"dev{i}"} for i in range(5)]
    transport = _mock_transport(
        {
            "/repos/acme/app/contributors": httpx.Response(200, json=contributors),
        }
    )
    provider = await _make_provider(transport)

    count, is_truncated = await provider.fetch_repository_contributors_count("acme/app")
    assert count == 5
    assert is_truncated is False


@pytest.mark.asyncio
async def test_search_issues() -> None:
    transport = _mock_transport(
        {
            "/search/issues": httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "number": 10,
                            "title": "Security bug",
                            "html_url": "https://github.com/acme/app/issues/10",
                            "state": "open",
                        },
                    ],
                },
            ),
        }
    )
    provider = await _make_provider(transport)

    results = await provider.search_issues("CVE-2024-0001 repo:acme/app")
    assert len(results) == 1
    assert results[0].number == 10


@pytest.mark.asyncio
async def test_not_implemented_methods_raise() -> None:
    provider = GitHubSCMProvider("fake-token")
    with pytest.raises(NotImplementedError):
        await provider.fetch_code_scanning_alerts("acme/app", "main")
    with pytest.raises(NotImplementedError):
        await provider.post_pr_comment("acme/app", 1, "comment")
    with pytest.raises(NotImplementedError):
        await provider.set_check_run("acme/app", "sha", name="test", status="completed")
    with pytest.raises(NotImplementedError):
        await provider.trigger_workflow("acme/app", ".github/workflows/ci.yml", "main")
    with pytest.raises(NotImplementedError):
        await provider.clone_repo("acme/app", "main", Path("/tmp/test"))


@pytest.mark.asyncio
async def test_context_manager() -> None:
    async with GitHubSCMProvider("fake-token") as provider:
        assert provider.client is not None

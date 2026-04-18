# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal, Self

from tools.github import (
    GitHubClient,
    GitHubIssueSearchItem,
    RepositoryMetadata,
)
from tools.scm.models import AdvisoryData
from tools.scm.protocol import DiffData


def _split_repo_slug(repo: str) -> tuple[str, str]:
    """Split ``"owner/name"`` into ``(owner, name)``; raises on bad format."""
    parts = repo.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        msg = f"repo slug must be 'owner/name', got {repo!r}"
        raise ValueError(msg)
    return parts[0], parts[1]


class GitHubSCMProvider:
    """``SCMProvider`` backed by the GitHub REST API via ``GitHubClient``."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        api_version: str = "2022-11-28",
    ) -> None:
        self._client = GitHubClient(token, base_url=base_url, api_version=api_version)

    @classmethod
    def from_client(cls, client: GitHubClient) -> GitHubSCMProvider:
        """Build a provider from an existing ``GitHubClient`` (useful in tests)."""
        instance = cls.__new__(cls)
        instance._client = client
        return instance

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.__aexit__(*args)

    @property
    def client(self) -> GitHubClient:
        """Escape hatch for GitHub-specific operations not on the protocol."""
        return self._client

    # -- Core SCMProvider methods --------------------------------------------

    async def fetch_advisory(
        self,
        advisory_id: str,
        *,
        repo: str | None = None,
        source: Literal["repository", "global"] = "repository",
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> AdvisoryData:
        if source == "repository":
            if repo is None:
                msg = "repo is required when source='repository'"
                raise ValueError(msg)
            owner, name = _split_repo_slug(repo)
            return await self._client.fetch_repository_security_advisory(
                owner,
                name,
                advisory_id,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
            )
        return await self._client.fetch_global_security_advisory(
            advisory_id,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def list_advisories(
        self,
        repo: str,
        *,
        state: str | None = None,
        severity: str | None = None,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[AdvisoryData, ...]:
        owner, name = _split_repo_slug(repo)
        return await self._client.list_repository_security_advisories(
            owner,
            name,
            state=state,
            severity=severity,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def fetch_code_scanning_alerts(
        self,
        repo: str,
        ref: str,
    ) -> list[dict[str, object]]:
        raise NotImplementedError("fetch_code_scanning_alerts is not implemented yet")

    async def fetch_pr_diff(
        self,
        repo: str,
        pr_number: int,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> DiffData:
        owner, name = _split_repo_slug(repo)
        pr_info = await self._client.fetch_pull_request(
            owner,
            name,
            pr_number,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        files = await self._client.fetch_pull_request_files(
            owner,
            name,
            pr_number,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        return DiffData(pull_request=pr_info, files=files)

    async def post_pr_comment(
        self,
        repo: str,
        pr_number: int,
        body: str,
    ) -> None:
        raise NotImplementedError("post_pr_comment is not implemented yet")

    async def set_check_run(
        self,
        repo: str,
        head_sha: str,
        *,
        name: str,
        status: str,
        conclusion: str | None = None,
    ) -> None:
        raise NotImplementedError("set_check_run is not implemented yet")

    async def trigger_workflow(
        self,
        repo: str,
        workflow_path: str,
        ref: str,
    ) -> None:
        raise NotImplementedError("trigger_workflow is not implemented yet")

    async def clone_repo(
        self,
        repo: str,
        ref: str,
        dest: Path,
    ) -> Path:
        raise NotImplementedError("clone_repo is not implemented yet")

    # -- Additional methods required by current agents -----------------------

    async def fetch_repository_metadata(
        self,
        repo: str,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> RepositoryMetadata:
        owner, name = _split_repo_slug(repo)
        return await self._client.fetch_repository_metadata(
            owner,
            name,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def fetch_repository_contributors_count(
        self,
        repo: str,
        *,
        per_page: int = 100,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[int, bool]:
        owner, name = _split_repo_slug(repo)
        return await self._client.fetch_repository_contributors_count_upper_bound(
            owner,
            name,
            per_page=per_page,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def search_issues(
        self,
        query: str,
        *,
        per_page: int = 30,
        page: int = 1,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[GitHubIssueSearchItem, ...]:
        return await self._client.search_issues(
            query,
            per_page=per_page,
            page=page,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )


__all__ = ["GitHubSCMProvider"]

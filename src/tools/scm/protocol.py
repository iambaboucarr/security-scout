# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from tools.scm.models import (
    AdvisoryData,
    IssueSearchItem,
    PullRequestFileInfo,
    PullRequestInfo,
    RepositoryMetadata,
)


class DiffData(BaseModel):
    """Combined pull-request metadata and changed-file listing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pull_request: PullRequestInfo
    files: tuple[PullRequestFileInfo, ...] = ()


@runtime_checkable
class SCMProvider(Protocol):
    async def fetch_advisory(
        self,
        advisory_id: str,
        *,
        repo: str | None = None,
        source: Literal["repository", "global"] = "repository",
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> AdvisoryData: ...

    async def list_advisories(
        self,
        repo: str,
        *,
        state: str | None = None,
        severity: str | None = None,
        max_pages: int = 20,
        per_page: int = 30,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[AdvisoryData, ...]:
        """List all security advisories for a repository."""
        ...

    def iter_list_advisories(
        self,
        repo: str,
        *,
        state: str | None = None,
        severity: str | None = None,
        per_page: int = 30,
        max_pages: int = 20,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
        poll_first_page_if_none_match: str | None = None,
        poll_on_first_page_not_modified: Callable[[], Awaitable[None]] | None = None,
        poll_on_first_page_etag: Callable[[str], Awaitable[None]] | None = None,
        poll_on_list_page_response: Callable[[object], Awaitable[None]] | None = None,
    ) -> AsyncIterator[tuple[AdvisoryData, ...]]:
        """Stream repository security advisories page by page (async generator; use ``async for``)."""
        ...

    async def fetch_code_scanning_alerts(
        self,
        repo: str,
        ref: str,
    ) -> list[dict[str, object]]:
        """Return code-scanning alerts for *repo* at *ref*."""
        ...

    async def fetch_pr_diff(
        self,
        repo: str,
        pr_number: int,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> DiffData:
        """Return PR metadata and per-file diffs."""
        ...

    async def post_pr_comment(
        self,
        repo: str,
        pr_number: int,
        body: str,
    ) -> None:
        """Post an inline or general comment on a pull request."""
        ...

    async def set_check_run(
        self,
        repo: str,
        head_sha: str,
        *,
        name: str,
        status: str,
        conclusion: str | None = None,
    ) -> None:
        """Create or update a check run on a commit."""
        ...

    async def trigger_workflow(
        self,
        repo: str,
        workflow_path: str,
        ref: str,
    ) -> None:
        """Dispatch a repository workflow."""
        ...

    async def clone_repo(
        self,
        repo: str,
        ref: str,
        dest: Path,
    ) -> Path:
        """Clone a repository to *dest* at the given *ref*."""
        ...

    # -- Additional methods required by current agents -----------------------

    async def fetch_repository_metadata(
        self,
        repo: str,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> RepositoryMetadata:
        """Fetch repository metadata (description, stars, default branch, etc.)."""
        ...

    async def fetch_repository_contributors_count(
        self,
        repo: str,
        *,
        per_page: int = 100,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[int, bool]:
        """Return ``(count, is_truncated)`` from the first page of contributors."""
        ...

    async def search_issues(
        self,
        query: str,
        *,
        per_page: int = 30,
        page: int = 1,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[IssueSearchItem, ...]:
        """Search issues/PRs."""
        ...


__all__ = [
    "DiffData",
    "SCMProvider",
]

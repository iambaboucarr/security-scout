# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Literal, Self

from exceptions import SecurityScoutError
from tools.github import (
    GitHubClient,
    GitHubIssueSearchItem,
    RepositoryMetadata,
)
from tools.scm.models import AdvisoryData
from tools.scm.protocol import DiffData

_CLONE_TIMEOUT_SECONDS = 120

_TOKEN_IN_URL_RE = re.compile(r"(https?://)[^@]+@", re.IGNORECASE)

_SHA_FULL_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SHA_SHORT_RE = re.compile(r"^[0-9a-f]{7,39}$", re.IGNORECASE)


def is_sha_ref(ref: str) -> bool:
    """Return ``True`` if *ref* looks like a Git commit SHA (7-40 hex chars)."""
    return bool(_SHA_FULL_RE.match(ref) or _SHA_SHORT_RE.match(ref))


class CloneError(SecurityScoutError):
    """Raised when ``git clone`` fails."""


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
        self._token = token
        self._client = GitHubClient(token, base_url=base_url, api_version=api_version)

    @classmethod
    def from_client(cls, client: GitHubClient, *, token: str = "") -> GitHubSCMProvider:
        """Build a provider from an existing ``GitHubClient`` (useful in tests).

        ``token`` is optional; leave empty if clone_repo is not needed.
        """
        instance = cls.__new__(cls)
        instance._token = token
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
        max_pages: int = 20,
        per_page: int = 30,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> tuple[AdvisoryData, ...]:
        owner, name = _split_repo_slug(repo)
        return await self._client.list_repository_security_advisories(
            owner,
            name,
            state=state,
            severity=severity,
            per_page=per_page,
            max_pages=max_pages,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )

    async def iter_list_advisories(
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
    ) -> AsyncIterator[tuple[AdvisoryData, ...]]:
        owner, name = _split_repo_slug(repo)
        async for page in self._client.iter_repository_security_advisories(
            owner,
            name,
            state=state,
            severity=severity,
            per_page=per_page,
            max_pages=max_pages,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            first_page_if_none_match=poll_first_page_if_none_match,
            on_first_page_not_modified=poll_on_first_page_not_modified,
            on_first_page_etag=poll_on_first_page_etag,
        ):
            yield page

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
        """Clone *repo* at *ref* into *dest*.

        For branch/tag refs: shallow clone with ``--branch``.
        For SHA refs (7-40 hex chars): full clone (``--no-checkout``) then
        ``git fetch origin <sha> && git checkout <sha>`` — required because
        ``--branch`` does not accept arbitrary commit SHAs.
        """
        owner, name = _split_repo_slug(repo)
        clone_url = f"https://x-access-token:{self._token}@github.com/{owner}/{name}.git"
        dest.mkdir(parents=True, exist_ok=True)
        repo_dir = dest / name

        if is_sha_ref(ref):
            await self._clone_at_sha(clone_url, ref, repo_dir, owner=owner, name=name)
        else:
            await self._clone_at_branch(clone_url, ref, repo_dir, owner=owner, name=name)

        return repo_dir

    async def _clone_at_branch(
        self,
        clone_url: str,
        ref: str,
        repo_dir: Path,
        *,
        owner: str,
        name: str,
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            "--single-branch",
            clone_url,
            str(repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_CLONE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            msg = f"git clone timed out for {owner}/{name}@{ref} after {_CLONE_TIMEOUT_SECONDS}s"
            raise CloneError(msg, is_transient=True) from exc

        if proc.returncode != 0:
            raw = stderr.decode("utf-8", errors="replace").strip()
            safe_msg = _TOKEN_IN_URL_RE.sub(r"\1<REDACTED>@", raw)
            msg = f"git clone failed for {owner}/{name}@{ref}: {safe_msg}"
            raise CloneError(msg)

    async def _clone_at_sha(
        self,
        clone_url: str,
        sha: str,
        repo_dir: Path,
        *,
        owner: str,
        name: str,
    ) -> None:
        """Clone without checkout, fetch the specific SHA, then checkout."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--no-checkout",
            clone_url,
            str(repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_CLONE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            msg = f"git clone timed out for {owner}/{name}@{sha} after {_CLONE_TIMEOUT_SECONDS}s"
            raise CloneError(msg, is_transient=True) from exc

        if proc.returncode != 0:
            raw = stderr.decode("utf-8", errors="replace").strip()
            safe_msg = _TOKEN_IN_URL_RE.sub(r"\1<REDACTED>@", raw)
            msg = f"git clone failed for {owner}/{name}@{sha}: {safe_msg}"
            raise CloneError(msg)

        fetch = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo_dir),
            "fetch",
            "origin",
            sha,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(fetch.communicate(), timeout=_CLONE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            fetch.kill()
            await fetch.wait()
            msg = f"git fetch timed out for {owner}/{name}@{sha}"
            raise CloneError(msg, is_transient=True) from exc

        if fetch.returncode != 0:
            raw = stderr.decode("utf-8", errors="replace").strip()
            safe_msg = _TOKEN_IN_URL_RE.sub(r"\1<REDACTED>@", raw)
            msg = f"git fetch failed for {owner}/{name}@{sha}: {safe_msg}"
            raise CloneError(msg)

        checkout = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo_dir),
            "checkout",
            sha,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(checkout.communicate(), timeout=30)
        except TimeoutError as exc:
            checkout.kill()
            await checkout.wait()
            msg = f"git checkout timed out for {owner}/{name}@{sha}"
            raise CloneError(msg, is_transient=True) from exc

        if checkout.returncode != 0:
            raw = stderr.decode("utf-8", errors="replace").strip()
            safe_msg = _TOKEN_IN_URL_RE.sub(r"\1<REDACTED>@", raw)
            msg = f"git checkout failed for {owner}/{name}@{sha}: {safe_msg}"
            raise CloneError(msg)

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


__all__ = ["CloneError", "GitHubSCMProvider", "is_sha_ref"]

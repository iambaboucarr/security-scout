# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from tools.scm import DiffData, PullRequestFileInfo, PullRequestInfo, SCMProvider
from tools.scm.github import GitHubSCMProvider
from tools.scm.models import AdvisoryData, IssueSearchItem, RepositoryMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pr_info(**overrides: object) -> PullRequestInfo:
    defaults: dict[str, object] = {
        "number": 42,
        "title": "Fix bug",
        "state": "open",
        "head_sha": "abc123",
        "base_sha": "def456",
        "html_url": "https://github.com/acme/app/pull/42",
    }
    defaults.update(overrides)
    return PullRequestInfo(**defaults)


def _make_file_info(**overrides: object) -> PullRequestFileInfo:
    defaults: dict[str, object] = {
        "filename": "src/main.py",
        "status": "modified",
    }
    defaults.update(overrides)
    return PullRequestFileInfo(**defaults)


# ---------------------------------------------------------------------------
# SCMProvider protocol — structural subtyping
# ---------------------------------------------------------------------------


def test_scm_provider_is_runtime_checkable() -> None:
    assert isinstance(GitHubSCMProvider("fake-token"), SCMProvider)


def test_scm_provider_protocol_rejects_incomplete_class() -> None:
    """A class missing required protocol methods does not satisfy isinstance."""

    class PartialProvider:
        async def fetch_advisory(self, advisory_id: str) -> None: ...

    assert not isinstance(PartialProvider(), SCMProvider)


def test_scm_provider_protocol_accepts_full_custom_implementation() -> None:
    """A class implementing every protocol method passes isinstance."""

    # Keep in sync with SCMProvider — if the protocol gains a method,
    # this class must be updated or the isinstance check will fail.
    class FakeSCMProvider:
        async def fetch_advisory(
            self,
            _advisory_id: str,
            *,
            _repo: str | None = None,
            _source: Literal["repository", "global"] = "repository",
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
        ) -> AdvisoryData:
            raise NotImplementedError

        async def list_advisories(
            self,
            _repo: str,
            *,
            _state: str | None = None,
            _severity: str | None = None,
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
        ) -> tuple[AdvisoryData, ...]:
            raise NotImplementedError

        async def iter_list_advisories(
            self,
            _repo: str,
            *,
            _state: str | None = None,
            _severity: str | None = None,
            _per_page: int = 30,
            _max_pages: int = 20,
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
            _poll_first_page_if_none_match: str | None = None,
            _poll_on_first_page_not_modified: object = None,
            _poll_on_first_page_etag: object = None,
        ) -> AsyncIterator[tuple[AdvisoryData, ...]]:
            if False:  # pragma: no cover
                yield ()  # type: ignore[unreachable]
            raise NotImplementedError

        async def fetch_code_scanning_alerts(
            self,
            _repo: str,
            _ref: str,
        ) -> list[dict[str, object]]:
            raise NotImplementedError

        async def fetch_pr_diff(
            self,
            _repo: str,
            _pr_number: int,
            *,
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
        ) -> DiffData:
            raise NotImplementedError

        async def post_pr_comment(
            self,
            _repo: str,
            _pr_number: int,
            _body: str,
        ) -> None:
            raise NotImplementedError

        async def set_check_run(
            self,
            _repo: str,
            _head_sha: str,
            *,
            _name: str,
            _status: str,
            _conclusion: str | None = None,
        ) -> None:
            raise NotImplementedError

        async def trigger_workflow(
            self,
            _repo: str,
            _workflow_path: str,
            _ref: str,
        ) -> None:
            raise NotImplementedError

        async def clone_repo(
            self,
            _repo: str,
            _ref: str,
            _dest: Path,
        ) -> Path:
            raise NotImplementedError

        async def fetch_repository_metadata(
            self,
            _repo: str,
            *,
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
        ) -> RepositoryMetadata:
            raise NotImplementedError

        async def fetch_repository_contributors_count(
            self,
            _repo: str,
            *,
            _per_page: int = 100,
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
        ) -> tuple[int, bool]:
            raise NotImplementedError

        async def search_issues(
            self,
            _query: str,
            *,
            _per_page: int = 30,
            _page: int = 1,
            _finding_id: str | None = None,
            _workflow_run_id: uuid.UUID | str | None = None,
        ) -> tuple[IssueSearchItem, ...]:
            raise NotImplementedError

    assert isinstance(FakeSCMProvider(), SCMProvider)


def test_scm_provider_protocol_surface_has_expected_methods() -> None:
    expected = {
        "fetch_advisory",
        "list_advisories",
        "iter_list_advisories",
        "fetch_code_scanning_alerts",
        "fetch_pr_diff",
        "post_pr_comment",
        "set_check_run",
        "trigger_workflow",
        "clone_repo",
        "fetch_repository_metadata",
        "fetch_repository_contributors_count",
        "search_issues",
    }
    protocol_methods = {
        name for name in dir(SCMProvider) if not name.startswith("_") and callable(getattr(SCMProvider, name, None))
    }
    assert expected <= protocol_methods


# ---------------------------------------------------------------------------
# DiffData model
# ---------------------------------------------------------------------------


def test_diff_data_combines_pr_info_and_files() -> None:
    pr = _make_pr_info()
    f = _make_file_info(patch="@@ -1 +1 @@")
    diff = DiffData(pull_request=pr, files=(f,))
    assert diff.pull_request.number == 42
    assert len(diff.files) == 1
    assert diff.files[0].filename == "src/main.py"


def test_diff_data_default_empty_files() -> None:
    pr = _make_pr_info(number=1, title="t", head_sha="a", base_sha="b", html_url="https://example.com/pull/1")
    diff = DiffData(pull_request=pr)
    assert diff.files == ()


def test_diff_data_is_frozen() -> None:
    pr = _make_pr_info()
    diff = DiffData(pull_request=pr, files=())
    with pytest.raises(ValidationError):
        diff.pull_request = _make_pr_info(number=99)


def test_diff_data_rejects_extra_fields() -> None:
    pr = _make_pr_info()
    with pytest.raises(ValidationError, match="extra"):
        DiffData(pull_request=pr, files=(), bonus="bad")


def test_diff_data_with_multiple_files() -> None:
    pr = _make_pr_info()
    files = tuple(_make_file_info(filename=f"file_{i}.py", status="added") for i in range(5))
    diff = DiffData(pull_request=pr, files=files)
    assert len(diff.files) == 5
    assert diff.files[3].filename == "file_3.py"


def test_diff_data_files_preserved_as_tuple() -> None:
    pr = _make_pr_info()
    files_list = [_make_file_info(filename="a.py"), _make_file_info(filename="b.py")]
    diff = DiffData(pull_request=pr, files=tuple(files_list))
    assert isinstance(diff.files, tuple)
    assert len(diff.files) == 2


# ---------------------------------------------------------------------------
# Module __all__ exports
# ---------------------------------------------------------------------------


def test_protocol_module_exports() -> None:
    from tools.scm import protocol

    assert "DiffData" in protocol.__all__
    assert "SCMProvider" in protocol.__all__

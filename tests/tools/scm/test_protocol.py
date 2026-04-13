from __future__ import annotations

from tools.scm import DiffData, PullRequestFileInfo, PullRequestInfo, SCMProvider
from tools.scm.github import GitHubSCMProvider


def test_scm_provider_is_runtime_checkable() -> None:
    assert isinstance(GitHubSCMProvider("fake-token"), SCMProvider)


def test_diff_data_combines_pr_info_and_files() -> None:
    pr = PullRequestInfo(
        number=42,
        title="Fix bug",
        state="open",
        head_sha="abc123",
        base_sha="def456",
        html_url="https://github.com/acme/app/pull/42",
    )
    f = PullRequestFileInfo(filename="src/main.py", status="modified", patch="@@ -1 +1 @@")
    diff = DiffData(pull_request=pr, files=(f,))
    assert diff.pull_request.number == 42
    assert len(diff.files) == 1
    assert diff.files[0].filename == "src/main.py"


def test_diff_data_default_empty_files() -> None:
    pr = PullRequestInfo(
        number=1,
        title="t",
        state="open",
        head_sha="a",
        base_sha="b",
        html_url="https://example.com/pull/1",
    )
    diff = DiffData(pull_request=pr)
    assert diff.files == ()

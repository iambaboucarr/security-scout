# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from tools.github import (
    GitHubAPIError,
    GitHubClient,
    GitHubMalformedResponseError,
    _as_json_array,
    _as_json_object,
    _coerce_positive_int,
    _cve_ids_from_identifiers,
    _cvss_vector_and_score_from_payload,
    _cwe_ids_from_cwes,
    _first_affected_package_from_payload,
    _issue_search_item_from_payload,
    _looks_like_github_rate_limit,
    _message_from_error_body,
    _parse_github_datetime,
    _pull_request_from_payload,
    _repository_metadata_from_payload,
    _require_pull_number,
)


def _transport(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        transport=handler,
    )


# ── _parse_github_datetime ────────────────────────────────────────────────


def test_parse_datetime_none() -> None:
    assert _parse_github_datetime(None) is None


def test_parse_datetime_already_datetime() -> None:
    dt = datetime(2024, 1, 1, tzinfo=UTC)
    assert _parse_github_datetime(dt) == dt


def test_parse_datetime_valid_string() -> None:
    result = _parse_github_datetime("2024-06-15T10:30:00Z")
    assert result is not None
    assert result.year == 2024
    assert result.month == 6


def test_parse_datetime_invalid_string() -> None:
    assert _parse_github_datetime("not-a-date") is None


def test_parse_datetime_non_string_non_none() -> None:
    assert _parse_github_datetime(12345) is None


# ── _message_from_error_body ──────────────────────────────────────────────


def test_error_body_with_message() -> None:
    response = httpx.Response(400, json={"message": "Bad Request"})
    assert _message_from_error_body(response) == "Bad Request"


def test_error_body_with_errors_array() -> None:
    response = httpx.Response(
        422,
        json={"errors": [{"message": "field error", "code": "invalid"}]},
    )
    assert _message_from_error_body(response) == "field error"


def test_error_body_non_json() -> None:
    response = httpx.Response(500, content=b"Internal Server Error")
    assert _message_from_error_body(response) == "Internal Server Error"


def test_error_body_non_dict_json() -> None:
    response = httpx.Response(400, json=["list", "not", "dict"])
    assert "list" in _message_from_error_body(response)


def test_error_body_empty_message() -> None:
    response = httpx.Response(400, json={"message": ""})
    result = _message_from_error_body(response)
    assert isinstance(result, str)


def test_error_body_errors_first_not_dict() -> None:
    response = httpx.Response(400, json={"errors": ["just a string"]})
    result = _message_from_error_body(response)
    assert isinstance(result, str)


def test_error_body_errors_first_dict_no_message() -> None:
    response = httpx.Response(400, json={"errors": [{"code": "invalid"}]})
    result = _message_from_error_body(response)
    assert isinstance(result, str)


# ── _looks_like_github_rate_limit ─────────────────────────────────────────


def test_rate_limit_non_403() -> None:
    response = httpx.Response(500, json={"message": "error"})
    assert _looks_like_github_rate_limit(response) is False


def test_rate_limit_by_remaining_header() -> None:
    response = httpx.Response(
        403,
        json={"message": "forbidden"},
        headers={"x-ratelimit-remaining": "0"},
    )
    assert _looks_like_github_rate_limit(response) is True


def test_rate_limit_by_retry_after_header() -> None:
    response = httpx.Response(
        403,
        json={"message": "forbidden"},
        headers={"retry-after": "60"},
    )
    assert _looks_like_github_rate_limit(response) is True


def test_rate_limit_by_body_text() -> None:
    response = httpx.Response(
        403,
        json={"message": "You have exceeded the secondary rate limit"},
    )
    assert _looks_like_github_rate_limit(response) is True


def test_rate_limit_normal_403() -> None:
    response = httpx.Response(
        403,
        json={"message": "Resource not accessible by integration"},
    )
    assert _looks_like_github_rate_limit(response) is False


# ── _coerce_positive_int ──────────────────────────────────────────────────


def test_coerce_positive_int_bool() -> None:
    assert _coerce_positive_int(True) is None
    assert _coerce_positive_int(False) is None


def test_coerce_positive_int_zero() -> None:
    assert _coerce_positive_int(0) is None


def test_coerce_positive_int_negative() -> None:
    assert _coerce_positive_int(-1) is None


def test_coerce_positive_int_valid() -> None:
    assert _coerce_positive_int(42) == 42


def test_coerce_positive_int_float_whole() -> None:
    assert _coerce_positive_int(7.0) == 7


def test_coerce_positive_int_float_fraction() -> None:
    assert _coerce_positive_int(7.5) is None


def test_coerce_positive_int_float_zero() -> None:
    assert _coerce_positive_int(0.0) is None


def test_coerce_positive_int_string() -> None:
    assert _coerce_positive_int("5") is None


# ── _require_pull_number ──────────────────────────────────────────────────


def test_require_pull_number_valid() -> None:
    assert _require_pull_number(1) == 1


def test_require_pull_number_zero_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        _require_pull_number(0)


# ── _cve_ids_from_identifiers ─────────────────────────────────────────────


def test_cve_ids_not_list() -> None:
    assert _cve_ids_from_identifiers("string") == ()


def test_cve_ids_non_dict_items() -> None:
    assert _cve_ids_from_identifiers([1, "two"]) == ()


def test_cve_ids_mixed() -> None:
    result = _cve_ids_from_identifiers(
        [
            {"type": "CVE", "value": "CVE-2024-1"},
            {"type": "GHSA", "value": "GHSA-1234-5678-ABCD"},
            {"type": "CVE", "value": "CVE-2024-2"},
        ]
    )
    assert result == ("CVE-2024-1", "CVE-2024-2")


# ── _cwe_ids_from_cwes ───────────────────────────────────────────────────


def test_cwe_ids_not_list() -> None:
    assert _cwe_ids_from_cwes("string") == ()


def test_cwe_ids_non_dict_items() -> None:
    assert _cwe_ids_from_cwes([42]) == ()


def test_cwe_ids_mixed() -> None:
    result = _cwe_ids_from_cwes(
        [
            {"cwe_id": "CWE-79"},
            {"cwe_id": ""},
            {"no_id": True},
            {"cwe_id": "CWE-89"},
        ]
    )
    assert result == ("CWE-79", "CWE-89")


# ── _cvss_vector_and_score_from_payload ───────────────────────────────────


def test_cvss_from_cvss_key() -> None:
    vec, score = _cvss_vector_and_score_from_payload(
        {
            "cvss": {"vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", "score": 9.8},
        }
    )
    assert vec is not None
    assert score == 9.8


def test_cvss_from_cvss_v3_key() -> None:
    vec, score = _cvss_vector_and_score_from_payload(
        {
            "cvss_v3": {"vector_string": "CVSS:3.0/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N", "score": 2.0},
        }
    )
    assert vec is not None
    assert score == 2.0


def test_cvss_no_block() -> None:
    vec, score = _cvss_vector_and_score_from_payload({})
    assert vec is None
    assert score is None


def test_cvss_score_only() -> None:
    vec, score = _cvss_vector_and_score_from_payload(
        {
            "cvss": {"score": 5.0},
        }
    )
    assert vec is None
    assert score == 5.0


def test_cvss_block_not_dict() -> None:
    vec, score = _cvss_vector_and_score_from_payload({"cvss": "string"})
    assert vec is None
    assert score is None


def test_cvss_empty_vector_string() -> None:
    vec, score = _cvss_vector_and_score_from_payload(
        {
            "cvss": {"vector_string": "  ", "score": 7.0},
        }
    )
    assert vec is None
    assert score == 7.0


# ── _first_affected_package_from_payload ──────────────────────────────────


def test_affected_package_valid() -> None:
    name, eco = _first_affected_package_from_payload(
        {
            "vulnerabilities": [
                {"package": {"name": "lodash", "ecosystem": "npm"}},
            ],
        }
    )
    assert name == "lodash"
    assert eco == "npm"


def test_affected_package_no_vulnerabilities() -> None:
    assert _first_affected_package_from_payload({}) == (None, None)


def test_affected_package_not_list() -> None:
    assert _first_affected_package_from_payload({"vulnerabilities": "string"}) == (None, None)


def test_affected_package_non_dict_item() -> None:
    assert _first_affected_package_from_payload({"vulnerabilities": [42]}) == (None, None)


def test_affected_package_no_package_key() -> None:
    assert _first_affected_package_from_payload({"vulnerabilities": [{}]}) == (None, None)


def test_affected_package_package_not_dict() -> None:
    assert _first_affected_package_from_payload({"vulnerabilities": [{"package": "string"}]}) == (None, None)


def test_affected_package_empty_name() -> None:
    name, eco = _first_affected_package_from_payload(
        {
            "vulnerabilities": [{"package": {"name": "  ", "ecosystem": "npm"}}],
        }
    )
    assert name is None
    assert eco == "npm"


# ── _as_json_object / _as_json_array ─────────────────────────────────────


def test_as_json_object_non_json() -> None:
    response = httpx.Response(200, content=b"not json")
    with pytest.raises(GitHubMalformedResponseError, match="non-JSON"):
        _as_json_object(response)


def test_as_json_object_not_dict() -> None:
    response = httpx.Response(200, json=[1])
    with pytest.raises(GitHubMalformedResponseError, match="expected JSON object"):
        _as_json_object(response)


def test_as_json_array_non_json() -> None:
    response = httpx.Response(200, content=b"not json")
    with pytest.raises(GitHubMalformedResponseError, match="non-JSON"):
        _as_json_array(response)


def test_as_json_array_not_list() -> None:
    response = httpx.Response(200, json={"key": "value"})
    with pytest.raises(GitHubMalformedResponseError, match="expected JSON array"):
        _as_json_array(response)


# ── _pull_request_from_payload edge cases ─────────────────────────────────


def test_pr_payload_missing_number() -> None:
    with pytest.raises(GitHubMalformedResponseError, match="missing number"):
        _pull_request_from_payload({})


def test_pr_payload_minimal() -> None:
    pr = _pull_request_from_payload({"number": 1})
    assert pr.number == 1
    assert pr.title == ""
    assert pr.state == ""
    assert pr.head_sha == ""
    assert pr.base_sha == ""
    assert pr.user_login is None
    assert pr.html_url == ""
    assert pr.additions == 0


def test_pr_payload_full() -> None:
    pr = _pull_request_from_payload(
        {
            "number": 42,
            "title": "PR",
            "state": "open",
            "head": {"sha": "abc"},
            "base": {"sha": "def"},
            "user": {"login": "dev"},
            "html_url": "https://github.com/o/r/pull/42",
            "additions": 10,
            "deletions": 5,
            "changed_files": 3,
        }
    )
    assert pr.number == 42
    assert pr.additions == 10
    assert pr.deletions == 5
    assert pr.changed_files == 3
    assert pr.user_login == "dev"


def test_pr_payload_non_int_additions() -> None:
    pr = _pull_request_from_payload(
        {
            "number": 1,
            "additions": "not_int",
            "deletions": None,
            "changed_files": 2.5,
        }
    )
    assert pr.additions == 0
    assert pr.deletions == 0
    assert pr.changed_files == 0


# ── _repository_metadata_from_payload edge cases ─────────────────────────


def test_repo_metadata_missing_full_name() -> None:
    with pytest.raises(GitHubMalformedResponseError, match="missing full_name"):
        _repository_metadata_from_payload({})


def test_repo_metadata_minimal() -> None:
    meta = _repository_metadata_from_payload({"full_name": "org/repo"})
    assert meta.full_name == "org/repo"
    assert meta.default_branch == "main"
    assert meta.private is False
    assert meta.html_url == ""
    assert meta.description is None
    assert meta.language is None


def test_repo_metadata_description_non_string() -> None:
    meta = _repository_metadata_from_payload({"full_name": "o/r", "description": 42})
    assert meta.description is None


def test_repo_metadata_language_non_string() -> None:
    meta = _repository_metadata_from_payload({"full_name": "o/r", "language": 42})
    assert meta.language is None


def test_repo_metadata_numeric_fields_non_int() -> None:
    meta = _repository_metadata_from_payload(
        {
            "full_name": "o/r",
            "stargazers_count": "many",
            "forks_count": None,
            "open_issues_count": 3.14,
        }
    )
    assert meta.stargazers_count == 0
    assert meta.forks_count == 0
    assert meta.open_issues_count == 0


# ── _issue_search_item_from_payload ───────────────────────────────────────


def test_issue_item_missing_number() -> None:
    with pytest.raises(GitHubMalformedResponseError, match="missing number"):
        _issue_search_item_from_payload({})


def test_issue_item_minimal() -> None:
    item = _issue_search_item_from_payload({"number": 7})
    assert item.number == 7
    assert item.title == ""
    assert item.html_url == ""
    assert item.state == ""
    assert item.body is None


def test_issue_item_body_non_string() -> None:
    item = _issue_search_item_from_payload({"number": 7, "body": 42})
    assert item.body is None


# ── GitHubAPIError.from_httpx_response ────────────────────────────────────


def test_from_httpx_response_rate_limit() -> None:
    response = httpx.Response(
        403,
        json={"message": "rate limit exceeded"},
        headers={"x-ratelimit-remaining": "0", "x-github-request-id": "req-9"},
    )
    err = GitHubAPIError.from_httpx_response(response, finding_id="f1")
    assert err.is_transient is True
    assert err.http_status == 403
    assert err.github_request_id == "req-9"
    assert err.finding_id == "f1"


def test_from_httpx_response_non_json_body() -> None:
    response = httpx.Response(502, content=b"Bad Gateway")
    err = GitHubAPIError.from_httpx_response(response)
    assert err.is_transient is True
    assert "Bad Gateway" in str(err)


# ── fetch_repository_contributors_count_upper_bound ───────────────────────


@pytest.mark.asyncio
async def test_contributors_single_page() -> None:
    items = [{"login": f"user{i}"} for i in range(5)]

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/contributors" in str(request.url)
        return httpx.Response(200, json=items)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        count, truncated = await gh.fetch_repository_contributors_count_upper_bound("acme", "app")

    assert count == 5
    assert truncated is False


@pytest.mark.asyncio
async def test_contributors_full_page_is_truncated() -> None:
    items = [{"login": f"user{i}"} for i in range(100)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=items)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        count, truncated = await gh.fetch_repository_contributors_count_upper_bound("acme", "app")

    assert count == 100
    assert truncated is True


@pytest.mark.asyncio
async def test_contributors_invalid_per_page() -> None:
    async with _transport(httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(ValueError, match="per_page"):
            await gh.fetch_repository_contributors_count_upper_bound("acme", "app", per_page=0)


# ── search_issues validation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_issues_invalid_per_page() -> None:
    async with _transport(httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(ValueError, match="per_page"):
            await gh.search_issues("query", per_page=0)


@pytest.mark.asyncio
async def test_search_issues_invalid_page() -> None:
    async with _transport(httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(ValueError, match="page"):
            await gh.search_issues("query", page=0)


@pytest.mark.asyncio
async def test_search_issues_missing_items_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"total_count": 0})

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubMalformedResponseError, match="missing items"):
            await gh.search_issues("query")


@pytest.mark.asyncio
async def test_search_issues_item_validation_error() -> None:
    payload = {
        "items": [{"number": True}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubMalformedResponseError, match="missing number"):
            await gh.search_issues("query")


@pytest.mark.asyncio
async def test_search_issues_non_dict_item_skipped() -> None:
    payload = {
        "items": [42, {"number": 3, "title": "T", "html_url": "u", "state": "open"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        items = await gh.search_issues("query")

    assert len(items) == 1
    assert items[0].number == 3


# ── PR files edge cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pr_files_non_dict_item_skipped() -> None:
    batch = [42, {"filename": "a.py", "status": "added", "additions": 1, "deletions": 0, "sha": "s"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=batch)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        files = await gh.fetch_pull_request_files("acme", "app", 1)

    assert len(files) == 1
    assert files[0].filename == "a.py"


@pytest.mark.asyncio
async def test_pr_files_validation_error_raises() -> None:
    batch = [{"filename": 42, "status": "added"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=batch)

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        with pytest.raises(GitHubMalformedResponseError, match="validation"):
            await gh.fetch_pull_request_files("acme", "app", 1)


# ── Context manager lifecycle ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_manager_creates_and_closes_client() -> None:
    gh = GitHubClient("token", base_url="https://api.github.com")
    async with gh:
        assert gh._client is not None
    assert gh._client is None


@pytest.mark.asyncio
async def test_injected_client_not_closed_on_exit() -> None:
    http = httpx.AsyncClient(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    gh = GitHubClient("token", client=http)
    async with gh:
        pass
    assert gh._client is http
    await http.aclose()


# ── Owner request sends auth headers for injected client ──────────────────


@pytest.mark.asyncio
async def test_injected_client_sends_auth_headers() -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"full_name": "o/r"})

    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("test-token-123", client=client)
        await gh.fetch_repository_metadata("acme", "app")

    assert "authorization" in captured_headers
    assert "test-token-123" in captured_headers["authorization"]

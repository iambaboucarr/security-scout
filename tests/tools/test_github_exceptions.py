# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid

from tools.github import GitHubAPIError
from tools.slack import SlackAPIError


def test_github_api_error_explicit_fields() -> None:
    wid = uuid.uuid4()
    err = GitHubAPIError(
        "rate limited",
        is_transient=True,
        http_status=429,
        github_request_id="abc123",
        finding_id="ghsa-1",
        workflow_run_id=wid,
    )
    assert err.is_transient is True
    assert err.http_status == 429
    assert err.github_request_id == "abc123"
    assert err.finding_id == "ghsa-1"
    assert err.workflow_run_id == wid
    assert err.is_resource_error is False
    assert str(err) == "rate limited"


def test_github_from_status_transient() -> None:
    for status in (429, 502, 503, 504):
        err = GitHubAPIError.from_status(status, "upstream")
        assert err.is_transient is True
        assert err.http_status == status


def test_github_from_status_permanent() -> None:
    for status in (400, 401, 403, 404, 422):
        err = GitHubAPIError.from_status(status, "bad request")
        assert err.is_transient is False
        assert err.http_status == status


def test_slack_api_error_explicit_fields() -> None:
    err = SlackAPIError(
        "invalid_auth",
        is_transient=False,
        http_status=401,
        slack_error_code="invalid_auth",
    )
    assert err.is_transient is False
    assert err.http_status == 401
    assert err.slack_error_code == "invalid_auth"


def test_slack_from_status() -> None:
    transient = SlackAPIError.from_status(503, "unavailable")
    assert transient.is_transient is True
    permanent = SlackAPIError.from_status(404, "not found")
    assert permanent.is_transient is False

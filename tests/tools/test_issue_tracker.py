from __future__ import annotations

import json
import uuid

import httpx
import pytest

from config import (
    GitHubIssuesTrackerConfig,
    JiraTrackerConfig,
    LinearTrackerConfig,
    configure_logging,
)
from models import Finding, Severity, WorkflowKind
from tools.github import GitHubClient
from tools.issue_tracker import (
    GitHubIssuesAdapter,
    IssueTrackerCredentials,
    JiraIssuesAdapter,
    LinearIssuesAdapter,
    ScoutHistoricalAdapter,
    TrackerMatch,
    _jira_escape_text,
    dedupe_tracker_matches,
    normalise_cve_id,
    run_dedup_checks,
)
from tools.scm.github import GitHubSCMProvider


def _transport(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        transport=handler,
    )


def test_normalise_cve_id() -> None:
    assert normalise_cve_id("cve-2024-1234") == "CVE-2024-1234"


def test_normalise_cve_id_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="invalid CVE"):
        normalise_cve_id("not-a-cve")


def test_jira_escape_text_escapes_reserved_chars() -> None:
    assert _jira_escape_text("CVE-2024-9999") == "CVE\\-2024\\-9999"


def test_jira_escape_text_escapes_quotes_without_double_escaping() -> None:
    result = _jira_escape_text('value with "quotes"')
    assert result == 'value with \\"quotes\\"'
    assert '\\\\"' not in result


@pytest.mark.asyncio
async def test_github_issues_adapter_tier1_cve() -> None:
    search_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search/issues"
        search_calls.append(str(request.url.params.get("q")))
        return httpx.Response(
            200,
            json={
                "total_count": 1,
                "items": [
                    {
                        "number": 42,
                        "title": "Track CVE-2024-9999",
                        "html_url": "https://github.com/acme/app/issues/42",
                        "state": "open",
                        "updated_at": "2024-06-01T12:00:00Z",
                        "body": "See CVE-2024-9999",
                    }
                ],
            },
        )

    cfg = GitHubIssuesTrackerConfig()
    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        scm = GitHubSCMProvider.from_client(gh)
        adapter = GitHubIssuesAdapter(scm, "acme", "app", cfg)
        matches = await adapter.search_known_vulnerability(
            "CVE-2024-9999",
            None,
            None,
            None,
            None,
            None,
        )

    assert len(matches) == 1
    m = matches[0]
    assert m.tracker == "github_issues"
    assert m.match_tier == 1
    assert m.match_field == "cve_id"
    assert m.matched_value == "CVE-2024-9999"
    assert m.issue_id == "acme/app#42"
    assert m.status == "open"
    assert "repo:acme/app" in search_calls[0]
    assert "label:security" in search_calls[0]


@pytest.mark.asyncio
async def test_github_issues_adapter_skips_issue_without_identifier_in_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_count": 1,
                "items": [
                    {
                        "number": 1,
                        "title": "Unrelated",
                        "html_url": "https://github.com/acme/app/issues/1",
                        "state": "open",
                        "updated_at": "2024-06-01T12:00:00Z",
                        "body": "no cve here",
                    }
                ],
            },
        )

    cfg = GitHubIssuesTrackerConfig()
    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        scm = GitHubSCMProvider.from_client(gh)
        adapter = GitHubIssuesAdapter(scm, "acme", "app", cfg)
        matches = await adapter.search_known_vulnerability(
            "CVE-2024-9999",
            None,
            None,
            None,
            None,
            None,
        )

    assert matches == []


@pytest.mark.asyncio
async def test_scout_historical_tier1_cve(db_session) -> None:
    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="GHSA-abcd-efgh-ijkl",
        severity=Severity.high,
        title="Prior",
        cve_id="CVE-2024-1000",
    )
    db_session.add(finding)
    await db_session.commit()

    adapter = ScoutHistoricalAdapter(db_session)
    matches = await adapter.search_known_vulnerability(
        "CVE-2024-1000",
        None,
        None,
        None,
        None,
        None,
    )

    assert len(matches) == 1
    assert matches[0].tracker == "scout_history"
    assert matches[0].issue_id == str(fid)
    assert matches[0].match_tier == 1
    assert matches[0].matched_value == "CVE-2024-1000"


@pytest.mark.asyncio
async def test_scout_historical_excludes_finding(db_session) -> None:
    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="GHSA-abcd-efgh-ijkl",
        severity=Severity.high,
        title="Prior",
        cve_id="CVE-2024-1000",
    )
    db_session.add(finding)
    await db_session.commit()

    adapter = ScoutHistoricalAdapter(db_session, exclude_finding_id=fid)
    matches = await adapter.search_known_vulnerability(
        "CVE-2024-1000",
        None,
        None,
        None,
        None,
        None,
    )

    assert matches == []


@pytest.mark.asyncio
async def test_scout_historical_tier1_ghsa_source_ref(db_session) -> None:
    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-ABCD-EFGH-IJKL",
        severity=Severity.medium,
        title="Old GHSA",
    )
    db_session.add(finding)
    await db_session.commit()

    adapter = ScoutHistoricalAdapter(db_session)
    matches = await adapter.search_known_vulnerability(
        None,
        "GHSA-ABCD-EFGH-IJKL",
        None,
        None,
        None,
        None,
    )

    assert len(matches) == 1
    assert matches[0].match_field == "ghsa_id"
    assert matches[0].matched_value == "GHSA-ABCD-EFGH-IJKL"


@pytest.mark.asyncio
async def test_scout_historical_tier2_cwe_when_no_tier1(db_session) -> None:
    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="https://example.com/x",
        severity=Severity.high,
        title="CWE overlap",
        cwe_ids=["CWE-89", "CWE-79"],
    )
    db_session.add(finding)
    await db_session.commit()

    adapter = ScoutHistoricalAdapter(db_session)
    matches = await adapter.search_known_vulnerability(
        None,
        None,
        "CWE-89",
        None,
        None,
        None,
    )

    assert len(matches) == 1
    assert matches[0].match_tier == 2
    assert matches[0].match_field == "cwe_id"


@pytest.mark.asyncio
async def test_run_dedup_checks_dedupes_and_logs_metrics(
    db_session,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")

    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="GHSA-abcd-efgh-ijkl",
        severity=Severity.high,
        title="Prior",
        cve_id="CVE-2024-1000",
    )
    db_session.add(finding)
    await db_session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_count": 0,
                "items": [],
            },
        )

    cfg = GitHubIssuesTrackerConfig()
    async with _transport(httpx.MockTransport(handler)) as client:
        gh = GitHubClient("token", client=client)
        scm = GitHubSCMProvider.from_client(gh)
        gh_adapter = GitHubIssuesAdapter(scm, "acme", "app", cfg)
        scout = ScoutHistoricalAdapter(db_session)
        out = await run_dedup_checks(
            cve_id="CVE-2024-1000",
            ghsa_id=None,
            cwe_id=None,
            affected_package=None,
            affected_versions=None,
            summary=None,
            adapters=[gh_adapter, scout],
        )

    assert len(out) == 1
    assert out[0].tracker == "scout_history"
    logged = capsys.readouterr().out
    assert "dedup_latency_seconds" in logged
    assert "dedup_match_total" in logged


@pytest.mark.asyncio
async def test_jira_adapter_searches_by_cve_with_basic_auth() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        assert request.url.path == "/rest/api/3/search"
        return httpx.Response(
            200,
            json={
                "issues": [
                    {
                        "key": "SEC-42",
                        "fields": {
                            "summary": "CVE-2024-9999 — patch libfoo",
                            "status": {
                                "name": "In Progress",
                                "statusCategory": {"key": "indeterminate"},
                            },
                            "updated": "2026-04-01T12:00:00.000+0000",
                        },
                    }
                ]
            },
        )

    cfg = JiraTrackerConfig(project_key="SEC", base_url="https://acme.atlassian.net")
    creds = IssueTrackerCredentials(jira_email="ops@acme.io", jira_api_token="token-abc")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = JiraIssuesAdapter(client, cfg, creds)
        matches = await adapter.search_known_vulnerability(
            "CVE-2024-9999",
            None,
            None,
            None,
            None,
            None,
        )

    assert len(matches) == 1
    m = matches[0]
    assert m.tracker == "jira"
    assert m.issue_id == "SEC-42"
    assert m.issue_url == "https://acme.atlassian.net/browse/SEC-42"
    assert m.match_field == "cve_id"
    assert m.matched_value == "CVE-2024-9999"
    assert m.status == "in_progress"
    assert m.last_updated is not None

    assert captured
    auth = captured[0].headers.get("Authorization")
    assert auth is not None
    assert auth.startswith("Basic ")
    jql = captured[0].url.params.get("jql")
    assert jql is not None
    assert 'project = "SEC"' in jql
    assert "CVE\\-2024\\-9999" in jql  # JQL reserved chars escaped


@pytest.mark.asyncio
async def test_jira_adapter_uses_bearer_when_no_email() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"issues": []})

    cfg = JiraTrackerConfig(project_key="SEC", base_url="https://jira.acme.local")
    creds = IssueTrackerCredentials(jira_email=None, jira_api_token="pat-xyz")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = JiraIssuesAdapter(client, cfg, creds)
        await adapter.search_known_vulnerability("CVE-2024-9999", None, None, None, None, None)

    assert captured[0].headers.get("Authorization") == "Bearer pat-xyz"


@pytest.mark.asyncio
async def test_jira_adapter_swallows_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    cfg = JiraTrackerConfig(project_key="SEC", base_url="https://acme.atlassian.net")
    creds = IssueTrackerCredentials(jira_email="ops@acme.io", jira_api_token="t")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = JiraIssuesAdapter(client, cfg, creds)
        matches = await adapter.search_known_vulnerability("CVE-2024-9999", None, None, None, None, None)
    assert matches == []


def test_jira_adapter_rejects_missing_token() -> None:
    cfg = JiraTrackerConfig(project_key="SEC", base_url="https://acme.atlassian.net")
    with pytest.raises(ValueError, match="JIRA_API_TOKEN"):
        JiraIssuesAdapter(httpx.AsyncClient(), cfg, IssueTrackerCredentials())


@pytest.mark.asyncio
async def test_linear_adapter_searches_by_ghsa_with_label_filter() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured.append(body)
        assert request.url == httpx.URL("https://api.linear.app/graphql")
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": {
                        "nodes": [
                            {
                                "id": "uuid-1",
                                "identifier": "SEC-7",
                                "title": "Triage GHSA-ABCD-EFGH-IJKL",
                                "url": "https://linear.app/acme/issue/SEC-7",
                                "updatedAt": "2026-03-15T08:00:00.000Z",
                                "state": {"name": "In Progress", "type": "started"},
                            }
                        ]
                    }
                }
            },
        )

    cfg = LinearTrackerConfig(team_id="TEAM-1", label_name="security")
    creds = IssueTrackerCredentials(linear_api_key="lin_api_xyz")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = LinearIssuesAdapter(client, cfg, creds)
        matches = await adapter.search_known_vulnerability(
            None,
            "GHSA-ABCD-EFGH-IJKL",
            None,
            None,
            None,
            None,
        )

    assert len(matches) == 1
    m = matches[0]
    assert m.tracker == "linear"
    assert m.issue_id == "SEC-7"
    assert m.issue_url == "https://linear.app/acme/issue/SEC-7"
    assert m.match_field == "ghsa_id"
    assert m.matched_value == "GHSA-ABCD-EFGH-IJKL"
    assert m.status == "in_progress"

    assert captured
    body = captured[0]
    assert "filter" in body["variables"]  # type: ignore[index]
    f = body["variables"]["filter"]  # type: ignore[index]
    assert f["team"]["id"]["eq"] == "TEAM-1"
    assert f["labels"]["name"]["eq"] == "security"


def test_linear_adapter_rejects_missing_api_key() -> None:
    cfg = LinearTrackerConfig(team_id="TEAM-1")
    with pytest.raises(ValueError, match="LINEAR_API_KEY"):
        LinearIssuesAdapter(httpx.AsyncClient(), cfg, IssueTrackerCredentials())


@pytest.mark.asyncio
async def test_linear_adapter_handles_graphql_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errors": [{"message": "bad request"}]})

    cfg = LinearTrackerConfig(team_id="TEAM-1")
    creds = IssueTrackerCredentials(linear_api_key="key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = LinearIssuesAdapter(client, cfg, creds)
        matches = await adapter.search_known_vulnerability("CVE-2024-9999", None, None, None, None, None)
    assert matches == []


def test_dedupe_tracker_matches() -> None:
    a = TrackerMatch(
        tracker="t",
        issue_id="1",
        issue_url="https://x/y",
        title="x",
        status="open",
        match_tier=1,
        match_field="cve_id",
        matched_value="CVE-1",
    )
    b = TrackerMatch(
        tracker="t",
        issue_id="2",
        issue_url="https://x/y",
        title="dup url",
        status="open",
        match_tier=1,
        match_field="cve_id",
        matched_value="CVE-1",
    )
    assert len(dedupe_tracker_matches([a, b])) == 1

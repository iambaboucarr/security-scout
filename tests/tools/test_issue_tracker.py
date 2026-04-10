from __future__ import annotations

import uuid

import httpx
import pytest

from config import GitHubIssuesTrackerConfig, configure_logging
from models import Finding, Severity, WorkflowKind
from tools.github import GitHubClient
from tools.issue_tracker import (
    GitHubIssuesAdapter,
    ScoutHistoricalAdapter,
    TrackerMatch,
    dedupe_tracker_matches,
    normalise_cve_id,
    run_dedup_checks,
)


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
        adapter = GitHubIssuesAdapter(gh, "acme", "app", cfg)
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
        adapter = GitHubIssuesAdapter(gh, "acme", "app", cfg)
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
        gh_adapter = GitHubIssuesAdapter(gh, "acme", "app", cfg)
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

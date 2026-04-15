# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy import select

import agents.triage as triage_mod
from agents.triage import (
    DependencyHealthSignals,
    apply_dependency_health_to_confidence,
    derive_cvss_base_and_vector,
    github_severity_to_severity,
    infer_exploitation_stage,
    run_advisory_triage,
    structured_base_confidence,
    structured_ssvc_action,
)
from config import (
    JiraTrackerConfig,
    LinearTrackerConfig,
    RepoConfig,
)
from models import Finding, FindingStatus, KnownStatus, Severity, SSVCAction, WorkflowKind
from tools.github import GitHubClient
from tools.issue_tracker import (
    IssueTrackerCredentials,
    JiraIssuesAdapter,
    LinearIssuesAdapter,
    TrackerMatch,
)
from tools.scm import AdvisoryData
from tools.scm.github import GitHubSCMProvider


def _transport(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.github.com",
        transport=handler,
    )


def _repo() -> RepoConfig:
    return RepoConfig(
        name="demo",
        github_org="acme",
        github_repo="app",
        slack_channel="#security",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["critical"],
        issue_trackers=[],
    )


def test_derive_cvss_from_vector() -> None:
    adv = AdvisoryData(
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        source="global",
        summary="x",
        description="y",
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        cvss_score_api=9.8,
    )
    base, vec = derive_cvss_base_and_vector(adv)
    assert base == pytest.approx(9.8)
    assert vec is not None
    assert vec.startswith("CVSS:3.1/")


def test_infer_exploitation_and_ssvc() -> None:
    adv = AdvisoryData(
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        source="global",
        summary="In the wild exploitation reported",
        description="",
    )
    assert infer_exploitation_stage(adv) == "active"
    act = structured_ssvc_action(7.5, "active", Severity.high)
    assert act == SSVCAction.immediate


def test_structured_base_confidence_bounds() -> None:
    adv = AdvisoryData(
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        source="global",
        summary="s",
        description="d",
        cve_ids=("CVE-2024-1",),
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    )
    c = structured_base_confidence(adv, adv.cvss_vector, "poc")
    assert 0.5 < c <= 0.97


def test_apply_dependency_health_adjusts_confidence() -> None:
    h = DependencyHealthSignals(
        github_contributors_upper_bound=1,
        github_contributors_truncated=False,
        upstream_last_push_at=None,
        days_since_upstream_push=400,
        osv_ecosystem="npm",
        osv_prior_vulnerabilities_excluding_current=2,
        osv_query_skipped_reason=None,
    )
    out = apply_dependency_health_to_confidence(0.9, h)
    assert out == pytest.approx(0.82)


def test_github_actions_ecosystem_resolves_owner_repo() -> None:
    assert triage_mod._normalise_github_ecosystem("GitHub Actions") == "github_actions"
    owner, repo = triage_mod._github_owner_repo_from_affected_package(
        "GitHub Actions",
        "actions/checkout",
    )
    assert owner == "actions"
    assert repo == "checkout"


@pytest.mark.asyncio
async def test_run_advisory_triage_persists_finding(db_session) -> None:
    payload = {
        "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
        "summary": "SQL injection",
        "description": "A proof of concept is available.",
        "severity": "high",
        "html_url": "https://github.com/advisories/GHSA-ABCD-EFGH-IJKL",
        "identifiers": [{"type": "CVE", "value": "CVE-2024-5000"}],
        "cwes": [{"cwe_id": "CWE-89", "name": "SQL"}],
        "published_at": "2024-06-01T00:00:00Z",
        "updated_at": None,
        "cvss": {
            "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "score": 9.8,
        },
        "vulnerabilities": [{"package": {"ecosystem": "npm", "name": "some-pkg"}}],
    }

    def gh_handler(request: httpx.Request) -> httpx.Response:
        assert "/repos/acme/app/security-advisories/" in str(request.url)
        return httpx.Response(200, json=payload)

    def osv_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.host == "api.osv.dev"
        return httpx.Response(200, json={"vulns": [{"id": "GHSA-OTHER"}]})

    async with _transport(httpx.MockTransport(gh_handler)) as gh_http:
        gh = GitHubClient("token", client=gh_http)
        scm = GitHubSCMProvider.from_client(gh)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(osv_handler),
        ) as osv_http:
            row = await run_advisory_triage(
                db_session,
                _repo(),
                scm,
                osv_http,
                ghsa_id="GHSA-ABCD-EFGH-IJKL",
                advisory_source="repository",
                llm=None,
            )

    assert row.id is not None
    assert row.cve_id == "CVE-2024-5000"
    assert row.cvss_score == pytest.approx(9.8)
    assert row.ssvc_action == SSVCAction.act
    assert row.triage_confidence is not None
    assert row.triage_confidence > 0.4
    assert row.evidence is not None
    assert row.evidence.get("dedup_matches") == []

    res = await db_session.execute(select(Finding).where(Finding.id == row.id))
    loaded = res.scalar_one()
    assert loaded.title == "SQL injection"


@pytest.mark.asyncio
async def test_run_advisory_triage_llm_refinement_uses_sanitised_prompt(db_session) -> None:
    payload = {
        "ghsa_id": "GHSA-ABCD-EFGH-IJKL",
        "summary": "Borderline",
        "description": "Unclear impact.",
        "severity": "medium",
        "identifiers": [],
        "cwes": [],
    }

    def gh_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    def osv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": []})

    from ai.provider import CompletionResult, TokenUsage

    llm_response = CompletionResult(
        text='{"ssvc_action": "attend", "confidence": 0.71, "rationale": "x"}',
        usage=TokenUsage(input_tokens=100, output_tokens=50),
    )
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=llm_response)
    mock_llm.capabilities = MagicMock(return_value=frozenset())

    async with _transport(httpx.MockTransport(gh_handler)) as gh_http:
        gh = GitHubClient("token", client=gh_http)
        scm = GitHubSCMProvider.from_client(gh)
        async with httpx.AsyncClient(transport=httpx.MockTransport(osv_handler)) as osv_http:
            await run_advisory_triage(
                db_session,
                _repo(),
                scm,
                osv_http,
                ghsa_id="GHSA-ABCD-EFGH-IJKL",
                llm=mock_llm,
                reasoning_model="claude-sonnet-4-6",
            )

    call_kw = mock_llm.complete.call_args.kwargs
    user_blocks = call_kw["messages"][0]["content"]
    assert "external_content" in user_blocks
    assert "advisory_text" in user_blocks


def test_github_severity_mapping() -> None:
    assert github_severity_to_severity("moderate") == Severity.medium


def _accepted_risk_match(*, last_updated: datetime | None) -> TrackerMatch:
    return TrackerMatch(
        tracker="scout_history",
        issue_id="prior-uuid",
        issue_url=None,
        title="Prior accepted risk",
        status="accepted_risk",
        match_tier=1,
        match_field="cve_id",
        matched_value="CVE-2024-1",
        last_updated=last_updated,
    )


def test_accepted_risk_match_within_ttl_returns_match() -> None:
    now = datetime(2026, 4, 13, tzinfo=UTC)
    m = _accepted_risk_match(last_updated=now - timedelta(days=30))
    assert triage_mod._accepted_risk_match([m], ttl_days=90, now=now) is m


def test_accepted_risk_match_past_ttl_returns_none() -> None:
    now = datetime(2026, 4, 13, tzinfo=UTC)
    m = _accepted_risk_match(last_updated=now - timedelta(days=180))
    assert triage_mod._accepted_risk_match([m], ttl_days=90, now=now) is None


def test_accepted_risk_match_zero_ttl_means_never_expire() -> None:
    now = datetime(2026, 4, 13, tzinfo=UTC)
    m = _accepted_risk_match(last_updated=now - timedelta(days=10_000))
    assert triage_mod._accepted_risk_match([m], ttl_days=0, now=now) is m


def test_accepted_risk_match_ignores_non_scout_tracker() -> None:
    now = datetime(2026, 4, 13, tzinfo=UTC)
    other = TrackerMatch(
        tracker="jira",
        issue_id="SEC-1",
        issue_url=None,
        title="t",
        status="accepted_risk",
        match_tier=1,
        match_field="cve_id",
        matched_value="CVE-2024-1",
        last_updated=now,
    )
    assert triage_mod._accepted_risk_match([other], ttl_days=90, now=now) is None


def test_accepted_risk_match_ignores_non_accepted_risk_status() -> None:
    now = datetime(2026, 4, 13, tzinfo=UTC)
    open_match = TrackerMatch(
        tracker="scout_history",
        issue_id="prior",
        issue_url=None,
        title="t",
        status="open",
        match_tier=1,
        match_field="cve_id",
        matched_value="CVE-2024-1",
        last_updated=now,
    )
    assert triage_mod._accepted_risk_match([open_match], ttl_days=90, now=now) is None


def test_build_issue_tracker_adapters_includes_jira_when_credentials_present(db_session) -> None:
    repo = _repo()
    repo_with_jira = repo.model_copy(
        update={
            "issue_trackers": [
                JiraTrackerConfig(project_key="SEC", base_url="https://acme.atlassian.net"),
            ],
        },
    )
    creds = IssueTrackerCredentials(jira_email="ops@acme.io", jira_api_token="t")
    scm = MagicMock()
    http = httpx.AsyncClient()
    adapters = triage_mod._build_issue_tracker_adapters(repo_with_jira, scm, db_session, http, creds)
    assert any(isinstance(a, JiraIssuesAdapter) for a in adapters)


def test_build_issue_tracker_adapters_skips_jira_when_no_token(db_session) -> None:
    repo = _repo()
    repo_with_jira = repo.model_copy(
        update={
            "issue_trackers": [
                JiraTrackerConfig(project_key="SEC", base_url="https://acme.atlassian.net"),
            ],
        },
    )
    scm = MagicMock()
    http = httpx.AsyncClient()
    adapters = triage_mod._build_issue_tracker_adapters(repo_with_jira, scm, db_session, http, None)
    assert not any(isinstance(a, JiraIssuesAdapter) for a in adapters)


def test_build_issue_tracker_adapters_includes_linear_when_credentials_present(db_session) -> None:
    repo = _repo()
    repo_with_linear = repo.model_copy(
        update={"issue_trackers": [LinearTrackerConfig(team_id="TEAM-1")]},
    )
    creds = IssueTrackerCredentials(linear_api_key="key")
    scm = MagicMock()
    http = httpx.AsyncClient()
    adapters = triage_mod._build_issue_tracker_adapters(repo_with_linear, scm, db_session, http, creds)
    assert any(isinstance(a, LinearIssuesAdapter) for a in adapters)


def test_build_issue_tracker_adapters_skips_linear_when_no_key(db_session) -> None:
    repo = _repo()
    repo_with_linear = repo.model_copy(
        update={"issue_trackers": [LinearTrackerConfig(team_id="TEAM-1")]},
    )
    scm = MagicMock()
    http = httpx.AsyncClient()
    adapters = triage_mod._build_issue_tracker_adapters(repo_with_linear, scm, db_session, http, None)
    assert not any(isinstance(a, LinearIssuesAdapter) for a in adapters)


@pytest.mark.asyncio
async def test_run_advisory_triage_flags_known_accepted_risk_when_history_matches(db_session) -> None:
    """End-to-end: a prior accepted-risk Finding within TTL flags incoming as known_accepted_risk."""
    prior = Finding(
        id=uuid.uuid4(),
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-PRIOR",
        severity=Severity.high,
        status=FindingStatus.accepted_risk,
        title="prior",
        cve_id="CVE-2024-5000",
    )
    db_session.add(prior)
    await db_session.commit()

    payload = {
        "ghsa_id": "GHSA-newx-aaaa-bbbb",
        "summary": "Same vuln re-detected",
        "description": "...",
        "severity": "high",
        "html_url": "https://github.com/advisories/GHSA-newx-aaaa-bbbb",
        "identifiers": [{"type": "CVE", "value": "CVE-2024-5000"}],
        "cwes": [],
    }

    def gh_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    def osv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": []})

    async with _transport(httpx.MockTransport(gh_handler)) as gh_http:
        gh = GitHubClient("token", client=gh_http)
        scm = GitHubSCMProvider.from_client(gh)
        async with httpx.AsyncClient(transport=httpx.MockTransport(osv_handler)) as osv_http:
            row = await run_advisory_triage(
                db_session,
                _repo(),
                scm,
                osv_http,
                ghsa_id="GHSA-newx-aaaa-bbbb",
            )

    assert row.known_status == KnownStatus.known_accepted_risk
    assert row.duplicate_tracker == "scout_history"
    assert row.duplicate_of == str(prior.id)


@pytest.mark.asyncio
async def test_run_advisory_triage_does_not_flag_accepted_risk_when_ttl_exceeded(db_session) -> None:
    """Older-than-TTL acceptances drop out: incoming finding behaves as if no accepted-risk match."""
    repo = _repo().model_copy(update={"accepted_risk_ttl_days": 30})
    stale_date = datetime.now(UTC) - timedelta(days=180)
    prior = Finding(
        id=uuid.uuid4(),
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-PRIOR",
        severity=Severity.high,
        status=FindingStatus.accepted_risk,
        title="prior",
        cve_id="CVE-2024-5000",
    )
    db_session.add(prior)
    await db_session.flush()
    # Force created_at to long ago so the TTL filter rejects.
    # ScoutHistoricalAdapter maps Finding.created_at → TrackerMatch.last_updated,
    # so mutating created_at is the correct lever for TTL behaviour.
    prior.created_at = stale_date
    await db_session.commit()

    payload = {
        "ghsa_id": "GHSA-newx-aaaa-bbbb",
        "summary": "Same vuln re-detected",
        "description": "...",
        "severity": "high",
        "identifiers": [{"type": "CVE", "value": "CVE-2024-5000"}],
        "cwes": [],
    }

    def gh_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    def osv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": []})

    async with _transport(httpx.MockTransport(gh_handler)) as gh_http:
        gh = GitHubClient("token", client=gh_http)
        scm = GitHubSCMProvider.from_client(gh)
        async with httpx.AsyncClient(transport=httpx.MockTransport(osv_handler)) as osv_http:
            row = await run_advisory_triage(
                db_session,
                repo,
                scm,
                osv_http,
                ghsa_id="GHSA-newx-aaaa-bbbb",
            )

    assert row.known_status is None
    # The dedup match still surfaces (Finding row exists), but it's not flagged accepted-risk.
    assert row.duplicate_tracker == "scout_history"
    # Verify the intermediate TrackerMatch carried the stale date through to evidence,
    # confirming the TTL path was exercised via Finding.created_at → TrackerMatch.last_updated.
    dedup_matches = row.evidence.get("dedup_matches", [])
    assert len(dedup_matches) >= 1
    scout_match = next(m for m in dedup_matches if m["tracker"] == "scout_history")
    assert scout_match["last_updated"] is not None

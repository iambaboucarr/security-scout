from __future__ import annotations

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
from config import RepoConfig
from models import Finding, Severity, SSVCAction
from tools.github import GitHubClient
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

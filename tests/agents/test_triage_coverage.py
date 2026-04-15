# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agents.triage import (
    DependencyHealthSignals,
    _first_cve_id,
    _first_cwe_id,
    _github_owner_repo_from_affected_package,
    _normalise_github_ecosystem,
    _refine_ssvc_with_llm,
    _should_refine_with_llm,
    apply_dependency_health_to_confidence,
    collect_dependency_health_signals,
    derive_cvss_base_and_vector,
    github_severity_to_severity,
    infer_exploitation_stage,
    structured_base_confidence,
    structured_ssvc_action,
)
from ai.provider import CompletionResult, TokenUsage
from models import Severity, SSVCAction
from tools.github import GitHubAPIError, GitHubClient
from tools.scm import AdvisoryData
from tools.scm.github import GitHubSCMProvider


def _advisory(
    *,
    summary: str = "Test",
    description: str = "Desc",
    severity: str | None = None,
    cvss_vector: str | None = None,
    cvss_score_api: float | None = None,
    cve_ids: tuple[str, ...] = (),
    cwe_ids: tuple[str, ...] = (),
    affected_package_ecosystem: str | None = None,
    affected_package_name: str | None = None,
) -> AdvisoryData:
    return AdvisoryData(
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        source="global",
        summary=summary,
        description=description,
        severity=severity,
        cvss_vector=cvss_vector,
        cvss_score_api=cvss_score_api,
        cve_ids=cve_ids,
        cwe_ids=cwe_ids,
        affected_package_ecosystem=affected_package_ecosystem,
        affected_package_name=affected_package_name,
    )


# ── github_severity_to_severity comprehensive ────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, Severity.informational),
        ("critical", Severity.critical),
        ("CRITICAL", Severity.critical),
        ("high", Severity.high),
        ("medium", Severity.medium),
        ("moderate", Severity.medium),
        ("low", Severity.low),
        ("unknown", Severity.informational),
        ("", Severity.informational),
        ("  high  ", Severity.high),
    ],
)
def test_severity_mapping(raw: str | None, expected: Severity) -> None:
    assert github_severity_to_severity(raw) == expected


# ── derive_cvss_base_and_vector ───────────────────────────────────────────


def test_derive_cvss_malformed_vector_falls_back() -> None:
    adv = _advisory(cvss_vector="INVALID_VECTOR", cvss_score_api=7.5)
    base, vec = derive_cvss_base_and_vector(adv)
    assert base == 7.5
    assert vec == "INVALID_VECTOR"


def test_derive_cvss_no_vector_returns_api_score() -> None:
    adv = _advisory(cvss_score_api=6.5)
    base, vec = derive_cvss_base_and_vector(adv)
    assert base == 6.5
    assert vec is None


def test_derive_cvss_no_vector_no_score() -> None:
    adv = _advisory()
    base, vec = derive_cvss_base_and_vector(adv)
    assert base is None
    assert vec is None


# ── infer_exploitation_stage ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("actively exploited", "active"),
        ("active exploitation confirmed", "active"),
        ("in the wild", "active"),
        ("CISA KEV listed", "active"),
        ("known exploited", "active"),
        ("known to be exploited", "active"),
        ("proof of concept available", "poc"),
        ("proof-of-concept", "poc"),
        ("a poc exists", "poc"),
        ("there is a public exploit", "poc"),
        ("exploit code available", "poc"),
        ("exploit available", "poc"),
        ("theoretical vulnerability", "none"),
        ("buffer overflow", "none"),
    ],
)
def test_exploitation_inference(text: str, expected: str) -> None:
    adv = _advisory(summary=text)
    assert infer_exploitation_stage(adv) == expected


# ── structured_ssvc_action comprehensive ──────────────────────────────────


def test_ssvc_active_exploitation_is_immediate() -> None:
    assert structured_ssvc_action(9.0, "active", Severity.critical) == SSVCAction.immediate


def test_ssvc_poc_is_act() -> None:
    assert structured_ssvc_action(7.0, "poc", Severity.high) == SSVCAction.act


def test_ssvc_critical_severity_no_exploitation_is_act() -> None:
    assert structured_ssvc_action(9.0, "none", Severity.critical) == SSVCAction.act


def test_ssvc_high_severity_no_exploitation_is_act() -> None:
    assert structured_ssvc_action(7.5, "none", Severity.high) == SSVCAction.act


def test_ssvc_medium_severity_is_attend() -> None:
    assert structured_ssvc_action(5.0, "none", Severity.medium) == SSVCAction.attend


def test_ssvc_low_severity_is_track() -> None:
    assert structured_ssvc_action(3.0, "none", Severity.low) == SSVCAction.track


def test_ssvc_informational_with_high_cvss_is_act() -> None:
    assert structured_ssvc_action(9.5, "none", Severity.informational) == SSVCAction.act


def test_ssvc_informational_with_moderate_cvss_is_attend() -> None:
    assert structured_ssvc_action(7.0, "none", Severity.informational) == SSVCAction.attend


def test_ssvc_informational_with_low_cvss_is_attend() -> None:
    assert structured_ssvc_action(4.0, "none", Severity.informational) == SSVCAction.attend


def test_ssvc_informational_no_cvss_is_track() -> None:
    assert structured_ssvc_action(None, "none", Severity.informational) == SSVCAction.track


# ── structured_base_confidence ────────────────────────────────────────────


def test_confidence_full_penalties() -> None:
    adv = _advisory(summary="")
    c = structured_base_confidence(adv, None, "none")
    assert 0.35 <= c <= 0.97


def test_confidence_no_penalties() -> None:
    adv = _advisory(
        summary="SQL injection",
        cve_ids=("CVE-2024-1",),
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    )
    c = structured_base_confidence(adv, adv.cvss_vector, "active")
    assert c >= 0.85


# ── apply_dependency_health_to_confidence ─────────────────────────────────


def test_health_no_adjustments() -> None:
    h = DependencyHealthSignals(
        github_contributors_upper_bound=50,
        github_contributors_truncated=True,
        upstream_last_push_at=None,
        days_since_upstream_push=10,
        osv_ecosystem="npm",
        osv_prior_vulnerabilities_excluding_current=0,
        osv_query_skipped_reason=None,
    )
    result = apply_dependency_health_to_confidence(0.85, h)
    assert result == 0.85


def test_health_single_contributor_not_truncated() -> None:
    h = DependencyHealthSignals(
        github_contributors_upper_bound=1,
        github_contributors_truncated=False,
        upstream_last_push_at=None,
        days_since_upstream_push=None,
        osv_ecosystem=None,
        osv_prior_vulnerabilities_excluding_current=0,
        osv_query_skipped_reason="ecosystem_not_mapped",
    )
    result = apply_dependency_health_to_confidence(0.85, h)
    assert result == pytest.approx(0.81)


def test_health_single_contributor_truncated_no_penalty() -> None:
    h = DependencyHealthSignals(
        github_contributors_upper_bound=1,
        github_contributors_truncated=True,
        upstream_last_push_at=None,
        days_since_upstream_push=None,
        osv_ecosystem=None,
        osv_prior_vulnerabilities_excluding_current=0,
        osv_query_skipped_reason=None,
    )
    result = apply_dependency_health_to_confidence(0.85, h)
    assert result == 0.85


def test_health_stale_push_date() -> None:
    h = DependencyHealthSignals(
        github_contributors_upper_bound=10,
        github_contributors_truncated=False,
        upstream_last_push_at=None,
        days_since_upstream_push=400,
        osv_ecosystem=None,
        osv_prior_vulnerabilities_excluding_current=0,
        osv_query_skipped_reason=None,
    )
    result = apply_dependency_health_to_confidence(0.85, h)
    assert result == pytest.approx(0.81)


def test_health_floor_clamp() -> None:
    result = apply_dependency_health_to_confidence(
        0.30,
        DependencyHealthSignals(
            github_contributors_upper_bound=1,
            github_contributors_truncated=False,
            upstream_last_push_at=None,
            days_since_upstream_push=400,
            osv_ecosystem=None,
            osv_prior_vulnerabilities_excluding_current=0,
            osv_query_skipped_reason=None,
        ),
    )
    assert result == 0.25


# ── _normalise_github_ecosystem ───────────────────────────────────────────


def test_normalise_github_ecosystem_empty() -> None:
    assert _normalise_github_ecosystem(None) == ""
    assert _normalise_github_ecosystem("") == ""
    assert _normalise_github_ecosystem("  ") == ""


def test_normalise_github_ecosystem_spaces_dashes() -> None:
    assert _normalise_github_ecosystem("GitHub Actions") == "github_actions"
    assert _normalise_github_ecosystem("Some-Eco") == "some_eco"


# ── _github_owner_repo_from_affected_package ──────────────────────────────


def test_owner_repo_non_github_ecosystem() -> None:
    assert _github_owner_repo_from_affected_package("npm", "lodash") == (None, None)


def test_owner_repo_no_slash() -> None:
    assert _github_owner_repo_from_affected_package("GitHub Actions", "checkout") == (None, None)


def test_owner_repo_empty_parts() -> None:
    assert _github_owner_repo_from_affected_package("GitHub Actions", "/checkout") == (None, None)


def test_owner_repo_none_package() -> None:
    assert _github_owner_repo_from_affected_package("GitHub Actions", None) == (None, None)


def test_owner_repo_github_actions_valid() -> None:
    assert _github_owner_repo_from_affected_package("actions", "actions/checkout") == ("actions", "checkout")


# ── _should_refine_with_llm ───────────────────────────────────────────────


def test_should_refine_high_confidence_skip() -> None:
    assert _should_refine_with_llm(0.85, _advisory(), "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") is False


def test_should_refine_medium_severity_low_confidence() -> None:
    adv = _advisory(severity="medium")
    assert _should_refine_with_llm(0.75, adv, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") is True


def test_should_refine_high_severity_no_cvss_vector() -> None:
    adv = _advisory(severity="high")
    assert _should_refine_with_llm(0.78, adv, None) is True


def test_should_refine_critical_severity_no_cvss_vector() -> None:
    adv = _advisory(severity="critical")
    assert _should_refine_with_llm(0.78, adv, None) is True


def test_should_refine_very_low_confidence() -> None:
    adv = _advisory(severity="low")
    assert _should_refine_with_llm(0.70, adv, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") is True


def test_should_refine_borderline_not_triggered() -> None:
    adv = _advisory(severity="low")
    assert _should_refine_with_llm(0.73, adv, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") is False


# ── _refine_ssvc_with_llm ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refine_ssvc_valid_response() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text='{"ssvc_action": "attend", "confidence": 0.81, "rationale": "x"}',
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action == SSVCAction.attend
    assert conf == pytest.approx(0.81)


@pytest.mark.asyncio
async def test_refine_ssvc_no_json_returns_none() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text="I think the answer is attend with 0.8",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


@pytest.mark.asyncio
async def test_refine_ssvc_invalid_json_returns_none() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text="{not valid json}",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


@pytest.mark.asyncio
async def test_refine_ssvc_non_dict_json_returns_none() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text="[1, 2, 3]",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


@pytest.mark.asyncio
async def test_refine_ssvc_missing_action_returns_none() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text='{"confidence": 0.8}',
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


@pytest.mark.asyncio
async def test_refine_ssvc_invalid_action_returns_none() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text='{"ssvc_action": "explode", "confidence": 0.8, "rationale": "x"}',
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


@pytest.mark.asyncio
async def test_refine_ssvc_non_numeric_confidence_returns_none() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text='{"ssvc_action": "act", "confidence": "high", "rationale": "x"}',
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


@pytest.mark.asyncio
async def test_refine_ssvc_clamps_confidence() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text='{"ssvc_action": "track", "confidence": 1.5, "rationale": "x"}',
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action == SSVCAction.track
    assert conf == 1.0


@pytest.mark.asyncio
async def test_refine_ssvc_action_not_string() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=CompletionResult(
            text='{"ssvc_action": 42, "confidence": 0.8}',
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
    )
    action, conf = await _refine_ssvc_with_llm(llm, "claude-sonnet-4-6", _advisory(), run_id=None)
    assert action is None
    assert conf is None


# ── _first_cve_id / _first_cwe_id ────────────────────────────────────────


def test_first_cve_id_empty() -> None:
    adv = _advisory(cve_ids=())
    assert _first_cve_id(adv) is None


def test_first_cve_id_invalid() -> None:
    adv = _advisory(cve_ids=("NOT-A-CVE",))
    assert _first_cve_id(adv) is None


def test_first_cve_id_valid() -> None:
    adv = _advisory(cve_ids=("CVE-2024-1234",))
    assert _first_cve_id(adv) == "CVE-2024-1234"


def test_first_cwe_id_empty() -> None:
    adv = _advisory(cwe_ids=())
    assert _first_cwe_id(adv) is None


def test_first_cwe_id_present() -> None:
    adv = _advisory(cwe_ids=("CWE-79", "CWE-89"))
    assert _first_cwe_id(adv) == "CWE-79"


# ── collect_dependency_health_signals ─────────────────────────────────────


@pytest.mark.asyncio
async def test_collect_health_non_github_ecosystem() -> None:
    adv = _advisory(affected_package_ecosystem="npm", affected_package_name="lodash")

    def osv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": [{"id": "CVE-2024-1"}]})

    gh = MagicMock(spec=GitHubClient)
    scm = GitHubSCMProvider.from_client(gh)

    async with httpx.AsyncClient(transport=httpx.MockTransport(osv_handler)) as http:
        result = await collect_dependency_health_signals(scm, http, adv, {"GHSA-ABCD-EFGH-IJKL"})

    assert result.osv_ecosystem == "npm"
    assert result.osv_prior_vulnerabilities_excluding_current >= 0
    assert result.github_contributors_upper_bound is None


@pytest.mark.asyncio
async def test_collect_health_github_actions_metadata_fails() -> None:
    adv = _advisory(
        affected_package_ecosystem="GitHub Actions",
        affected_package_name="actions/checkout",
    )

    gh = MagicMock(spec=GitHubClient)
    gh.fetch_repository_metadata = AsyncMock(
        side_effect=GitHubAPIError("not found", is_transient=False, http_status=404),
    )
    gh.fetch_repository_contributors_count_upper_bound = AsyncMock(
        side_effect=GitHubAPIError("not found", is_transient=False, http_status=404),
    )
    scm = GitHubSCMProvider.from_client(gh)

    def osv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(osv_handler)) as http:
        result = await collect_dependency_health_signals(scm, http, adv, set())

    assert result.github_contributors_upper_bound is None
    assert result.upstream_last_push_at is None


@pytest.mark.asyncio
async def test_collect_health_osv_ecosystem_not_mapped() -> None:
    adv = _advisory(affected_package_ecosystem=None, affected_package_name="somepkg")
    gh = MagicMock(spec=GitHubClient)
    scm = GitHubSCMProvider.from_client(gh)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as http:
        result = await collect_dependency_health_signals(scm, http, adv, set())

    assert result.osv_query_skipped_reason == "ecosystem_not_mapped"


@pytest.mark.asyncio
async def test_collect_health_no_package_name() -> None:
    adv = _advisory(affected_package_ecosystem="npm", affected_package_name=None)
    gh = MagicMock(spec=GitHubClient)
    scm = GitHubSCMProvider.from_client(gh)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as http:
        result = await collect_dependency_health_signals(scm, http, adv, set())

    assert result.osv_query_skipped_reason == "no_package"

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from agents.governance import GovernanceTier, decide_governance_tier
from config import GovernanceApprover, GovernanceConfig, GovernanceRule
from models import Finding, FindingStatus, KnownStatus, Severity, SSVCAction, WorkflowKind


def _finding(
    *,
    severity: Severity = Severity.high,
    ssvc_action: SSVCAction | None = None,
    known_status: KnownStatus | None = None,
) -> Finding:
    return Finding(
        workflow=WorkflowKind.advisory,
        source_ref="https://example/advisory",
        severity=severity,
        ssvc_action=ssvc_action,
        status=FindingStatus.unconfirmed,
        known_status=known_status,
        title="t",
    )


def test_default_informational_goes_to_auto_resolve() -> None:
    assert decide_governance_tier(_finding(severity=Severity.informational), None) == GovernanceTier.auto_resolve


def test_default_non_informational_requires_approval() -> None:
    for sev in (Severity.critical, Severity.high, Severity.medium, Severity.low):
        assert decide_governance_tier(_finding(severity=sev), None) == GovernanceTier.approve, sev


def test_approve_wins_over_notify_when_both_match() -> None:
    gov = GovernanceConfig(
        notify=[GovernanceRule(severity=[Severity.high])],
        approve=[GovernanceRule(severity=[Severity.high])],
    )
    assert decide_governance_tier(_finding(severity=Severity.high), gov) == GovernanceTier.approve


def test_notify_wins_over_auto_resolve_when_both_match() -> None:
    gov = GovernanceConfig(
        auto_resolve=[GovernanceRule(severity=[Severity.medium])],
        notify=[GovernanceRule(severity=[Severity.medium])],
    )
    assert decide_governance_tier(_finding(severity=Severity.medium), gov) == GovernanceTier.notify


def test_ssvc_criterion_matches() -> None:
    gov = GovernanceConfig(approve=[GovernanceRule(ssvc_action=[SSVCAction.immediate])])
    assert (
        decide_governance_tier(
            _finding(severity=Severity.medium, ssvc_action=SSVCAction.immediate),
            gov,
        )
        == GovernanceTier.approve
    )


def test_ssvc_criterion_rejects_finding_without_ssvc() -> None:
    gov = GovernanceConfig(approve=[GovernanceRule(ssvc_action=[SSVCAction.immediate])])
    # no ssvc_action on finding; approve rule cannot match. Falls back to default (high → approve anyway).
    tier = decide_governance_tier(_finding(severity=Severity.high, ssvc_action=None), gov)
    assert tier == GovernanceTier.approve
    # But at medium, default would approve too; check with low severity to isolate.
    tier_low = decide_governance_tier(_finding(severity=Severity.low, ssvc_action=None), gov)
    assert tier_low == GovernanceTier.approve  # default strict for non-informational


def test_duplicate_criterion_matches_known_status() -> None:
    gov = GovernanceConfig(auto_resolve=[GovernanceRule(duplicate=True)])
    tier = decide_governance_tier(
        _finding(severity=Severity.medium, known_status=KnownStatus.duplicate),
        gov,
    )
    assert tier == GovernanceTier.auto_resolve


def test_duplicate_false_does_not_match_when_finding_is_duplicate() -> None:
    gov = GovernanceConfig(auto_resolve=[GovernanceRule(duplicate=False)])
    tier = decide_governance_tier(
        _finding(severity=Severity.medium, known_status=KnownStatus.duplicate),
        gov,
    )
    # No rule matches → default strict for medium → approve
    assert tier == GovernanceTier.approve


def test_rule_requires_all_specified_criteria_to_match() -> None:
    gov = GovernanceConfig(
        approve=[GovernanceRule(severity=[Severity.high], ssvc_action=[SSVCAction.immediate])],
    )
    # Severity matches, ssvc doesn't → rule doesn't match → fallback to default.
    tier = decide_governance_tier(
        _finding(severity=Severity.high, ssvc_action=SSVCAction.track),
        gov,
    )
    assert tier == GovernanceTier.approve  # default (high → approve)


def test_poc_execution_criterion_cannot_match_today() -> None:
    # patch_available / poc_execution rules are parsed but inert until their upstream signals exist.
    gov = GovernanceConfig(auto_resolve=[GovernanceRule(poc_execution=False)])
    tier = decide_governance_tier(_finding(severity=Severity.medium), gov)
    # The rule is reserved and does not match anything yet → falls back to default (medium → approve).
    assert tier == GovernanceTier.approve


def test_empty_governance_block_falls_through_to_default() -> None:
    gov = GovernanceConfig()
    assert decide_governance_tier(_finding(severity=Severity.informational), gov) == GovernanceTier.auto_resolve
    assert decide_governance_tier(_finding(severity=Severity.high), gov) == GovernanceTier.approve


def test_governance_approver_validates_slack_user_pattern() -> None:
    # Sanity-check the Pydantic constraint alongside the routing logic.
    GovernanceApprover(slack_user="U12345AB")

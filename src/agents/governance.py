# SPDX-License-Identifier: Apache-2.0
"""Governance routing: maps a triaged finding to an auto_resolve / notify / approve tier.

Pure logic, no I/O. Precedence is approve > notify > auto_resolve (most restrictive wins) so
an ``auto_resolve`` rule matching a critical finding never suppresses an ``approve`` rule that
also matches. Default behaviour (no ``governance`` block configured) preserves strict
approval for anything above informational severity.
"""

from __future__ import annotations

from enum import StrEnum

from config import GovernanceConfig, GovernanceRule
from models import Finding, KnownStatus, Severity

__all__ = ["GovernanceTier", "decide_governance_tier"]


class GovernanceTier(StrEnum):
    auto_resolve = "auto_resolve"
    notify = "notify"
    approve = "approve"


def _default_tier(finding: Finding) -> GovernanceTier:
    if finding.severity == Severity.informational:
        return GovernanceTier.auto_resolve
    return GovernanceTier.approve


def _rule_matches(rule: GovernanceRule, finding: Finding) -> bool:
    if rule.severity is not None and finding.severity not in rule.severity:
        return False
    if rule.ssvc_action is not None and (finding.ssvc_action is None or finding.ssvc_action not in rule.ssvc_action):
        return False
    if rule.duplicate is not None:
        is_dup = finding.known_status == KnownStatus.duplicate
        if rule.duplicate != is_dup:
            return False
    # ``patch_available`` / ``poc_execution`` are reserved for future workflows that will
    # populate those signals on the Finding. Until then, a rule that specifies either
    # criterion cannot match anything — treat as no-match rather than error.
    return not (rule.patch_available is not None or rule.poc_execution is not None)


def decide_governance_tier(
    finding: Finding,
    governance: GovernanceConfig | None,
) -> GovernanceTier:
    if governance is None:
        return _default_tier(finding)
    if any(_rule_matches(r, finding) for r in governance.approve):
        return GovernanceTier.approve
    if any(_rule_matches(r, finding) for r in governance.notify):
        return GovernanceTier.notify
    if any(_rule_matches(r, finding) for r in governance.auto_resolve):
        return GovernanceTier.auto_resolve
    return _default_tier(finding)

#!/usr/bin/env python3
"""Smoke test: verify GitHub PAT access to dhis2-core advisories.

Hits the real GitHub API using the PAT from .env. Tests:
  1. Repository metadata (confirms PAT + repo access)
  2. List all repository security advisories (confirms advisory list scope)
  3. Fetch single repository advisory (confirms advisory read scope)
  4. Global security advisory (confirms global advisory access)
  5. Contributor count (confirms general API access)

Usage:
    uv run python scripts/smoke_test_github.py
    # Custom org/repo:
    uv run python scripts/smoke_test_github.py --owner dhis2 --repo dhis2-core
    # Specific advisory:
    uv run python scripts/smoke_test_github.py --ghsa GHSA-fj38-585h-hxgj
    # Filter by state/severity:
    uv run python scripts/smoke_test_github.py --state published --severity high
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src/ to path so imports work without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Settings
from tools.github import GitHubAPIError, GitHubClient


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"


def _print_advisory_compact(adv: object, index: int) -> None:
    ghsa = getattr(adv, "ghsa_id", "?")
    sev = getattr(adv, "severity", "?") or "?"
    summary = getattr(adv, "summary", "")
    cves = getattr(adv, "cve_ids", ())
    cve_str = f" ({', '.join(cves)})" if cves else ""
    sev_color = {"critical": _red, "high": _red, "medium": _yellow, "low": _dim}.get(
        sev.lower() if isinstance(sev, str) else "", lambda s: s
    )
    sev_label = sev_color(f"{sev.upper():>8s}")
    print(f"    {index:>3}. {ghsa}  {sev_label}{cve_str}")
    if summary:
        print(f"         {summary[:90]}{'...' if len(summary) > 90 else ''}")


def _print_advisory_detail(adv: object, label: str) -> None:
    print(f"\n  {_bold(label)}")
    for field in (
        "ghsa_id", "source", "summary", "severity", "cve_ids", "cwe_ids",
        "cvss_vector", "cvss_score_api", "affected_package_name",
        "affected_package_ecosystem", "html_url", "published_at",
    ):
        val = getattr(adv, field, None)
        if val is not None and val != () and val != "":
            print(f"    {field}: {val}")


async def run(
    owner: str,
    repo: str,
    ghsa_id: str | None,
    *,
    state: str | None,
    severity: str | None,
) -> bool:
    settings = Settings()

    pat_preview = settings.github_pat[:8] + "..." + settings.github_pat[-4:]
    print(f"\n{_bold('GitHub PAT Smoke Test')}")
    print(f"  PAT: {pat_preview}")
    print(f"  Target: {owner}/{repo}")
    if state:
        print(f"  Filter state: {state}")
    if severity:
        print(f"  Filter severity: {severity}")
    print()

    if settings.github_pat in ("dev-local-github-pat", "ghp_your-fine-grained-pat"):
        print(_red("FAIL: PAT is still a placeholder. Set GITHUB_PAT in .env"))
        return False

    all_passed = True
    discovered_ghsa: str | None = None

    async with GitHubClient(settings.github_pat) as gh:
        # --- Test 1: Repository metadata ---
        print(f"[1/5] Fetching repository metadata for {owner}/{repo}...")
        try:
            meta = await gh.fetch_repository_metadata(owner, repo)
            print(_green(f"  PASS: {meta.full_name}"))
            print(f"    description: {meta.description}")
            print(f"    private: {meta.private}")
            print(f"    default_branch: {meta.default_branch}")
            print(f"    stars: {meta.stargazers_count}  forks: {meta.forks_count}")
            print(f"    language: {meta.language}")
            print(f"    pushed_at: {meta.pushed_at}")
        except GitHubAPIError as exc:
            print(_red(f"  FAIL: {exc} (HTTP {exc.http_status})"))
            all_passed = False

        # --- Test 2: List all advisories ---
        print(f"\n[2/5] Listing security advisories for {owner}/{repo}...")
        try:
            advisories = await gh.list_repository_security_advisories(
                owner,
                repo,
                state=state,
                severity=severity,
                per_page=100,
            )
            print(_green(f"  PASS: {len(advisories)} advisories found"))
            if advisories:
                # Show severity breakdown
                sev_counts: dict[str, int] = {}
                for a in advisories:
                    s = (a.severity or "unknown").lower()
                    sev_counts[s] = sev_counts.get(s, 0) + 1
                breakdown = ", ".join(f"{v} {k}" for k, v in sorted(sev_counts.items()))
                print(f"    Severity breakdown: {breakdown}")
                print()

                # Show all advisories
                for i, a in enumerate(advisories, 1):
                    _print_advisory_compact(a, i)

                # Use first advisory for single-fetch test if no --ghsa given
                if ghsa_id is None:
                    discovered_ghsa = advisories[0].ghsa_id
            else:
                print(_yellow("    No advisories found (filters may be too restrictive)"))
        except GitHubAPIError as exc:
            print(_red(f"  FAIL: {exc} (HTTP {exc.http_status})"))
            all_passed = False

        # --- Test 3: Fetch single advisory ---
        test_ghsa = ghsa_id or discovered_ghsa
        if test_ghsa:
            print(f"\n[3/5] Fetching repository advisory {test_ghsa}...")
            try:
                adv = await gh.fetch_repository_security_advisory(owner, repo, test_ghsa)
                print(_green("  PASS: Repository advisory found"))
                _print_advisory_detail(adv, "Repository Advisory")
            except GitHubAPIError as exc:
                if exc.http_status == 404:
                    print(_yellow("  SKIP: 404 — advisory may not exist as repo advisory"))
                else:
                    print(_red(f"  FAIL: {exc} (HTTP {exc.http_status})"))
                    all_passed = False
        else:
            print(f"\n[3/5] {_yellow('SKIP: no advisory ID available for single-fetch test')}")

        # --- Test 4: Global advisory ---
        if test_ghsa:
            print(f"\n[4/5] Fetching global advisory {test_ghsa}...")
            try:
                gadv = await gh.fetch_global_security_advisory(test_ghsa)
                print(_green("  PASS: Global advisory found"))
                _print_advisory_detail(gadv, "Global Advisory")
            except GitHubAPIError as exc:
                if exc.http_status == 404:
                    print(_yellow("  SKIP: 404 — advisory not in global database"))
                else:
                    print(_red(f"  FAIL: {exc} (HTTP {exc.http_status})"))
                    all_passed = False
        else:
            print(f"\n[4/5] {_yellow('SKIP: no advisory ID available for global-fetch test')}")

        # --- Test 5: Contributors ---
        print(f"\n[5/5] Fetching contributor count for {owner}/{repo}...")
        try:
            count, truncated = await gh.fetch_repository_contributors_count_upper_bound(
                owner, repo
            )
            suffix = "+" if truncated else ""
            print(_green(f"  PASS: {count}{suffix} contributors"))
        except GitHubAPIError as exc:
            print(_red(f"  FAIL: {exc} (HTTP {exc.http_status})"))
            all_passed = False

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if all_passed:
        print(_green(_bold("ALL TESTS PASSED — PAT access confirmed")))
        print(f"  Your PAT can list and read {owner}/{repo} advisories.")
    else:
        print(_red(_bold("SOME TESTS FAILED — check PAT scopes")))
        print("  Required fine-grained PAT permissions:")
        print("    - Repository: Security advisories (read)")
        print("    - Repository: Metadata (read)")
        print("    - Account: Security advisories (read)")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test GitHub PAT access")
    parser.add_argument("--owner", default="dhis2", help="GitHub org/owner")
    parser.add_argument("--repo", default="dhis2-core", help="Repository name")
    parser.add_argument("--ghsa", default=None, help="GHSA ID to fetch (auto-discovered if omitted)")
    parser.add_argument("--state", default=None, help="Filter by state: published, closed, draft, triage")
    parser.add_argument("--severity", default=None, help="Filter by severity: critical, high, medium, low")
    args = parser.parse_args()

    passed = asyncio.run(run(args.owner, args.repo, args.ghsa, state=args.state, severity=args.severity))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

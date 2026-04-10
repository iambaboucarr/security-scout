"""OSV.dev queries for supply-chain context (ADR-016 §6).

IDs returned include historical CVEs for the package; callers should not treat
raw counts as malicious-package proof without a dedicated OSV/malicious signal.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

import httpx
import structlog

_LOG = structlog.get_logger(__name__)

OSV_QUERY_URL: Final[str] = "https://api.osv.dev/v1/query"

# GitHub Security Advisory `package.ecosystem` -> OSV `package.ecosystem`
_GITHUB_TO_OSV_ECOSYSTEM: Final[dict[str, str]] = {
    "npm": "npm",
    "pip": "PyPI",
    "pypi": "PyPI",
    "rubygems": "RubyGems",
    "go": "Go",
    "rust": "crates.io",
    "cargo": "crates.io",
    "maven": "Maven",
    "composer": "Packagist",
    "nuget": "NuGet",
    "hex": "Hex",
    "pub": "Pub",
}


def github_ecosystem_to_osv(ecosystem: str | None) -> str | None:
    if ecosystem is None:
        return None
    key = ecosystem.strip()
    if not key:
        return None
    mapped = _GITHUB_TO_OSV_ECOSYSTEM.get(key.lower())
    if mapped is not None:
        return mapped
    if re.fullmatch(r"[a-z0-9_-]+", key, flags=re.IGNORECASE):
        return key
    return None


async def query_osv_vulnerability_ids(
    client: httpx.AsyncClient,
    package_name: str,
    ecosystem: str,
) -> list[str]:
    """Return OSV vulnerability IDs for ``package_name`` in ``ecosystem`` (may be empty)."""
    payload = {"package": {"name": package_name, "ecosystem": ecosystem}}
    try:
        response = await client.post(OSV_QUERY_URL, json=payload, timeout=30.0)
    except httpx.HTTPError:
        _LOG.warning("osv_query_failed", error_type="HTTPError")
        return []
    if not response.is_success:
        _LOG.warning("osv_query_http", status_code=response.status_code)
        return []
    try:
        data = response.json()
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    vulns = data.get("vulns")
    if not isinstance(vulns, list):
        return []
    out: list[str] = []
    for item in vulns:
        if not isinstance(item, dict):
            continue
        vid = item.get("id")
        if isinstance(vid, str) and vid:
            out.append(vid)
    return out


def count_prior_vulnerabilities(ids: Sequence[str], exclude_ids: set[str]) -> int:
    """Count IDs not in ``exclude_ids`` (case-insensitive for CVE/GHSA-style ids)."""
    ex = {x.upper() for x in exclude_ids}
    n = 0
    for vid in ids:
        if vid.upper() not in ex:
            n += 1
    return n

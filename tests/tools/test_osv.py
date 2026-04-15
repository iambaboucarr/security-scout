# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import httpx
import pytest

from tools.osv import (
    count_prior_vulnerabilities,
    github_ecosystem_to_osv,
    query_osv_vulnerability_ids,
)

# ── github_ecosystem_to_osv ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("input_eco", "expected"),
    [
        ("npm", "npm"),
        ("pip", "PyPI"),
        ("pypi", "PyPI"),
        ("rubygems", "RubyGems"),
        ("go", "Go"),
        ("rust", "crates.io"),
        ("cargo", "crates.io"),
        ("maven", "Maven"),
        ("composer", "Packagist"),
        ("nuget", "NuGet"),
        ("hex", "Hex"),
        ("pub", "Pub"),
        ("NPM", "npm"),
        ("PIP", "PyPI"),
    ],
)
def test_known_ecosystem_mappings(input_eco: str, expected: str) -> None:
    assert github_ecosystem_to_osv(input_eco) == expected


def test_ecosystem_none() -> None:
    assert github_ecosystem_to_osv(None) is None


def test_ecosystem_empty() -> None:
    assert github_ecosystem_to_osv("") is None


def test_ecosystem_whitespace_only() -> None:
    assert github_ecosystem_to_osv("   ") is None


def test_ecosystem_unknown_alphanumeric_passthrough() -> None:
    assert github_ecosystem_to_osv("swift") == "swift"


def test_ecosystem_with_hyphens_underscores() -> None:
    assert github_ecosystem_to_osv("my-ecosystem_2") == "my-ecosystem_2"


def test_ecosystem_with_special_chars_returns_none() -> None:
    assert github_ecosystem_to_osv("bad/eco") is None
    assert github_ecosystem_to_osv("bad eco") is None
    assert github_ecosystem_to_osv("eco!") is None


def test_ecosystem_leading_trailing_whitespace_stripped() -> None:
    assert github_ecosystem_to_osv("  npm  ") == "npm"


# ── query_osv_vulnerability_ids ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_osv_success_with_vulns() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "vulns": [
                    {"id": "CVE-2024-1234"},
                    {"id": "GHSA-ABCD-EFGH-IJKL"},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "some-pkg", "npm")

    assert result == ["CVE-2024-1234", "GHSA-ABCD-EFGH-IJKL"]


@pytest.mark.asyncio
async def test_query_osv_success_empty_vulns() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "safe-pkg", "PyPI")

    assert result == []


@pytest.mark.asyncio
async def test_query_osv_success_no_vulns_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == []


@pytest.mark.asyncio
async def test_query_osv_non_success_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == []


@pytest.mark.asyncio
async def test_query_osv_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == []


@pytest.mark.asyncio
async def test_query_osv_non_dict_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == []


@pytest.mark.asyncio
async def test_query_osv_vulns_not_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": "oops"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == []


@pytest.mark.asyncio
async def test_query_osv_non_dict_items_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": [42, "str", {"id": "CVE-2024-1"}, {"no_id": True}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == ["CVE-2024-1"]


@pytest.mark.asyncio
async def test_query_osv_empty_id_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vulns": [{"id": ""}, {"id": "CVE-2024-2"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == ["CVE-2024-2"]


@pytest.mark.asyncio
async def test_query_osv_http_error_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await query_osv_vulnerability_ids(client, "pkg", "npm")

    assert result == []


# ── count_prior_vulnerabilities ───────────────────────────────────────────


def test_count_prior_basic() -> None:
    ids = ["CVE-2024-1", "CVE-2024-2", "GHSA-XXXX-YYYY-ZZZZ"]
    assert count_prior_vulnerabilities(ids, {"CVE-2024-1"}) == 2


def test_count_prior_case_insensitive() -> None:
    ids = ["cve-2024-1"]
    assert count_prior_vulnerabilities(ids, {"CVE-2024-1"}) == 0


def test_count_prior_empty_ids() -> None:
    assert count_prior_vulnerabilities([], {"CVE-2024-1"}) == 0


def test_count_prior_empty_exclusions() -> None:
    assert count_prior_vulnerabilities(["A", "B"], set()) == 2

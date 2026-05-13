# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import cli

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(
        transport=transport,
        base_url="http://api.test/api/v1",
        headers={
            "Authorization": "Bearer test-token",
            "Accept": "application/json",
            "User-Agent": "scout-cli",
        },
        timeout=5.0,
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def auth_env() -> dict[str, str]:
    return {"SCOUT_API_KEY": "test-token", "SCOUT_API_URL": "http://api.test"}


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, httpx.Request] = {}

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> dict[str, httpx.Request]:
        captured.clear()

        def wrapped(request: httpx.Request) -> httpx.Response:
            captured["last"] = request
            captured.setdefault("all", [])
            captured["all"].append(request)  # type: ignore[arg-type]
            return handler(request)

        monkeypatch.setattr(cli, "_make_client", lambda _base, _token: _mock_client(wrapped))
        return captured

    return install


# ── Auth resolution ──────────────────────────────────────────────────────────


def test_resolve_token_prefers_env(tmp_path: Path) -> None:
    out = cli.resolve_token(env={"SCOUT_API_KEY": "from-env"}, token_path=tmp_path / "missing")
    assert out == "from-env"


def test_resolve_token_strips_whitespace(tmp_path: Path) -> None:
    out = cli.resolve_token(env={"SCOUT_API_KEY": "  spaced  "}, token_path=tmp_path / "missing")
    assert out == "spaced"


def test_resolve_token_reads_file_when_env_missing(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("file-token\n", encoding="utf-8")
    if hasattr(os, "chmod"):
        os.chmod(token_path, 0o600)
    assert cli.resolve_token(env={}, token_path=token_path) == "file-token"


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX-only permission check")
def test_resolve_token_rejects_loose_file_permissions(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("file-token", encoding="utf-8")
    os.chmod(token_path, 0o644)
    with pytest.raises(cli.CliError) as exc_info:
        cli.resolve_token(env={}, token_path=token_path)
    assert "too open" in str(exc_info.value.message)
    assert "chmod 600" in str(exc_info.value.message)


def test_resolve_token_missing_fails_with_grant_hint(tmp_path: Path) -> None:
    with pytest.raises(cli.CliError) as exc_info:
        cli.resolve_token(env={}, token_path=tmp_path / "missing")
    msg = exc_info.value.message
    assert "No API token found" in msg
    assert "python -m src.admin create_token" in msg


def test_resolve_token_empty_file_falls_through(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("   \n", encoding="utf-8")
    if hasattr(os, "chmod"):
        os.chmod(token_path, 0o600)
    with pytest.raises(cli.CliError):
        cli.resolve_token(env={}, token_path=token_path)


def test_resolve_token_rejects_control_chars_in_env() -> None:
    with pytest.raises(cli.CliError) as exc_info:
        cli.resolve_token(env={"SCOUT_API_KEY": "abc\r\nX-Foo: bar"})
    assert "control characters" in str(exc_info.value.message)


def test_resolve_token_rejects_control_chars_in_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("abc\x01def", encoding="utf-8")
    if hasattr(os, "chmod"):
        os.chmod(token_path, 0o600)
    with pytest.raises(cli.CliError) as exc_info:
        cli.resolve_token(env={}, token_path=token_path)
    assert "control characters" in str(exc_info.value.message)
    assert str(token_path) in str(exc_info.value.message)


@pytest.mark.skipif(not hasattr(os, "chmod"), reason="POSIX-only chmod 0 test")
def test_read_token_file_oserror_on_read(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("x", encoding="utf-8")
    os.chmod(token_path, 0o600)
    # Drop read permission for the owner: read_text() raises PermissionError → CliError.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses POSIX permission checks")
    os.chmod(token_path, 0o000)
    try:
        with pytest.raises(cli.CliError) as exc_info:
            cli._read_token_file(token_path)
        assert "unable to read token file" in str(exc_info.value.message)
    finally:
        os.chmod(token_path, 0o600)


def test_resolve_base_url_default() -> None:
    assert cli.resolve_base_url({}) == cli.DEFAULT_API_URL


def test_resolve_base_url_strips_trailing_slash() -> None:
    assert cli.resolve_base_url({"SCOUT_API_URL": "https://api.example.com/"}) == "https://api.example.com"


def test_resolve_base_url_falls_back_when_empty() -> None:
    assert cli.resolve_base_url({"SCOUT_API_URL": "   "}) == cli.DEFAULT_API_URL


# ── Helpers ──────────────────────────────────────────────────────────────────


def test_parse_pkg_at_version() -> None:
    assert cli._parse_pkg_at_version("requests@2.31.0") == ("requests", "2.31.0")
    assert cli._parse_pkg_at_version("requests") == ("requests", "")
    assert cli._parse_pkg_at_version("@scope/pkg@1.0") == ("@scope/pkg", "1.0")
    assert cli._parse_pkg_at_version("@scope/pkg") == ("@scope/pkg", "")
    assert cli._parse_pkg_at_version("  requests@1.0  ") == ("requests", "1.0")


def test_split_csv() -> None:
    assert cli._split_csv("a,b ,c") == ["a", "b", "c"]
    assert cli._split_csv("") == []
    assert cli._split_csv(None) == []


def test_dedupe_findings_keeps_first() -> None:
    rows = [{"id": "a", "v": 1}, {"id": "a", "v": 2}, {"id": "b", "v": 3}]
    assert cli._dedupe_findings(rows) == [{"id": "a", "v": 1}, {"id": "b", "v": 3}]


def test_render_table_aligns_columns() -> None:
    out = cli._render_table([{"a": "1", "b": "long"}, {"a": "22", "b": "x"}], ["a", "b"])
    lines = out.splitlines()
    assert lines[0].startswith("a")
    # column widths consistent
    assert len(lines[0]) == len(lines[1]) == len(lines[2])


def test_render_table_empty_rows_returns_marker() -> None:
    assert cli._render_table([], ["a"]) == "(no rows)"


def test_drop_none() -> None:
    assert cli._drop_none({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}


# ── findings ─────────────────────────────────────────────────────────────────


def test_findings_default_format_is_json(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "1", "title": "t", "severity": "high"}])

    patched_client(handler)
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo"], env=auth_env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["id"] == "1"


def test_findings_passes_severity_status_limit(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    requests_seen: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests_seen.append(dict(req.url.params))
        return httpx.Response(200, json=[])

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        [
            "findings",
            "--repo",
            "demo",
            "--severity",
            "critical",
            "--status",
            "confirmed_high",
            "--limit",
            "10",
        ],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert requests_seen == [
        {"repo": "demo", "severity": "critical", "finding_status": "confirmed_high", "limit": "10"},
    ]


def test_findings_multi_severity_fans_out_and_dedupes(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        sev = req.url.params.get("severity")
        if sev == "critical":
            return httpx.Response(200, json=[{"id": "a", "severity": "critical"}])
        if sev == "high":
            return httpx.Response(
                200,
                json=[
                    {"id": "a", "severity": "critical"},  # duplicate from prior call
                    {"id": "b", "severity": "high"},
                ],
            )
        return httpx.Response(200, json=[])

    captured = patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["findings", "--repo", "demo", "--severity", "critical,high"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [row["id"] for row in payload] == ["a", "b"]
    assert len(captured["all"]) == 2


def test_findings_table_format(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"id": "abc", "title": "Bad", "severity": "high", "status": "unconfirmed"}],
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["findings", "--repo", "demo", "--format", "table"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert "id" in result.output
    assert "abc" in result.output
    assert "Bad" in result.output


def test_findings_invalid_limit_rejected(runner: CliRunner, auth_env: dict[str, str]) -> None:
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo", "--limit", "0"], env=auth_env)
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "Error" in result.output


# ── check ────────────────────────────────────────────────────────────────────


def test_check_parses_pkg_at_version_and_passes_to_api(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["package"] == "requests"
        assert req.url.params["version"] == "2.31.0"
        assert req.url.params["ecosystem"] == "pypi"
        return httpx.Response(
            200,
            json={
                "package": "requests",
                "version": "2.31.0",
                "ecosystem": "pypi",
                "advisory_count": 1,
                "advisories": [
                    {
                        "id": "a-1",
                        "title": "RCE",
                        "severity": "high",
                        "ssvc_action": "act",
                        "source_ref": "GHSA-x",
                    },
                ],
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["check", "requests@2.31.0", "--ecosystem", "pypi"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["advisory_count"] == 1


def test_check_table_renders_summary_and_advisories(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "package": "p",
                "version": "1",
                "ecosystem": "npm",
                "advisory_count": 2,
                "advisories": [
                    {"id": "x", "title": "T1", "severity": "high", "ssvc_action": None, "source_ref": "S"},
                    {"id": "y", "title": "T2", "severity": "low", "ssvc_action": None, "source_ref": "S"},
                ],
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["check", "p@1", "--ecosystem", "npm", "--format", "table"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert "advisory_count" in result.output
    assert "T1" in result.output
    assert "T2" in result.output


def test_check_table_no_advisories(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "package": "p",
                "version": "1",
                "ecosystem": "npm",
                "advisory_count": 0,
                "advisories": [],
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["check", "p@1", "--format", "table"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert "(no advisories)" in result.output


def test_check_missing_package_name_fails(
    runner: CliRunner,
    auth_env: dict[str, str],
) -> None:
    # Whitespace-only target strips to empty package, which we explicitly reject.
    result = runner.invoke(cli.scout, ["check", "   "], env=auth_env)
    assert result.exit_code == 1
    assert "missing package name" in result.output


def test_check_scoped_npm_package(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["package"] == "@scope/pkg"
        assert req.url.params["version"] == "1.0.0"
        return httpx.Response(
            200,
            json={
                "package": "@scope/pkg",
                "version": "1.0.0",
                "ecosystem": "npm",
                "advisory_count": 0,
                "advisories": [],
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["check", "@scope/pkg@1.0.0", "--ecosystem", "npm"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["package"] == "@scope/pkg"


# ── finding ──────────────────────────────────────────────────────────────────


def test_finding_summary_default(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    full = {
        "id": "abc",
        "title": "T",
        "description": "desc",
        "severity": "high",
        "status": "unconfirmed",
        "triage_confidence": 0.5,
        "source_ref": "GHSA-x",
        "cve_id": "CVE-1",
        "cwe_ids": ["CWE-79"],
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=full)

    patched_client(handler)
    result = runner.invoke(cli.scout, ["finding", "abc"], env=auth_env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload.keys()) == set(cli._FINDING_DETAIL_SUMMARY_COLUMNS)
    assert payload["id"] == "abc"
    assert "description" not in payload


def test_finding_detail_returns_all_fields(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    full = {
        "id": "abc",
        "title": "T",
        "description": "desc",
        "severity": "high",
        "status": "unconfirmed",
        "triage_confidence": 0.5,
        "source_ref": "GHSA-x",
        "cve_id": "CVE-1",
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=full)

    patched_client(handler)
    result = runner.invoke(cli.scout, ["finding", "abc", "--detail"], env=auth_env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["description"] == "desc"


def test_finding_url_encodes_path_param(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    captured: dict[str, bytes] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["raw_path"] = req.url.raw_path
        return httpx.Response(
            200,
            json={
                "id": "x",
                "title": "T",
                "severity": "low",
                "status": "unconfirmed",
                "triage_confidence": None,
                "source_ref": "S",
            },
        )

    patched_client(handler)
    result = runner.invoke(cli.scout, ["finding", "../../admin"], env=auth_env)
    assert result.exit_code == 0, result.output
    # The traversal payload must be percent-encoded on the wire (raw_path).
    raw = captured["raw_path"]
    assert b"../../admin" not in raw
    assert b"..%2F..%2Fadmin" in raw


def test_triage_url_encodes_path_param(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    captured: dict[str, bytes] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["raw_path"] = req.url.raw_path
        return httpx.Response(200, json={"advisory_id": "x", "found": False})

    patched_client(handler)
    result = runner.invoke(cli.scout, ["triage", "GHSA-1/../admin"], env=auth_env)
    assert result.exit_code == 0, result.output
    assert b"/../" not in captured["raw_path"]
    assert b"%2F" in captured["raw_path"]


def test_review_url_encodes_path_param(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    captured: dict[str, bytes] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["raw_path"] = req.url.raw_path
        return httpx.Response(
            200,
            json={
                "decision": "approve",
                "finding_id": "abc",
                "workflow_run_id": "w1",
                "reviewer_id": "api:t",
                "finding_status": "confirmed_high",
                "workflow_state": "awaiting_approval",
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["review", "abc/../def", "--decision", "approve"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert b"/../" not in captured["raw_path"]
    assert b"%2F" in captured["raw_path"]


def test_request_non_json_success_body_fails(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all", headers={"Content-Type": "text/plain"})

    patched_client(handler)
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo"], env=auth_env)
    assert result.exit_code == 1
    assert "non-JSON success body" in result.output


def test_finding_non_object_body_fails(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    patched_client(handler)
    result = runner.invoke(cli.scout, ["finding", "abc"], env=auth_env)
    assert result.exit_code == 1
    assert "non-object body" in result.output


def test_finding_table_summary(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "abc",
                "title": "T",
                "severity": "high",
                "status": "unconfirmed",
                "triage_confidence": 0.5,
                "source_ref": "GHSA-x",
            },
        )

    patched_client(handler)
    result = runner.invoke(cli.scout, ["finding", "abc", "--format", "table"], env=auth_env)
    assert result.exit_code == 0, result.output
    assert "id" in result.output
    assert "abc" in result.output


# ── review ───────────────────────────────────────────────────────────────────


def test_review_posts_decision_and_context(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    sent: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent["method"] = req.method
        sent["path"] = req.url.path
        sent["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "decision": "approve",
                "finding_id": "abc",
                "workflow_run_id": "w1",
                "reviewer_id": "api:t",
                "finding_status": "confirmed_high",
                "workflow_state": "awaiting_approval",
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["review", "abc", "--decision", "approve", "--context", "looks legit"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert sent["method"] == "POST"
    assert sent["path"].endswith("/findings/abc/review")
    assert sent["body"] == {"decision": "approve", "context": "looks legit"}


def test_review_omits_empty_context(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    sent: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "decision": "reject",
                "finding_id": "abc",
                "workflow_run_id": "w1",
                "reviewer_id": "api:t",
                "finding_status": "false_positive",
                "workflow_state": "complete",
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["review", "abc", "--decision", "reject"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert sent["body"] == {"decision": "reject"}


def test_review_invalid_decision_rejected(runner: CliRunner, auth_env: dict[str, str]) -> None:
    result = runner.invoke(cli.scout, ["review", "abc", "--decision", "nope"], env=auth_env)
    assert result.exit_code != 0


# ── triage ───────────────────────────────────────────────────────────────────


def test_triage_returns_kv_in_table(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/triage/GHSA-xxxx")
        return httpx.Response(
            200,
            json={
                "advisory_id": "GHSA-xxxx",
                "found": True,
                "finding_id": "abc",
                "severity": "high",
                "ssvc_action": "act",
                "status": "unconfirmed",
                "known_status": None,
            },
        )

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["triage", "GHSA-xxxx", "--format", "table"],
        env=auth_env,
    )
    assert result.exit_code == 0, result.output
    assert "advisory_id" in result.output
    assert "GHSA-xxxx" in result.output


# ── Error paths ──────────────────────────────────────────────────────────────


def test_missing_token_exits_with_grant_hint(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.scout,
        ["findings", "--repo", "demo"],
        env={"HOME": str(tmp_path)},
    )
    assert result.exit_code == 1
    assert "No API token found" in result.output
    assert "python -m src.admin create_token" in result.output


def test_http_401_shows_auth_hint(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid or revoked token"})

    patched_client(handler)
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo"], env=auth_env)
    assert result.exit_code == 1
    assert "authentication failed" in result.output
    assert "401" in result.output


def test_http_403_shows_auth_hint(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Token lacks required scope"})

    patched_client(handler)
    result = runner.invoke(
        cli.scout,
        ["review", "abc", "--decision", "approve"],
        env=auth_env,
    )
    assert result.exit_code == 1
    assert "authentication failed" in result.output


def test_http_404_returns_not_found(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "finding not found"})

    patched_client(handler)
    result = runner.invoke(cli.scout, ["finding", "missing"], env=auth_env)
    assert result.exit_code == 1
    assert "not found" in result.output


def test_http_429_shows_retry_after(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"}, json={"detail": "rate limited"})

    patched_client(handler)
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo"], env=auth_env)
    assert result.exit_code == 1
    assert "rate limited" in result.output
    assert "30" in result.output


def test_http_500_non_json_body(
    runner: CliRunner,
    auth_env: dict[str, str],
    patched_client,
) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    patched_client(handler)
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo"], env=auth_env)
    assert result.exit_code == 1
    assert "500" in result.output
    assert "boom" in result.output


def test_network_error_surfaces(
    runner: CliRunner,
    auth_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(cli, "_make_client", lambda _b, _t: _mock_client(boom))
    result = runner.invoke(cli.scout, ["findings", "--repo", "demo"], env=auth_env)
    assert result.exit_code == 1
    assert "network error" in result.output


# ── main() entry ────────────────────────────────────────────────────────────


def test_main_returns_zero_on_help(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``--help`` makes click raise SystemExit(0) — main maps that to 0.
    monkeypatch.delenv("SCOUT_API_KEY", raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")
    code = cli.main(["--help"])
    assert code == 0


def test_main_returns_nonzero_on_missing_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SCOUT_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    code = cli.main(["findings", "--repo", "demo"])
    assert code == 1

# SPDX-License-Identifier: Apache-2.0
"""``scout`` CLI — thin authenticated client over the Security Scout HTTP API.

Subcommands talk to ``$SCOUT_API_URL`` (default ``http://localhost:8000``)
using a bearer token resolved from one of:

1. ``$SCOUT_API_KEY`` environment variable.
2. ``~/.config/scout/token`` (must be ``chmod 600``; the CLI refuses to
   read a token file that is group- or world-readable).

If neither source yields a token the CLI exits non-zero with instructions
on how to mint one via the host admin CLI (``python -m src.admin
create_token``).

Output defaults to indented JSON; ``--format table`` renders a
column-aligned text table. Every subcommand maps a single HTTP call to a
single CLI invocation, so error semantics mirror the API: 401/403 →
authentication problem, 404 → no such resource, 429 → rate limited.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import click
import httpx

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "scout" / "token"
_API_PREFIX = "/api/v1"
_REQUEST_TIMEOUT_SECONDS = 30.0
_FORMAT_CHOICES = ("json", "table")

_GRANT_HINT = (
    "No API token found.\n"
    "  • Set SCOUT_API_KEY in your environment, or\n"
    "  • Write the token to ~/.config/scout/token (chmod 600).\n"
    "Mint one on the host running Security Scout:\n"
    "  python -m src.admin create_token --name <label> --scope findings:read "
    "[--scope findings:write] --owner <U_SLACK_ID>"
)


class CliError(click.ClickException):
    """Raised for any non-usage error. Maps to exit code 1."""

    exit_code = 1

    def show(self, file: Any = None) -> None:
        stream = file or click.get_text_stream("stderr")
        click.echo(f"error: {self.message}", file=stream)


# ── Auth resolution ──────────────────────────────────────────────────────────


def _read_token_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        info = path.stat()
    except OSError as exc:
        raise CliError(f"unable to read token file {path}: {exc}") from exc

    # Permission check (skipped on platforms without POSIX modes).
    if hasattr(os, "geteuid"):
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o077:
            raise CliError(
                f"refusing to read token file {path}: permissions {oct(mode)} are too open. Run: chmod 600 {path}",
            )

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"unable to read token file {path}: {exc}") from exc

    token = raw.strip()
    return token or None


def _validate_token_shape(token: str, *, source: str) -> str:
    """Reject tokens containing characters that would corrupt the HTTP header.

    ``Authorization`` headers reject CR/LF and other control bytes. h11 would
    raise a confusing low-level error; surface a clear CLI message instead.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in token):
        raise CliError(
            f"token from {source} contains control characters; mint a new token via `python -m src.admin create_token`",
        )
    return token


def resolve_token(*, env: Mapping[str, str], token_path: Path = DEFAULT_TOKEN_PATH) -> str:
    env_token = env.get("SCOUT_API_KEY", "").strip()
    if env_token:
        return _validate_token_shape(env_token, source="SCOUT_API_KEY")
    file_token = _read_token_file(token_path)
    if file_token:
        return _validate_token_shape(file_token, source=str(token_path))
    raise CliError(_GRANT_HINT)


def resolve_base_url(env: Mapping[str, str]) -> str:
    raw = env.get("SCOUT_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
    return raw.rstrip("/")


# ── HTTP transport ───────────────────────────────────────────────────────────


def _make_client(base_url: str, token: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url + _API_PREFIX,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "scout-cli",
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def _format_http_error(response: httpx.Response) -> str:
    detail: str
    try:
        body = response.json()
    except ValueError:
        detail = (response.text or "").strip()[:500] or "(no body)"
    else:
        if isinstance(body, dict) and "detail" in body:
            raw_detail = body["detail"]
            detail = raw_detail if isinstance(raw_detail, str) else json.dumps(raw_detail)
        else:
            detail = json.dumps(body)

    if response.status_code in (401, 403):
        return (
            f"authentication failed (HTTP {response.status_code}): {detail}\n"
            "Verify SCOUT_API_KEY is set or ~/.config/scout/token is readable; "
            "the token may be revoked or missing the required scope."
        )
    if response.status_code == 404:
        return f"not found (HTTP 404): {detail}"
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        return f"rate limited (HTTP 429): retry after {retry_after}s"
    return f"request failed (HTTP {response.status_code}): {detail}"


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    json_body: Any | None = None,
) -> Any:
    try:
        response = client.request(method, path, params=params, json=json_body)
    except httpx.RequestError as exc:
        raise CliError(f"network error talking to API: {exc}") from exc

    if response.is_success:
        try:
            return response.json()
        except ValueError as exc:
            raise CliError(f"API returned non-JSON success body: {exc}") from exc

    raise CliError(_format_http_error(response))


# ── Output formatting ────────────────────────────────────────────────────────


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | dict | tuple):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _render_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "(no rows)"
    header = tuple(columns)
    body = [tuple(_cell(row.get(col)) for col in columns) for row in rows]
    grid = [header, *body]
    widths = [max(len(cell) for cell in column) for column in zip(*grid, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(fmt.format(*line) for line in grid)


def _render_kv(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "(empty)"
    key_width = max(len(k) for k in payload)
    fmt = f"{{:<{key_width}}}  {{}}"
    return "\n".join(fmt.format(key, _cell(payload[key])) for key in payload)


def _emit(payload: Any, *, fmt: str, columns: Sequence[str] | None = None) -> None:
    if fmt == "json":
        click.echo(_json_dumps(payload))
        return
    if isinstance(payload, list):
        rows = [item if isinstance(item, dict) else {"value": item} for item in payload]
        cols = list(columns) if columns is not None else _default_columns(rows)
        click.echo(_render_table(rows, cols))
        return
    if isinstance(payload, dict):
        click.echo(_render_kv(payload))
        return
    click.echo(_cell(payload))


def _default_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


# ── Argument parsing helpers ─────────────────────────────────────────────────


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_pkg_at_version(target: str) -> tuple[str, str]:
    """Split ``<pkg>[@<version>]`` while keeping the npm ``@scope/`` prefix.

    A leading ``@`` is part of the package name (``@scope/pkg``); the version
    separator is the *next* ``@`` after it. For non-scoped names the first
    ``@`` is the separator. Whitespace is trimmed from both halves.
    """
    cleaned = target.strip()
    if cleaned.startswith("@"):
        body = cleaned[1:]
        if "@" in body:
            name, _, version = body.partition("@")
            return f"@{name}".strip(), version.strip()
        return cleaned, ""
    if "@" not in cleaned:
        return cleaned, ""
    pkg, _, version = cleaned.partition("@")
    return pkg.strip(), version.strip()


def _dedupe_findings(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        finding_id = str(item.get("id", ""))
        if finding_id and finding_id in seen:
            continue
        if finding_id:
            seen.add(finding_id)
        out.append(dict(item))
    return out


# ── Click commands ───────────────────────────────────────────────────────────


@click.group(
    name="scout",
    help="Authenticated CLI for the Security Scout HTTP API.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def scout(ctx: click.Context) -> None:
    env = os.environ
    token = resolve_token(env=env)
    base_url = resolve_base_url(env)
    ctx.obj = {"token": token, "base_url": base_url}


def _client_from_ctx(ctx: click.Context) -> httpx.Client:
    obj = ctx.obj or {}
    return _make_client(obj["base_url"], obj["token"])


_FORMAT_OPTION = click.option(
    "--format",
    "output_format",
    type=click.Choice(_FORMAT_CHOICES, case_sensitive=False),
    default="json",
    show_default=True,
    help="Output format.",
)


_FINDING_LIST_COLUMNS = ("id", "title", "severity", "ssvc_action", "status", "triage_confidence")
_FINDING_DETAIL_SUMMARY_COLUMNS = ("id", "title", "severity", "status", "triage_confidence", "source_ref")
_TRIAGE_COLUMNS = ("advisory_id", "found", "finding_id", "severity", "ssvc_action", "status")
_DEP_ADVISORY_COLUMNS = ("id", "title", "severity", "ssvc_action", "source_ref")


@scout.command("findings", help="List findings for a repository.")
@click.option("--repo", required=True, help="Repository config name (matches repos.yaml).")
@click.option(
    "--severity",
    "severity_csv",
    default=None,
    help="Comma-separated severities (e.g. critical,high). Empty = all.",
)
@click.option(
    "--status",
    "status_filter",
    default=None,
    help="FindingStatus filter (e.g. confirmed_high).",
)
@click.option("--limit", default=50, show_default=True, type=click.IntRange(min=1, max=200))
@_FORMAT_OPTION
@click.pass_context
def cmd_findings(
    ctx: click.Context,
    repo: str,
    severity_csv: str | None,
    status_filter: str | None,
    limit: int,
    output_format: str,
) -> None:
    severities = _split_csv(severity_csv)
    with _client_from_ctx(ctx) as client:
        if not severities:
            rows = _request(
                client,
                "GET",
                "/findings",
                params=_drop_none(
                    {"repo": repo, "finding_status": status_filter, "limit": limit},
                ),
            )
            _emit(rows, fmt=output_format, columns=_FINDING_LIST_COLUMNS)
            return

        merged: list[dict[str, Any]] = []
        for sev in severities:
            batch = _request(
                client,
                "GET",
                "/findings",
                params=_drop_none(
                    {
                        "repo": repo,
                        "severity": sev,
                        "finding_status": status_filter,
                        "limit": limit,
                    },
                ),
            )
            if isinstance(batch, list):
                merged.extend(batch)
        merged_unique = _dedupe_findings(merged)
        _emit(merged_unique, fmt=output_format, columns=_FINDING_LIST_COLUMNS)


def _drop_none(params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


@scout.command("check", help="Check whether <pkg>@<version> is affected by known advisories.")
@click.argument("target", required=True)
@click.option("--ecosystem", default="", help="Package ecosystem (npm, pypi, maven, …).")
@_FORMAT_OPTION
@click.pass_context
def cmd_check(ctx: click.Context, target: str, ecosystem: str, output_format: str) -> None:
    package, version = _parse_pkg_at_version(target)
    if not package:
        raise CliError("missing package name; usage: scout check <pkg>[@<version>]")
    with _client_from_ctx(ctx) as client:
        payload = _request(
            client,
            "GET",
            "/check",
            params={"package": package, "version": version, "ecosystem": ecosystem},
        )

    if output_format == "json":
        _emit(payload, fmt=output_format)
        return

    if not isinstance(payload, dict):
        click.echo(_cell(payload))
        return

    header_keys = ("package", "version", "ecosystem", "advisory_count")
    summary = {key: payload.get(key) for key in header_keys}
    click.echo(_render_kv(summary))
    advisories = payload.get("advisories") or []
    if not isinstance(advisories, list) or not advisories:
        click.echo("\n(no advisories)")
        return
    click.echo("\nadvisories:")
    click.echo(_render_table(advisories, _DEP_ADVISORY_COLUMNS))


@scout.command("finding", help="Show a single finding by UUID.")
@click.argument("finding_id", required=True)
@click.option(
    "--detail",
    is_flag=True,
    default=False,
    help="Show all fields; otherwise prints a summary subset.",
)
@_FORMAT_OPTION
@click.pass_context
def cmd_finding(
    ctx: click.Context,
    finding_id: str,
    detail: bool,
    output_format: str,
) -> None:
    encoded = quote(finding_id, safe="")
    with _client_from_ctx(ctx) as client:
        payload = _request(client, "GET", f"/findings/{encoded}")
    if not isinstance(payload, dict):
        raise CliError("API returned non-object body for finding")

    if detail:
        _emit(payload, fmt=output_format)
        return

    if output_format == "json":
        subset = {key: payload.get(key) for key in _FINDING_DETAIL_SUMMARY_COLUMNS}
        _emit(subset, fmt=output_format)
        return
    summary = {key: payload.get(key) for key in _FINDING_DETAIL_SUMMARY_COLUMNS}
    click.echo(_render_kv(summary))


@scout.command("review", help="Apply a review decision to a finding.")
@click.argument("finding_id", required=True)
@click.option(
    "--decision",
    type=click.Choice(["approve", "reject", "escalate"], case_sensitive=False),
    required=True,
    help="Decision to record.",
)
@click.option("--context", "context_text", default=None, help="Optional reviewer context.")
@_FORMAT_OPTION
@click.pass_context
def cmd_review(
    ctx: click.Context,
    finding_id: str,
    decision: str,
    context_text: str | None,
    output_format: str,
) -> None:
    body: dict[str, Any] = {"decision": decision.lower()}
    if context_text:
        body["context"] = context_text
    encoded = quote(finding_id, safe="")
    with _client_from_ctx(ctx) as client:
        payload = _request(client, "POST", f"/findings/{encoded}/review", json_body=body)
    _emit(payload, fmt=output_format)


@scout.command("triage", help="Show triage status for an advisory.")
@click.argument("advisory_id", required=True)
@_FORMAT_OPTION
@click.pass_context
def cmd_triage(ctx: click.Context, advisory_id: str, output_format: str) -> None:
    encoded = quote(advisory_id, safe="")
    with _client_from_ctx(ctx) as client:
        payload = _request(client, "GET", f"/triage/{encoded}")
    if output_format == "json" or not isinstance(payload, dict):
        _emit(payload, fmt=output_format, columns=_TRIAGE_COLUMNS)
        return
    click.echo(_render_kv({key: payload.get(key) for key in _TRIAGE_COLUMNS}))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        scout.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    except click.UsageError as exc:
        exc.show()
        return exc.exit_code
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except SystemExit as exc:  # pragma: no cover - click sometimes raises directly
        return int(exc.code) if isinstance(exc.code, int) else 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())


__all__ = [
    "DEFAULT_API_URL",
    "DEFAULT_TOKEN_PATH",
    "CliError",
    "main",
    "resolve_base_url",
    "resolve_token",
    "scout",
]

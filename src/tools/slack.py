from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Literal, Self, cast

import httpx
import structlog
from pydantic import BaseModel, ConfigDict
from slack_sdk.models.blocks import ContextBlock, DividerBlock, HeaderBlock, SectionBlock
from slack_sdk.models.blocks.basic_components import MarkdownTextObject, PlainTextObject

from exceptions import SecurityScoutError

__all__ = [
    "DedupMatchInfo",
    "FindingReportPayload",
    "SlackAPIError",
    "SlackClient",
    "SlackMalformedResponseError",
    "SlackPostMessageResult",
    "build_finding_blocks",
    "fallback_notification_text",
]

_LOG = structlog.get_logger(__name__)

_MAX_HEADER_CHARS = 150
_MAX_DESCRIPTION_EXCERPT = 300
_MAX_EVIDENCE_EXCERPT = 500
_MAX_REPRODUCTION_EXCERPT = 2000

_SLACK_TRANSIENT_ERRORS = frozenset(
    {
        "rate_limited",
        "service_unavailable",
        "internal_error",
        "request_timeout",
        "ekm_access_denied",
    }
)


class SlackAPIError(SecurityScoutError):
    def __init__(
        self,
        message: str | None = None,
        *,
        is_transient: bool,
        http_status: int | None = None,
        slack_error_code: str | None = None,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=is_transient,
            is_resource_error=False,
        )
        self.http_status = http_status
        self.slack_error_code = slack_error_code

    @classmethod
    def from_status(cls, status: int, message: str) -> SlackAPIError:
        transient = status in (408, 425, 429, 500, 502, 503, 504)
        return cls(message, is_transient=transient, http_status=status)

    @classmethod
    def from_slack_error(cls, code: str, message: str | None = None) -> SlackAPIError:
        transient = code in _SLACK_TRANSIENT_ERRORS
        text = message if message else f"Slack API error: {code}"
        return cls(text, is_transient=transient, slack_error_code=code)


class SlackMalformedResponseError(SecurityScoutError):
    """HTTP 200 with an unexpected JSON shape from Slack."""

    def __init__(
        self,
        message: str | None = None,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=False,
            is_resource_error=False,
        )


class DedupMatchInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: Literal[1, 2]
    tracker_name: str
    match_url: str | None = None
    duplicate_of: str | None = None


class FindingReportPayload(BaseModel):
    """Fields required for HITL visibility (architecture § Human Approval Gate)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: uuid.UUID
    title: str
    severity: str
    ssvc_action: str | None
    confidence: float | None
    source_url: str
    affected_versions: str | None = None
    cve_ids: tuple[str, ...] = ()
    cwe_ids: tuple[str, ...] = ()
    cvss_score: float | None = None
    cvss_vector: str | None = None
    description_excerpt: str | None = None
    evidence_excerpt: str | None = None
    poc_type_label: str | None = None
    reproduction_steps: str | None = None
    dedup: DedupMatchInfo | None = None


class SlackPostMessageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: str
    message_ts: str


def escape_slack_mrkdwn(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"
    return text[: max_chars - 1] + "…"


_WS_RE = re.compile(r"\s+")


def _plain_single_line(text: str) -> str:
    """Collapse whitespace for plain-text / header fields (no literal newlines)."""
    return _WS_RE.sub(" ", text.strip())


def _slack_link_url(url: str) -> str:
    """Escape characters that break Slack ``<url|label>`` link syntax (``|``, ``<``, ``>``)."""

    return url.replace("|", "%7C").replace(">", "%3E").replace("<", "%3C")


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "—"
    if not math.isfinite(confidence):
        return "—"
    return f"{confidence:.2f}"


def fallback_notification_text(report: FindingReportPayload) -> str:
    """Plain-text fallback for notifications and accessibility (required with blocks).

    This is **not** mrkdwn — do not HTML-escape ``&`` / ``<`` / ``>`` or notifications show ``&amp;`` literally.
    """

    title = _truncate(_plain_single_line(report.title), 120)
    sev = _plain_single_line(report.severity)
    return f"[{sev.upper()}] {title} — {report.source_url}"


def build_finding_blocks(
    report: FindingReportPayload,
    *,
    workflow_run_id: uuid.UUID | None = None,
) -> list[dict[str, Any]]:
    blocks: list[Any] = []
    sev_upper = _plain_single_line(report.severity).upper()
    header_plain = _truncate(f"[{sev_upper}] {_plain_single_line(report.title)}", _MAX_HEADER_CHARS)
    blocks.append(HeaderBlock(text=PlainTextObject(text=header_plain)))

    blocks.append(DividerBlock())

    cve_line = ", ".join(report.cve_ids) if report.cve_ids else "—"
    cwe_line = ", ".join(report.cwe_ids) if report.cwe_ids else "—"
    cvss_line = "—"
    if report.cvss_score is not None:
        cvss_line = f"{report.cvss_score:g}"
        if report.cvss_vector:
            vec_safe = escape_slack_mrkdwn(report.cvss_vector).replace("`", "'")
            cvss_line = f"{cvss_line} (`{vec_safe}`)"

    ssvc = escape_slack_mrkdwn(report.ssvc_action) if report.ssvc_action else "—"
    conf = _format_confidence(report.confidence)

    blocks.append(
        SectionBlock(
            fields=[
                MarkdownTextObject(text=f"*Severity*\n{escape_slack_mrkdwn(report.severity)}"),
                MarkdownTextObject(text=f"*CVSS*\n{cvss_line}"),
                MarkdownTextObject(text=f"*SSVC action*\n{ssvc}"),
                MarkdownTextObject(text=f"*Confidence*\n{conf}"),
                MarkdownTextObject(text=f"*CVEs*\n{escape_slack_mrkdwn(cve_line)}"),
                MarkdownTextObject(text=f"*CWEs*\n{escape_slack_mrkdwn(cwe_line)}"),
            ]
        )
    )

    source = report.source_url
    source_link = _slack_link_url(source)
    blocks.append(
        SectionBlock(
            text=MarkdownTextObject(
                text=f"*Source*\n<{source_link}|Open in GitHub>",
            )
        )
    )

    ver = report.affected_versions
    ver_line = escape_slack_mrkdwn(ver) if ver else "— (not specified)"
    blocks.append(SectionBlock(text=MarkdownTextObject(text=f"*Affected versions*\n{ver_line}")))

    desc_raw = report.description_excerpt
    desc = _truncate(escape_slack_mrkdwn(desc_raw), _MAX_DESCRIPTION_EXCERPT) if desc_raw else "—"
    blocks.append(SectionBlock(text=MarkdownTextObject(text=f"*Description*\n{desc}")))

    ev_raw = report.evidence_excerpt
    ev = (
        _truncate(escape_slack_mrkdwn(ev_raw), _MAX_EVIDENCE_EXCERPT)
        if ev_raw
        else "— (no PoC execution yet — advisory triage only)"
    )
    blocks.append(SectionBlock(text=MarkdownTextObject(text=f"*Evidence excerpt*\n{ev}")))

    poc = report.poc_type_label
    poc_line = escape_slack_mrkdwn(poc) if poc else "— (not yet classified)"
    blocks.append(SectionBlock(text=MarkdownTextObject(text=f"*PoC type*\n{poc_line}")))

    repro_raw = report.reproduction_steps
    repro = _truncate(escape_slack_mrkdwn(repro_raw), _MAX_REPRODUCTION_EXCERPT) if repro_raw else "—"
    blocks.append(SectionBlock(text=MarkdownTextObject(text=f"*Reproduction steps*\n{repro}")))

    if report.dedup is not None:
        d = report.dedup
        lines = [
            f"*Known vulnerability match* (tier {d.tier})",
            f"*Tracker:* {escape_slack_mrkdwn(d.tracker_name)}",
        ]
        if d.duplicate_of:
            lines.append(f"*Match:* {escape_slack_mrkdwn(d.duplicate_of)}")
        if d.match_url:
            u = _slack_link_url(d.match_url)
            lines.append(f"*Link:* <{u}|View match>")
        blocks.append(SectionBlock(text=MarkdownTextObject(text="\n".join(lines))))

    blocks.append(DividerBlock())

    ctx = f"Finding `{report.finding_id}`"
    if workflow_run_id is not None:
        ctx += f" · workflow run `{workflow_run_id}`"
    ctx += f" · <{source_link}|source>"
    blocks.append(ContextBlock(elements=[MarkdownTextObject(text=ctx)]))

    return [cast(dict[str, Any], b.to_dict()) for b in blocks]


_DEFAULT_TIMEOUT = httpx.Timeout(30.0)


def _as_slack_json(
    response: httpx.Response,
    *,
    finding_id: str | None,
    workflow_run_id: uuid.UUID | str | None,
) -> dict[str, Any]:
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        msg = "Slack API returned non-JSON body"
        raise SlackMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        ) from e
    if not isinstance(data, dict):
        msg = "Slack API JSON root must be an object"
        raise SlackMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    return data


class SlackClient:
    """Posts Block Kit messages via ``chat.postMessage``.

    If you pass a custom ``httpx.AsyncClient``, set ``base_url`` to ``https://slack.com/api``
    (or equivalent) so ``POST /chat.postMessage`` resolves correctly.
    """

    def __init__(
        self,
        bot_token: str,
        *,
        base_url: str = "https://slack.com/api",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not bot_token:
            msg = "Slack bot token is empty"
            raise ValueError(msg)
        self._token = bot_token
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> Self:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=_DEFAULT_TIMEOUT)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None if self._owns_client else self._client

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "SlackClient must be used as a context manager or constructed with a client"
            raise RuntimeError(msg)
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def send_finding(
        self,
        channel: str,
        report: FindingReportPayload,
        *,
        workflow_run_id: uuid.UUID | None = None,
    ) -> SlackPostMessageResult:
        if not channel:
            msg = "Slack channel is empty"
            raise ValueError(msg)

        finding_id = str(report.finding_id)
        body: dict[str, Any] = {
            "channel": channel,
            "text": fallback_notification_text(report),
            "blocks": build_finding_blocks(report, workflow_run_id=workflow_run_id),
            "unfurl_links": False,
            "unfurl_media": False,
        }

        client = self._client_or_raise()
        response = await client.post(
            "/chat.postMessage",
            json=body,
            headers=self._auth_headers(),
        )

        if not response.is_success:
            err = SlackAPIError.from_status(
                response.status_code,
                response.reason_phrase or f"HTTP {response.status_code}",
            )
            _LOG.warning(
                "slack_api_http_error",
                metric_name="api_error_total",
                api="slack",
                http_status=err.http_status,
                finding_id=finding_id,
                workflow_run_id=str(workflow_run_id) if workflow_run_id else None,
            )
            raise err

        payload = _as_slack_json(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        if not payload.get("ok"):
            code = payload.get("error")
            code_str = code if isinstance(code, str) else "unknown"
            warn = payload.get("warning")
            if isinstance(warn, str):
                _LOG.warning("slack_api_warning", warning=warn, finding_id=finding_id)
            err = SlackAPIError.from_slack_error(code_str, f"Slack chat.postMessage failed: {code_str}")
            _LOG.warning(
                "slack_api_error",
                metric_name="api_error_total",
                api="slack",
                slack_error=code_str,
                finding_id=finding_id,
                workflow_run_id=str(workflow_run_id) if workflow_run_id else None,
            )
            raise err

        ch = payload.get("channel")
        ts = payload.get("ts")
        if not isinstance(ch, str) or not isinstance(ts, str):
            msg = "Slack chat.postMessage response missing channel or ts"
            raise SlackMalformedResponseError(
                msg,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
            )

        _LOG.info(
            "slack_finding_posted",
            metric_name="slack_finding_posted",
            channel=ch,
            finding_id=finding_id,
            workflow_run_id=str(workflow_run_id) if workflow_run_id else None,
        )
        return SlackPostMessageResult(channel=ch, message_ts=ts)

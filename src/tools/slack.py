from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Literal, NoReturn, Self, cast

import httpx
import structlog
from pydantic import BaseModel, ConfigDict
from slack_sdk.models.blocks import ActionsBlock, ContextBlock, DividerBlock, HeaderBlock, SectionBlock
from slack_sdk.models.blocks.basic_components import MarkdownTextObject, PlainTextObject
from slack_sdk.models.blocks.block_elements import ButtonElement

from exceptions import SecurityScoutError
from models import Finding, KnownStatus

__all__ = [
    "ACTION_ID_APPROVE",
    "ACTION_ID_DEDUP_CONFIRM",
    "ACTION_ID_DEDUP_NEW_INSTANCE",
    "ACTION_ID_DEDUP_REOPEN",
    "ACTION_ID_DEDUP_RESOLVED",
    "ACTION_ID_ESCALATE",
    "ACTION_ID_REJECT",
    "ACTION_ID_RISK_REEVALUATE",
    "ACTION_ID_RISK_STILL_ACCEPTED",
    "ApprovalButtonContext",
    "DedupMatchInfo",
    "FindingReportPayload",
    "SlackAPIError",
    "SlackClient",
    "SlackMalformedResponseError",
    "SlackPostMessageResult",
    "build_finding_blocks",
    "fallback_notification_text",
    "finding_to_report_payload",
]

ACTION_ID_APPROVE = "security_scout:approve"
ACTION_ID_REJECT = "security_scout:reject"
ACTION_ID_ESCALATE = "security_scout:escalate"

ACTION_ID_DEDUP_CONFIRM = "security_scout:dedup_confirm"
ACTION_ID_DEDUP_NEW_INSTANCE = "security_scout:dedup_new_instance"
ACTION_ID_DEDUP_REOPEN = "security_scout:dedup_reopen"
ACTION_ID_DEDUP_RESOLVED = "security_scout:dedup_resolved"

ACTION_ID_RISK_STILL_ACCEPTED = "security_scout:risk_still_accepted"
ACTION_ID_RISK_REEVALUATE = "security_scout:risk_reevaluate"

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
    # ``True`` when the match is a previously-accepted-risk finding within TTL.
    # Renders the "Previously Accepted Risk" variant with Still Accepted / Re-evaluate buttons
    # in place of the standard Approve / Reject / Escalate block.
    is_accepted_risk: bool = False


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


class ApprovalButtonContext(BaseModel):
    """Encodes the identifiers Slack must echo back from a button click.

    The encoded string is carried as a button ``value`` (Slack caps this at 2000
    chars, which is plenty for three UUIDs and a repo name).  Slack signs the
    outer request, so the value is only ever set by us; the approval handler
    still validates every field against the database before acting on it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: uuid.UUID
    workflow_run_id: uuid.UUID
    repo_name: str

    def encode(self) -> str:
        if "|" in self.repo_name:
            msg = "repo_name must not contain '|'"
            raise ValueError(msg)
        return f"{self.finding_id}|{self.workflow_run_id}|{self.repo_name}"

    @classmethod
    def decode(cls, value: str) -> ApprovalButtonContext:
        parts = value.split("|")
        if len(parts) != 3:
            msg = "expected 3 pipe-separated fields"
            raise ValueError(msg)
        fid_raw, run_raw, repo_name = parts
        if not repo_name:
            msg = "repo_name is empty"
            raise ValueError(msg)
        return cls(
            finding_id=uuid.UUID(fid_raw),
            workflow_run_id=uuid.UUID(run_raw),
            repo_name=repo_name,
        )


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


def finding_to_report_payload(finding: Finding) -> FindingReportPayload:
    cve_ids: tuple[str, ...] = (finding.cve_id,) if finding.cve_id else ()
    cwe_ids = tuple(finding.cwe_ids) if finding.cwe_ids else ()
    dedup: DedupMatchInfo | None = None
    if finding.duplicate_tracker and finding.duplicate_of:
        dedup = DedupMatchInfo(
            tier=1,
            tracker_name=finding.duplicate_tracker,
            match_url=finding.duplicate_url,
            duplicate_of=finding.duplicate_of,
            is_accepted_risk=finding.known_status == KnownStatus.known_accepted_risk,
        )
    desc_excerpt: str | None = None
    if finding.description:
        desc_excerpt = _truncate(_plain_single_line(finding.description), _MAX_DESCRIPTION_EXCERPT)
    ev_excerpt: str | None = None
    if finding.evidence:
        try:
            raw = json.dumps(finding.evidence, ensure_ascii=False)
        except TypeError:
            raw = None
        except ValueError:
            raw = None
        if raw is not None:
            ev_excerpt = _truncate(raw, _MAX_EVIDENCE_EXCERPT)
    return FindingReportPayload(
        finding_id=finding.id,
        title=finding.title,
        severity=finding.severity.value,
        ssvc_action=finding.ssvc_action.value if finding.ssvc_action else None,
        confidence=finding.triage_confidence,
        source_url=finding.source_ref,
        affected_versions=None,
        cve_ids=cve_ids,
        cwe_ids=cwe_ids,
        cvss_score=finding.cvss_score,
        cvss_vector=finding.cvss_vector,
        description_excerpt=desc_excerpt,
        evidence_excerpt=ev_excerpt,
        poc_type_label=None,
        reproduction_steps=finding.reproduction,
        dedup=dedup,
    )


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


def _cvss_line_for_report(report: FindingReportPayload) -> str:
    if report.cvss_score is None:
        return "—"
    line = f"{report.cvss_score:g}"
    if not report.cvss_vector:
        return line
    vec_safe = escape_slack_mrkdwn(report.cvss_vector).replace("`", "'")
    return f"{line} (`{vec_safe}`)"


def _dedup_section_block(dedup: DedupMatchInfo) -> SectionBlock:
    if dedup.is_accepted_risk:
        lines = [
            "*Previously Accepted Risk*",
            f"*Tracker:* {escape_slack_mrkdwn(dedup.tracker_name)}",
        ]
    else:
        lines = [
            f"*Known vulnerability match* (tier {dedup.tier})",
            f"*Tracker:* {escape_slack_mrkdwn(dedup.tracker_name)}",
        ]
    if dedup.duplicate_of:
        lines.append(f"*Match:* {escape_slack_mrkdwn(dedup.duplicate_of)}")
    if dedup.match_url:
        u = _slack_link_url(dedup.match_url)
        lines.append(f"*Link:* <{u}|View match>")
    return SectionBlock(text=MarkdownTextObject(text="\n".join(lines)))


def _build_dedup_actions_block(ctx: ApprovalButtonContext) -> ActionsBlock:
    encoded = ctx.encode()
    return ActionsBlock(
        block_id="security_scout_dedup_actions",
        elements=[
            ButtonElement(
                text=PlainTextObject(text="Confirm Duplicate"),
                action_id=ACTION_ID_DEDUP_CONFIRM,
                value=encoded,
            ),
            ButtonElement(
                text=PlainTextObject(text="New Instance"),
                action_id=ACTION_ID_DEDUP_NEW_INSTANCE,
                value=encoded,
            ),
            ButtonElement(
                text=PlainTextObject(text="Reopen"),
                action_id=ACTION_ID_DEDUP_REOPEN,
                value=encoded,
            ),
            ButtonElement(
                text=PlainTextObject(text="Confirm Resolved"),
                action_id=ACTION_ID_DEDUP_RESOLVED,
                value=encoded,
            ),
        ],
    )


def _build_accepted_risk_actions_block(ctx: ApprovalButtonContext) -> ActionsBlock:
    encoded = ctx.encode()
    return ActionsBlock(
        block_id="security_scout_risk_actions",
        elements=[
            ButtonElement(
                text=PlainTextObject(text="Still Accepted"),
                action_id=ACTION_ID_RISK_STILL_ACCEPTED,
                style="primary",
                value=encoded,
            ),
            ButtonElement(
                text=PlainTextObject(text="Re-evaluate"),
                action_id=ACTION_ID_RISK_REEVALUATE,
                style="danger",
                value=encoded,
            ),
        ],
    )


def _footer_context_text(
    report: FindingReportPayload,
    source_link: str,
    workflow_run_id: uuid.UUID | None,
) -> str:
    ctx = f"Finding `{report.finding_id}`"
    if workflow_run_id is not None:
        ctx += f" · workflow run `{workflow_run_id}`"
    return f"{ctx} · <{source_link}|source>"


def _build_approval_actions_block(ctx: ApprovalButtonContext) -> ActionsBlock:
    encoded = ctx.encode()
    return ActionsBlock(
        block_id="security_scout_actions",
        elements=[
            ButtonElement(
                text=PlainTextObject(text="Approve"),
                action_id=ACTION_ID_APPROVE,
                style="primary",
                value=encoded,
            ),
            ButtonElement(
                text=PlainTextObject(text="Reject"),
                action_id=ACTION_ID_REJECT,
                style="danger",
                value=encoded,
            ),
            ButtonElement(
                text=PlainTextObject(text="Escalate"),
                action_id=ACTION_ID_ESCALATE,
                value=encoded,
            ),
        ],
    )


def _informational_context_block() -> ContextBlock:
    return ContextBlock(
        elements=[MarkdownTextObject(text="_Informational — no action required_")],
    )


def build_finding_blocks(
    report: FindingReportPayload,
    *,
    workflow_run_id: uuid.UUID | None = None,
    approval_context: ApprovalButtonContext | None = None,
    informational: bool = False,
) -> list[dict[str, Any]]:
    blocks: list[Any] = []
    sev_upper = _plain_single_line(report.severity).upper()
    header_plain = _truncate(f"[{sev_upper}] {_plain_single_line(report.title)}", _MAX_HEADER_CHARS)
    blocks.append(HeaderBlock(text=PlainTextObject(text=header_plain)))

    if informational:
        blocks.append(_informational_context_block())

    blocks.append(DividerBlock())

    cve_line = ", ".join(report.cve_ids) if report.cve_ids else "—"
    cwe_line = ", ".join(report.cwe_ids) if report.cwe_ids else "—"
    cvss_line = _cvss_line_for_report(report)

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
        blocks.append(_dedup_section_block(report.dedup))
        if approval_context is not None and not report.dedup.is_accepted_risk:
            blocks.append(_build_dedup_actions_block(approval_context))

    blocks.append(DividerBlock())

    ctx = _footer_context_text(report, source_link, workflow_run_id)
    blocks.append(ContextBlock(elements=[MarkdownTextObject(text=ctx)]))

    if approval_context is not None:
        if report.dedup is not None and report.dedup.is_accepted_risk:
            blocks.append(_build_accepted_risk_actions_block(approval_context))
        else:
            blocks.append(_build_approval_actions_block(approval_context))

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


def _raise_slack_chat_post_http_error(
    response: httpx.Response,
    *,
    finding_id: str,
    workflow_run_id: uuid.UUID | None,
) -> NoReturn:
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


def _maybe_log_slack_chat_post_warning(
    payload: dict[str, Any],
    *,
    finding_id: str,
    log_slack_api_warning: bool,
) -> None:
    if not log_slack_api_warning:
        return
    warn = payload.get("warning")
    if isinstance(warn, str):
        _LOG.warning("slack_api_warning", warning=warn, finding_id=finding_id)


def _raise_slack_chat_post_not_ok(
    payload: dict[str, Any],
    *,
    finding_id: str,
    workflow_run_id: uuid.UUID | None,
    log_slack_api_warning: bool,
) -> NoReturn:
    _maybe_log_slack_chat_post_warning(
        payload,
        finding_id=finding_id,
        log_slack_api_warning=log_slack_api_warning,
    )
    code = payload.get("error")
    code_str = code if isinstance(code, str) else "unknown"
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


def _slack_chat_post_channel_ts_or_raise(
    payload: dict[str, Any],
    *,
    finding_id: str,
    workflow_run_id: uuid.UUID | None,
) -> tuple[str, str]:
    ch = payload.get("channel")
    ts = payload.get("ts")
    if not isinstance(ch, str) or not isinstance(ts, str):
        msg = "Slack chat.postMessage response missing channel or ts"
        raise SlackMalformedResponseError(
            msg,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
    return ch, ts


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

    def _chat_post_message_response_to_result(
        self,
        response: httpx.Response,
        *,
        finding_id: str,
        workflow_run_id: uuid.UUID | None,
        success_log_event: str,
        success_metric_name: str,
        log_slack_api_warning: bool = True,
    ) -> SlackPostMessageResult:
        if not response.is_success:
            _raise_slack_chat_post_http_error(
                response,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
            )
        payload = _as_slack_json(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        if not payload.get("ok"):
            _raise_slack_chat_post_not_ok(
                payload,
                finding_id=finding_id,
                workflow_run_id=workflow_run_id,
                log_slack_api_warning=log_slack_api_warning,
            )
        ch, ts = _slack_chat_post_channel_ts_or_raise(
            payload,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
        )
        _LOG.info(
            success_log_event,
            metric_name=success_metric_name,
            channel=ch,
            finding_id=finding_id,
            workflow_run_id=str(workflow_run_id) if workflow_run_id else None,
        )
        return SlackPostMessageResult(channel=ch, message_ts=ts)

    async def _post_finding_message(
        self,
        channel: str,
        report: FindingReportPayload,
        *,
        workflow_run_id: uuid.UUID | None,
        blocks: list[dict[str, Any]],
    ) -> SlackPostMessageResult:
        if not channel:
            msg = "Slack channel is empty"
            raise ValueError(msg)

        finding_id = str(report.finding_id)
        body: dict[str, Any] = {
            "channel": channel,
            "text": fallback_notification_text(report),
            "blocks": blocks,
            "unfurl_links": False,
            "unfurl_media": False,
        }

        client = self._client_or_raise()
        response = await client.post(
            "/chat.postMessage",
            json=body,
            headers=self._auth_headers(),
        )
        return self._chat_post_message_response_to_result(
            response,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            success_log_event="slack_finding_posted",
            success_metric_name="slack_finding_posted",
        )

    async def send_finding(
        self,
        channel: str,
        report: FindingReportPayload,
        *,
        workflow_run_id: uuid.UUID | None = None,
        informational: bool = False,
    ) -> SlackPostMessageResult:
        blocks = build_finding_blocks(
            report,
            workflow_run_id=workflow_run_id,
            informational=informational,
        )
        return await self._post_finding_message(
            channel,
            report,
            workflow_run_id=workflow_run_id,
            blocks=blocks,
        )

    async def send_finding_for_approval(
        self,
        channel: str,
        report: FindingReportPayload,
        *,
        workflow_run_id: uuid.UUID,
        approval_context: ApprovalButtonContext,
    ) -> SlackPostMessageResult:
        blocks = build_finding_blocks(
            report,
            workflow_run_id=workflow_run_id,
            approval_context=approval_context,
        )
        return await self._post_finding_message(
            channel,
            report,
            workflow_run_id=workflow_run_id,
            blocks=blocks,
        )

    async def post_thread_reply(
        self,
        channel: str,
        *,
        thread_ts: str,
        text: str,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | None = None,
    ) -> SlackPostMessageResult:
        if not channel:
            msg = "Slack channel is empty"
            raise ValueError(msg)
        if not thread_ts:
            msg = "thread_ts is empty"
            raise ValueError(msg)
        body: dict[str, Any] = {
            "channel": channel,
            "thread_ts": thread_ts,
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        client = self._client_or_raise()
        response = await client.post(
            "/chat.postMessage",
            json=body,
            headers=self._auth_headers(),
        )
        return self._chat_post_message_response_to_result(
            response,
            finding_id=finding_id or "none",
            workflow_run_id=workflow_run_id,
            success_log_event="slack_thread_reply_posted",
            success_metric_name="slack_thread_reply_posted",
        )

    async def send_dm(
        self,
        user_id: str,
        *,
        text: str,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | None = None,
    ) -> SlackPostMessageResult:
        if not user_id:
            msg = "Slack user id is empty"
            raise ValueError(msg)
        body: dict[str, Any] = {
            "channel": user_id,
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        client = self._client_or_raise()
        response = await client.post(
            "/chat.postMessage",
            json=body,
            headers=self._auth_headers(),
        )
        return self._chat_post_message_response_to_result(
            response,
            finding_id=finding_id or "none",
            workflow_run_id=workflow_run_id,
            success_log_event="slack_dm_posted",
            success_metric_name="slack_dm_posted",
        )

    async def notify_workflow_error(
        self,
        channel: str,
        *,
        title: str,
        detail: str,
        workflow_run_id: uuid.UUID | None = None,
        finding_id: str | None = None,
    ) -> SlackPostMessageResult:
        if not channel:
            msg = "Slack channel is empty"
            raise ValueError(msg)
        detail_safe = _truncate(_plain_single_line(detail), 2800)
        title_plain = _truncate(_plain_single_line(title), 200)
        parts: list[str] = [f"*Workflow error:* {escape_slack_mrkdwn(title_plain)}"]
        parts.append(escape_slack_mrkdwn(detail_safe))
        if workflow_run_id is not None:
            parts.append(f"*Workflow run:* `{workflow_run_id}`")
        if finding_id is not None:
            parts.append(f"*Finding:* `{finding_id}`")
        body: dict[str, Any] = {
            "channel": channel,
            "text": f"Workflow error: {title_plain}",
            "blocks": [
                cast(
                    dict[str, Any], HeaderBlock(text=PlainTextObject(text="Security Scout — workflow error")).to_dict()
                ),
                cast(
                    dict[str, Any],
                    SectionBlock(text=MarkdownTextObject(text="\n".join(parts))).to_dict(),
                ),
            ],
            "unfurl_links": False,
            "unfurl_media": False,
        }
        client = self._client_or_raise()
        response = await client.post(
            "/chat.postMessage",
            json=body,
            headers=self._auth_headers(),
        )
        fid = finding_id or "none"
        return self._chat_post_message_response_to_result(
            response,
            finding_id=fid,
            workflow_run_id=workflow_run_id,
            success_log_event="slack_workflow_error_posted",
            success_metric_name="slack_workflow_error_posted",
            log_slack_api_warning=False,
        )

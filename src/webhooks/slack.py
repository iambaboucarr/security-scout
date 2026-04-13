"""Slack interactive component webhook.

Mirrors the three-requirement verification used for GitHub webhooks:

* read the raw request body before JSON parsing,
* verify the ``X-Slack-Signature`` header with :func:`hmac.compare_digest`,
* reject deliveries whose ``X-Slack-Request-Timestamp`` is older than 5 minutes.

The handler parses the URL-encoded ``payload`` field (Slack's interactive
component transport) and dispatches approve / reject / escalate actions to the
approval handler. Signature verification uses the *raw* request bytes; any
decoding/parsing happens only after the signature has been validated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qs

import structlog
from fastapi import APIRouter, HTTPException, Request, Response

from agents.approval import ApprovalContext, handle_slack_approval
from agents.dedup import DedupContext, handle_slack_dedup_decision
from config import AppConfig, Settings
from db import session_scope
from exceptions import SecurityScoutError
from tools.slack import SlackClient

_LOG = structlog.get_logger(__name__)

_REPLAY_WINDOW_SEC = 300
_MAX_FUTURE_SKEW_SEC = 60


class SlackVerificationError(SecurityScoutError):
    """Slack signature or freshness check failed."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message, is_transient=False, is_resource_error=False)


class SlackActionId(StrEnum):
    approve = "security_scout:approve"
    reject = "security_scout:reject"
    escalate = "security_scout:escalate"
    dedup_confirm = "security_scout:dedup_confirm"
    dedup_new_instance = "security_scout:dedup_new_instance"
    dedup_reopen = "security_scout:dedup_reopen"
    dedup_resolved = "security_scout:dedup_resolved"
    risk_still_accepted = "security_scout:risk_still_accepted"
    risk_reevaluate = "security_scout:risk_reevaluate"


_APPROVAL_ACTION_IDS: frozenset[SlackActionId] = frozenset(
    {SlackActionId.approve, SlackActionId.reject, SlackActionId.escalate},
)
_DEDUP_ACTION_IDS: frozenset[SlackActionId] = frozenset(
    {
        SlackActionId.dedup_confirm,
        SlackActionId.dedup_new_instance,
        SlackActionId.dedup_reopen,
        SlackActionId.dedup_resolved,
        SlackActionId.risk_still_accepted,
        SlackActionId.risk_reevaluate,
    },
)


@dataclass(frozen=True, slots=True)
class SlackInteractivePayload:
    action: SlackActionId
    user_id: str
    channel_id: str
    message_ts: str
    button_value: str


class SlackWebhookProvider:
    """Verifies Slack request signatures and freshness."""

    def verify_signature(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> None:
        sig_header = headers.get("x-slack-signature") or headers.get("X-Slack-Signature")
        ts_header = headers.get("x-slack-request-timestamp") or headers.get("X-Slack-Request-Timestamp")
        if not sig_header:
            raise SlackVerificationError("missing X-Slack-Signature")
        if not ts_header:
            raise SlackVerificationError("missing X-Slack-Request-Timestamp")
        basestring = b"v0:" + ts_header.encode("utf-8") + b":" + raw_body
        mac = hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256)
        expected = "v0=" + mac.hexdigest()
        if not hmac.compare_digest(expected, sig_header):
            raise SlackVerificationError("invalid Slack signature")

    def assert_delivery_fresh(
        self,
        headers: Mapping[str, str],
        *,
        now: datetime,
        replay_window_sec: int = _REPLAY_WINDOW_SEC,
    ) -> None:
        ts_header = headers.get("x-slack-request-timestamp") or headers.get("X-Slack-Request-Timestamp")
        if not ts_header:
            raise SlackVerificationError("missing X-Slack-Request-Timestamp")
        try:
            ts = int(ts_header)
        except ValueError as exc:
            raise SlackVerificationError("invalid X-Slack-Request-Timestamp") from exc
        skew = now.timestamp() - ts
        if skew < -_MAX_FUTURE_SKEW_SEC:
            raise SlackVerificationError("Slack timestamp too far in the future")
        if skew > replay_window_sec:
            raise SlackVerificationError("stale Slack delivery")


def parse_interactive_payload(raw_body: bytes) -> SlackInteractivePayload:
    """Decode a Slack ``application/x-www-form-urlencoded`` interactive payload."""
    try:
        form = parse_qs(raw_body.decode("utf-8"), strict_parsing=True)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SlackVerificationError("invalid form body") from exc
    payload_field = form.get("payload")
    if not payload_field:
        raise SlackVerificationError("missing payload field")
    try:
        payload: dict[str, Any] = json.loads(payload_field[0])
    except json.JSONDecodeError as exc:
        raise SlackVerificationError("invalid payload JSON") from exc
    if not isinstance(payload, dict):
        raise SlackVerificationError("payload is not a JSON object")
    ptype = payload.get("type")
    if ptype != "block_actions":
        raise SlackVerificationError(f"unsupported payload type: {ptype!r}")

    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise SlackVerificationError("missing actions")
    first = actions[0]
    if not isinstance(first, dict):
        raise SlackVerificationError("malformed action entry")
    action_id_raw = first.get("action_id")
    button_value = first.get("value")
    if not isinstance(action_id_raw, str) or not isinstance(button_value, str):
        raise SlackVerificationError("malformed action fields")
    try:
        action_id = SlackActionId(action_id_raw)
    except ValueError as exc:
        raise SlackVerificationError(f"unknown action_id: {action_id_raw!r}") from exc

    user = payload.get("user")
    if not isinstance(user, dict):
        raise SlackVerificationError("missing user")
    user_id = user.get("id")
    if not isinstance(user_id, str) or not user_id:
        raise SlackVerificationError("missing user id")

    container = payload.get("container")
    channel = payload.get("channel")
    message_ts: str | None = None
    channel_id: str | None = None
    if isinstance(container, dict):
        mts = container.get("message_ts")
        ch = container.get("channel_id")
        if isinstance(mts, str):
            message_ts = mts
        if isinstance(ch, str):
            channel_id = ch
    if channel_id is None and isinstance(channel, dict):
        ch = channel.get("id")
        if isinstance(ch, str):
            channel_id = ch
    if not message_ts or not channel_id:
        raise SlackVerificationError("missing channel or message_ts")

    return SlackInteractivePayload(
        action=action_id,
        user_id=user_id,
        channel_id=channel_id,
        message_ts=message_ts,
        button_value=button_value,
    )


async def slack_webhook(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    app_config: AppConfig = request.app.state.app_config
    session_factory = request.app.state.session_factory

    raw = await request.body()
    headers = dict(request.headers)
    provider = SlackWebhookProvider()
    try:
        provider.verify_signature(raw, headers, settings.slack_signing_secret)
        provider.assert_delivery_fresh(headers, now=datetime.now(UTC))
    except SlackVerificationError as exc:
        _LOG.warning("slack_webhook_verification_failed", reason=str(exc))
        raise HTTPException(status_code=401, detail="unauthorized") from exc

    try:
        payload = parse_interactive_payload(raw)
    except SlackVerificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log = _LOG.bind(
        slack_action=payload.action.value,
        slack_user=payload.user_id,
    )

    is_dedup = payload.action in _DEDUP_ACTION_IDS
    try:
        if is_dedup:
            dedup_ctx = DedupContext.from_button_value(payload.button_value)
        else:
            ctx = ApprovalContext.from_button_value(payload.button_value)
    except ValueError as exc:
        log.warning("slack_webhook_bad_value", err=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with (
        SlackClient(settings.slack_bot_token) as slack,
        session_scope(session_factory) as session,
    ):
        if is_dedup:
            await handle_slack_dedup_decision(
                session,
                slack,
                ctx=dedup_ctx,
                action_id=payload.action,
                user_id=payload.user_id,
                channel_id=payload.channel_id,
                message_ts=payload.message_ts,
            )
            handled_finding_id = dedup_ctx.finding_id
            handled_run_id = dedup_ctx.workflow_run_id
        else:
            await handle_slack_approval(
                session,
                app_config,
                slack,
                ctx=ctx,
                action=payload.action,
                user_id=payload.user_id,
                channel_id=payload.channel_id,
                message_ts=payload.message_ts,
            )
            handled_finding_id = ctx.finding_id
            handled_run_id = ctx.workflow_run_id

    log.info(
        "slack_webhook_handled",
        metric_name="slack_webhook_handled_total",
        finding_id=str(handled_finding_id),
        workflow_run_id=str(handled_run_id),
    )
    return Response(status_code=200)


def create_slack_webhook_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/webhooks/slack", slack_webhook, methods=["POST"])
    return router


__all__ = [
    "SlackActionId",
    "SlackInteractivePayload",
    "SlackVerificationError",
    "SlackWebhookProvider",
    "create_slack_webhook_router",
    "parse_interactive_payload",
    "slack_webhook",
]

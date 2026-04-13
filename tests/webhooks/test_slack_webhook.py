from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from urllib.parse import urlencode

import pytest

from webhooks.slack import (
    SlackActionId,
    SlackVerificationError,
    SlackWebhookProvider,
    parse_interactive_payload,
)

_PROVIDER = SlackWebhookProvider()
_SECRET = "shhh-signing-secret"


def _sign(ts: str, body: bytes, secret: str = _SECRET) -> str:
    basestring = b"v0:" + ts.encode("utf-8") + b":" + body
    mac = hmac.new(secret.encode("utf-8"), basestring, hashlib.sha256)
    return "v0=" + mac.hexdigest()


def test_verify_signature_accepts_canonical_slack_example() -> None:
    # Mirrors Slack's published verification recipe.
    ts = str(int(time.time()))
    body = b"token=xoxb&team_id=T0001"
    sig = _sign(ts, body)
    _PROVIDER.verify_signature(
        body,
        {"X-Slack-Signature": sig, "X-Slack-Request-Timestamp": ts},
        _SECRET,
    )


def test_verify_signature_rejects_tampered_body() -> None:
    ts = str(int(time.time()))
    body = b"payload=1"
    sig = _sign(ts, body)
    with pytest.raises(SlackVerificationError, match="invalid Slack signature"):
        _PROVIDER.verify_signature(
            body + b"!",
            {"X-Slack-Signature": sig, "X-Slack-Request-Timestamp": ts},
            _SECRET,
        )


def test_verify_signature_requires_headers() -> None:
    with pytest.raises(SlackVerificationError, match="X-Slack-Signature"):
        _PROVIDER.verify_signature(b"", {}, _SECRET)
    with pytest.raises(SlackVerificationError, match="Timestamp"):
        _PROVIDER.verify_signature(b"", {"X-Slack-Signature": "v0=deadbeef"}, _SECRET)


def test_assert_delivery_fresh_rejects_stale_timestamp() -> None:
    now = time.time()
    stale = str(int(now - 1000))
    with pytest.raises(SlackVerificationError, match="stale"):
        _PROVIDER.assert_delivery_fresh(
            {"X-Slack-Request-Timestamp": stale},
            now=_dt_from_epoch(now),
        )


def test_assert_delivery_fresh_rejects_future_skew() -> None:
    now = time.time()
    future = str(int(now + 1000))
    with pytest.raises(SlackVerificationError, match="future"):
        _PROVIDER.assert_delivery_fresh(
            {"X-Slack-Request-Timestamp": future},
            now=_dt_from_epoch(now),
        )


def test_assert_delivery_fresh_rejects_malformed_timestamp() -> None:
    with pytest.raises(SlackVerificationError, match="invalid"):
        _PROVIDER.assert_delivery_fresh(
            {"X-Slack-Request-Timestamp": "not-a-number"},
            now=_dt_from_epoch(time.time()),
        )


def _dt_from_epoch(epoch: float):
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, tz=UTC)


def _encoded_button_value() -> str:
    return f"{uuid.uuid4()}|{uuid.uuid4()}|demo"


def test_parse_interactive_payload_decodes_approve_action() -> None:
    value = _encoded_button_value()
    payload = {
        "type": "block_actions",
        "user": {"id": "U01ABC"},
        "container": {"type": "message", "message_ts": "1111.2222", "channel_id": "C123"},
        "channel": {"id": "C123"},
        "actions": [
            {
                "action_id": "security_scout:approve",
                "value": value,
                "type": "button",
            }
        ],
    }
    body = urlencode({"payload": json.dumps(payload)}).encode("utf-8")
    out = parse_interactive_payload(body)
    assert out.action == SlackActionId.approve
    assert out.user_id == "U01ABC"
    assert out.channel_id == "C123"
    assert out.message_ts == "1111.2222"
    assert out.button_value == value


def test_parse_interactive_payload_rejects_unknown_action() -> None:
    payload = {
        "type": "block_actions",
        "user": {"id": "U01"},
        "container": {"message_ts": "1.0", "channel_id": "C1"},
        "actions": [{"action_id": "not_ours", "value": "v", "type": "button"}],
    }
    body = urlencode({"payload": json.dumps(payload)}).encode("utf-8")
    with pytest.raises(SlackVerificationError, match="unknown action_id"):
        parse_interactive_payload(body)


def test_parse_interactive_payload_rejects_non_block_actions() -> None:
    payload = {"type": "view_submission"}
    body = urlencode({"payload": json.dumps(payload)}).encode("utf-8")
    with pytest.raises(SlackVerificationError, match="unsupported"):
        parse_interactive_payload(body)


def test_parse_interactive_payload_requires_payload_field() -> None:
    with pytest.raises(SlackVerificationError, match="missing payload"):
        parse_interactive_payload(b"other=thing")

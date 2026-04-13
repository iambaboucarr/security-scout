"""GitHub implementation of the WebhookProvider protocol (ADR-027).

Wraps the existing verification helpers from ``webhooks.github`` without
changing their logic.  The webhook route handler uses this provider instead
of calling those helpers directly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import structlog

from webhooks.scm.protocol import WebhookEvent, WebhookVerificationError

_LOG = structlog.get_logger(__name__)

_REPLAY_WINDOW_SEC = 300
_MAX_FUTURE_SKEW_SEC = 60


class GitHubWebhookProvider:
    """``WebhookProvider`` backed by GitHub webhook conventions."""

    def verify_signature(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> None:
        sig_header = headers.get("x-hub-signature-256") or headers.get("X-Hub-Signature-256")
        if not sig_header:
            raise WebhookVerificationError("missing X-Hub-Signature-256")
        mac = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256)
        expected = "sha256=" + mac.hexdigest()
        if not hmac.compare_digest(expected, sig_header):
            raise WebhookVerificationError("invalid webhook signature")

    def assert_delivery_fresh(
        self,
        headers: Mapping[str, str],
        *,
        now: datetime,
        replay_window_sec: int = _REPLAY_WINDOW_SEC,
    ) -> None:
        date_header = headers.get("date") or headers.get("Date")
        if not date_header:
            _LOG.info(
                "webhook_delivery_date_freshness_skipped",
                metric_name="webhook_delivery_date_freshness_skipped",
                reason="no_http_date_header",
            )
            return
        try:
            parsed = parsedate_to_datetime(date_header)
        except TypeError, ValueError:
            raise WebhookVerificationError("invalid HTTP Date header") from None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        skew = (now - parsed.astimezone(UTC)).total_seconds()
        if skew < -_MAX_FUTURE_SKEW_SEC:
            raise WebhookVerificationError("webhook Date too far in the future")
        if skew > replay_window_sec:
            raise WebhookVerificationError("stale webhook delivery")

    def parse_event(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
    ) -> WebhookEvent:
        event_type = (headers.get("x-github-event") or headers.get("X-GitHub-Event") or "").strip()
        delivery_id = (headers.get("x-github-delivery") or headers.get("X-GitHub-Delivery") or "").strip()

        try:
            payload: dict[str, Any] = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WebhookVerificationError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise WebhookVerificationError("webhook payload is not a JSON object")

        action = payload.get("action")
        action_str = action if isinstance(action, str) else None

        owner: str | None = None
        name: str | None = None
        repo_obj = payload.get("repository")
        if isinstance(repo_obj, dict):
            owner, name = _owner_repo_from_repository_dict(repo_obj)

        return WebhookEvent(
            event_type=event_type,
            delivery_id=delivery_id,
            action=action_str,
            payload=payload,
            repo_owner=owner,
            repo_name=name,
        )


def _owner_repo_from_repository_dict(repo_obj: dict[str, Any]) -> tuple[str | None, str | None]:
    full_name = repo_obj.get("full_name")
    if isinstance(full_name, str) and "/" in full_name:
        fn_owner, _, fn_name = full_name.partition("/")
        if fn_owner and fn_name:
            return fn_owner, fn_name
    owner_obj = repo_obj.get("owner")
    if isinstance(owner_obj, dict):
        login = owner_obj.get("login")
        repo_name = repo_obj.get("name")
        if isinstance(login, str) and isinstance(repo_name, str):
            return login, repo_name
    return None, None


__all__ = ["GitHubWebhookProvider"]

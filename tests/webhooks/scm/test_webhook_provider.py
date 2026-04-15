# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from webhooks.scm.github import GitHubWebhookProvider
from webhooks.scm.protocol import WebhookEvent, WebhookProvider, WebhookVerificationError


def test_github_webhook_provider_is_runtime_checkable() -> None:
    assert isinstance(GitHubWebhookProvider(), WebhookProvider)


class TestVerifySignature:
    def _sign(self, body: bytes, secret: str) -> str:
        mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
        return "sha256=" + mac.hexdigest()

    def test_valid_signature_accepted(self) -> None:
        provider = GitHubWebhookProvider()
        body = b"Hello, World!"
        secret = "test-secret"
        headers = {"X-Hub-Signature-256": self._sign(body, secret)}
        provider.verify_signature(body, headers, secret)

    def test_valid_signature_lowercase_header(self) -> None:
        provider = GitHubWebhookProvider()
        body = b"payload"
        secret = "s3cret"
        headers = {"x-hub-signature-256": self._sign(body, secret)}
        provider.verify_signature(body, headers, secret)

    def test_missing_signature_raises(self) -> None:
        provider = GitHubWebhookProvider()
        with pytest.raises(WebhookVerificationError, match="missing"):
            provider.verify_signature(b"body", {}, "secret")

    def test_tampered_body_raises(self) -> None:
        provider = GitHubWebhookProvider()
        body = b"original"
        secret = "secret"
        headers = {"X-Hub-Signature-256": self._sign(body, secret)}
        with pytest.raises(WebhookVerificationError, match="invalid"):
            provider.verify_signature(b"tampered", headers, secret)


class TestAssertDeliveryFresh:
    def test_missing_date_header_allowed(self) -> None:
        provider = GitHubWebhookProvider()
        provider.assert_delivery_fresh({}, now=datetime.now(UTC))

    def test_fresh_date_accepted(self) -> None:
        provider = GitHubWebhookProvider()
        now = datetime.now(UTC)
        headers = {"Date": format_datetime(now - timedelta(seconds=10))}
        provider.assert_delivery_fresh(headers, now=now)

    def test_stale_date_rejected(self) -> None:
        provider = GitHubWebhookProvider()
        now = datetime.now(UTC)
        headers = {"Date": format_datetime(now - timedelta(seconds=400))}
        with pytest.raises(WebhookVerificationError, match="stale"):
            provider.assert_delivery_fresh(headers, now=now)

    def test_future_date_rejected(self) -> None:
        provider = GitHubWebhookProvider()
        now = datetime.now(UTC)
        headers = {"Date": format_datetime(now + timedelta(seconds=120))}
        with pytest.raises(WebhookVerificationError, match="future"):
            provider.assert_delivery_fresh(headers, now=now)

    def test_malformed_date_rejected(self) -> None:
        provider = GitHubWebhookProvider()
        headers = {"Date": "not-a-date"}
        with pytest.raises(WebhookVerificationError, match="invalid"):
            provider.assert_delivery_fresh(headers, now=datetime.now(UTC))


class TestParseEvent:
    def _make_headers(
        self,
        event: str = "repository_advisory",
        delivery: str = "abc-123",
    ) -> dict[str, str]:
        return {
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
        }

    def test_parses_advisory_event(self) -> None:
        provider = GitHubWebhookProvider()
        payload = {
            "action": "published",
            "repository": {"full_name": "acme/app"},
            "repository_advisory": {"ghsa_id": "GHSA-ABCD-EFGH-IJKL"},
        }
        raw = json.dumps(payload).encode()
        event = provider.parse_event(raw, self._make_headers())
        assert isinstance(event, WebhookEvent)
        assert event.event_type == "repository_advisory"
        assert event.delivery_id == "abc-123"
        assert event.action == "published"
        assert event.repo_owner == "acme"
        assert event.repo_name == "app"
        assert event.payload["repository_advisory"]["ghsa_id"] == "GHSA-ABCD-EFGH-IJKL"

    def test_parses_ping_event(self) -> None:
        provider = GitHubWebhookProvider()
        payload = {"zen": "Keep it logically awesome."}
        raw = json.dumps(payload).encode()
        event = provider.parse_event(raw, self._make_headers(event="ping"))
        assert event.event_type == "ping"
        assert event.repo_owner is None
        assert event.repo_name is None

    def test_parses_repo_from_owner_login(self) -> None:
        provider = GitHubWebhookProvider()
        payload = {
            "action": "created",
            "repository": {"owner": {"login": "acme"}, "name": "app"},
        }
        raw = json.dumps(payload).encode()
        event = provider.parse_event(raw, self._make_headers())
        assert event.repo_owner == "acme"
        assert event.repo_name == "app"

    def test_invalid_json_raises(self) -> None:
        provider = GitHubWebhookProvider()
        with pytest.raises(WebhookVerificationError, match="invalid JSON"):
            provider.parse_event(b"not json", self._make_headers())

    def test_non_object_json_raises(self) -> None:
        provider = GitHubWebhookProvider()
        with pytest.raises(WebhookVerificationError, match="not a JSON object"):
            provider.parse_event(b"[]", self._make_headers())

    def test_lowercase_headers(self) -> None:
        provider = GitHubWebhookProvider()
        payload = {"action": "opened"}
        raw = json.dumps(payload).encode()
        headers = {"x-github-event": "pull_request", "x-github-delivery": "xyz"}
        event = provider.parse_event(raw, headers)
        assert event.event_type == "pull_request"
        assert event.delivery_id == "xyz"

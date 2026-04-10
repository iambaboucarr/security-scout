from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from webhooks.github import (
    assert_delivery_fresh_http_date,
    verify_github_hub_signature_256,
)


def test_verify_github_hub_signature_256_accepts_github_doc_example() -> None:
    """Official test vector from GitHub validating webhook deliveries (current GitHub Docs)."""
    secret = "It's a Secret to Everybody"
    payload = b"Hello, World!"
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    header = "sha256=" + mac.hexdigest()
    verify_github_hub_signature_256(payload, secret, header)


def test_verify_github_hub_signature_256_rejects_tamper() -> None:
    from fastapi import HTTPException

    secret = "It's a Secret to Everybody"
    payload = b"Hello, World!"
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    header = "sha256=" + mac.hexdigest()
    with pytest.raises(HTTPException) as ei:
        verify_github_hub_signature_256(payload + b"!", secret, header)
    assert ei.value.status_code == 401


def test_assert_delivery_fresh_http_date_rejects_stale() -> None:
    from email.utils import format_datetime

    from fastapi import HTTPException

    old = format_datetime(datetime.now(UTC) - timedelta(seconds=400))
    with pytest.raises(HTTPException) as ei:
        assert_delivery_fresh_http_date(old, now=datetime.now(UTC))
    assert ei.value.status_code == 401


def test_assert_delivery_fresh_http_date_allows_no_date() -> None:
    assert_delivery_fresh_http_date(None, now=datetime.now(UTC))


def test_github_webhook_ping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "repos.yaml"
    manifest.write_text(
        "repos:\n"
        "  - name: demo\n"
        "    github_org: acme\n"
        "    github_repo: app\n"
        "    slack_channel: '#sec'\n"
        "    allowed_workflows: []\n"
        "    notify_on_severity: [high]\n"
        "    require_approval_for: [critical]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "whsec-test")
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))

    from main import create_app

    app = create_app()

    class _Pool:
        async def close(self) -> None:
            return None

        async def enqueue_job(self, *_a: object, **_kw: object) -> str:
            return "job-id"

    async def _fake_create_pool(_rs: object) -> _Pool:
        return _Pool()

    monkeypatch.setattr("main.create_pool", _fake_create_pool)

    body = b"{}"
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()

    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "ping",
                "X-Hub-Signature-256": sig,
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 200


def test_github_webhook_enqueues_repository_advisory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "repos.yaml"
    manifest.write_text(
        "repos:\n"
        "  - name: demo\n"
        "    github_org: acme\n"
        "    github_repo: app\n"
        "    slack_channel: '#sec'\n"
        "    allowed_workflows: []\n"
        "    notify_on_severity: [high]\n"
        "    require_approval_for: [critical]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "whsec-test")
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))

    from main import create_app

    app = create_app()
    enqueued: list[dict[str, str]] = []

    class _Pool:
        async def close(self) -> None:
            return None

        async def enqueue_job(self, *_a: object, **kw: object) -> str:
            enqueued.append({k: str(v) for k, v in kw.items()})
            return "job-id"

    async def _fake_create_pool(_rs: object) -> _Pool:
        return _Pool()

    monkeypatch.setattr("main.create_pool", _fake_create_pool)

    payload = {
        "repository": {"full_name": "acme/app"},
        "repository_advisory": {"ghsa_id": "GHSA-xxxx-yyyy-zzzz"},
    }
    body = json.dumps(payload).encode("utf-8")
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()

    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "repository_advisory",
                "X-Hub-Signature-256": sig,
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 202
    assert len(enqueued) == 1
    assert enqueued[0]["repo_name"] == "demo"
    assert enqueued[0]["ghsa_id"] == "GHSA-XXXX-YYYY-ZZZZ"

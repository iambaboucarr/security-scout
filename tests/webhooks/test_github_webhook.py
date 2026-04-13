from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from webhooks.scm.github import GitHubWebhookProvider
from webhooks.scm.protocol import WebhookVerificationError

_PROVIDER = GitHubWebhookProvider()


def test_verify_github_hub_signature_256_accepts_github_doc_example() -> None:
    """Official test vector from GitHub validating webhook deliveries (current GitHub Docs)."""
    secret = "It's a Secret to Everybody"
    payload = b"Hello, World!"
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    header = "sha256=" + mac.hexdigest()
    _PROVIDER.verify_signature(payload, {"X-Hub-Signature-256": header}, secret)


def test_verify_github_hub_signature_256_rejects_tamper() -> None:
    secret = "It's a Secret to Everybody"
    payload = b"Hello, World!"
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    header = "sha256=" + mac.hexdigest()
    with pytest.raises(WebhookVerificationError, match="invalid"):
        _PROVIDER.verify_signature(payload + b"!", {"X-Hub-Signature-256": header}, secret)


def test_assert_delivery_fresh_http_date_rejects_stale() -> None:
    old = format_datetime(datetime.now(UTC) - timedelta(seconds=400))
    with pytest.raises(WebhookVerificationError, match="stale"):
        _PROVIDER.assert_delivery_fresh({"Date": old}, now=datetime.now(UTC))


def test_assert_delivery_fresh_http_date_rejects_malformed() -> None:
    with pytest.raises(WebhookVerificationError, match="invalid"):
        _PROVIDER.assert_delivery_fresh({"Date": "not-a-date!!!"}, now=datetime.now(UTC))


def test_assert_delivery_fresh_http_date_allows_no_date() -> None:
    _PROVIDER.assert_delivery_fresh({}, now=datetime.now(UTC))


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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'app.db'}")

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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'app.db'}")

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


def _webhook_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, list[dict[str, str]]]:
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'app.db'}")

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
    return app, enqueued


def test_github_webhook_dependabot_alert_enqueues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, enqueued = _webhook_app(tmp_path, monkeypatch)
    payload = {
        "repository": {"full_name": "acme/app"},
        "alert": {
            "security_advisory": {"ghsa_id": "GHSA-aaaa-bbbb-cccc"},
        },
    }
    body = json.dumps(payload).encode("utf-8")
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()
    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "dependabot_alert",
                "X-Hub-Signature-256": sig,
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 202
    assert len(enqueued) == 1
    assert enqueued[0]["ghsa_id"] == "GHSA-AAAA-BBBB-CCCC"


def test_github_webhook_unknown_repo_no_enqueue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, enqueued = _webhook_app(tmp_path, monkeypatch)
    payload = {
        "repository": {"full_name": "otherorg/other"},
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
    assert enqueued == []


def test_github_webhook_pull_request_deferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _enqueued = _webhook_app(tmp_path, monkeypatch)
    body = b"{}"
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()
    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": sig,
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 202


def test_github_webhook_security_advisory_global_deferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _enqueued = _webhook_app(tmp_path, monkeypatch)
    body = b"{}"
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()
    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "security_advisory",
                "X-Hub-Signature-256": sig,
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 202


def test_github_webhook_invalid_json_returns_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _enqueued = _webhook_app(tmp_path, monkeypatch)
    body = b"{"
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
    assert r.status_code == 400


def test_github_webhook_missing_signature_returns_401(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _enqueued = _webhook_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=b"{}",
            headers={
                "X-GitHub-Event": "ping",
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 401


def test_github_webhook_repository_owner_name_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, enqueued = _webhook_app(tmp_path, monkeypatch)
    payload = {
        "repository": {"owner": {"login": "acme"}, "name": "app"},
        "repository_advisory": {"ghsa_id": "GHSA-zzzz-yyyy-xxxx"},
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


def test_github_webhook_unhandled_event_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, enqueued = _webhook_app(tmp_path, monkeypatch)
    body = b"{}"
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()
    with TestClient(app) as client:
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "Date": format_datetime(datetime.now(UTC)),
            },
        )
    assert r.status_code == 202
    assert enqueued == []

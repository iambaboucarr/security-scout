# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from starlette.testclient import TestClient

from agents.orchestrator import AdvisoryWorkflowState
from db import session_scope
from models import Finding, WorkflowKind, WorkflowRun
from worker import process_advisory_workflow_job

pytestmark = pytest.mark.integration

_GHSA = "GHSA-XXXX-YYYY-ZZZZ"
_ADVISORY_JSON: dict[str, Any] = {
    "ghsa_id": _GHSA,
    "summary": "Integration test advisory",
    "description": "Synthetic body for pipeline test.",
    "severity": "high",
    "identifiers": [{"type": "CVE", "value": "CVE-2024-12345"}],
    "cwes": [{"cwe_id": "CWE-79"}],
    "cvss": {
        "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "score": 9.8,
    },
    "html_url": "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz",
    "vulnerabilities": [],
}


def _manifest(tmp_path: Path) -> Path:
    p = tmp_path / "repos.yaml"
    # Governance routes `high` to the notify tier so the end-to-end pipeline terminates
    # in `done` (the interactive approval loop is not yet implemented).
    p.write_text(
        "repos:\n"
        "  - name: demo\n"
        "    github_org: acme\n"
        "    github_repo: app\n"
        "    slack_channel: '#sec-alerts'\n"
        "    allowed_workflows: []\n"
        "    notify_on_severity: [high]\n"
        "    require_approval_for: [critical]\n"
        "    governance:\n"
        "      notify:\n"
        "        - severity: [high]\n",
        encoding="utf-8",
    )
    return p


def _mock_transport() -> tuple[httpx.MockTransport, list[str]]:
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        url = str(request.url)
        if request.method == "GET" and request.url.path == f"/repos/acme/app/security-advisories/{_GHSA}":
            return httpx.Response(200, json=_ADVISORY_JSON)
        if request.method == "POST" and request.url.host.endswith("slack.com"):
            return httpx.Response(200, json={"ok": True, "channel": "CINTEGRATION", "ts": "1111.2222"})
        if request.method == "POST" and request.url.host == "api.osv.dev":
            return httpx.Response(200, json={"vulns": []})
        return httpx.Response(404, text=f"unexpected request: {request.method} {url}")

    return httpx.MockTransport(handle), paths


class _ImmediateRedisPool:
    """Runs the advisory worker job in-process instead of Redis (ARQ contract preserved)."""

    def __init__(self, holder: dict[str, Any]) -> None:
        self._holder = holder
        self._dedup_keys: dict[str, str] = {}

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool | None:
        if nx and key in self._dedup_keys:
            return None
        self._dedup_keys[key] = value
        return True

    async def close(self) -> None:
        return None

    async def enqueue_job(
        self,
        _job_name: str,
        *,
        repo_name: str,
        ghsa_id: str,
        advisory_source: str = "repository",
        resume_workflow_run_id: str | None = None,
    ) -> str:
        ctx = self._holder["ctx"]
        await process_advisory_workflow_job(
            ctx,
            repo_name=repo_name,
            ghsa_id=ghsa_id,
            advisory_source=advisory_source,
            resume_workflow_run_id=resume_workflow_run_id,
        )
        return "integration-job-id"


class _FakeRedis:
    async def enqueue_job(self, *_a: object, **_kw: object) -> str:
        return "deferred"


@pytest.mark.asyncio
async def test_webhook_enqueues_worker_writes_db_and_slack_mocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, http_paths = _mock_transport()
    real = httpx.AsyncClient

    def _client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        if kwargs.get("transport") is None:
            kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)

    db_file = tmp_path / "scout.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(_manifest(tmp_path)))
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "whsec-test")
    monkeypatch.setenv("GITHUB_PAT", "ghp-test-token")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    holder: dict[str, Any] = {}

    async def _fake_create_pool(_rs: object) -> _ImmediateRedisPool:
        return _ImmediateRedisPool(holder)

    monkeypatch.setattr("main.create_pool", _fake_create_pool)

    from main import create_app

    app = create_app()

    body = json.dumps(
        {
            "repository": {"full_name": "acme/app"},
            "repository_advisory": {"ghsa_id": "ghsa-xxxx-yyyy-zzzz"},
        },
    ).encode("utf-8")
    mac = hmac.new(b"whsec-test", body, hashlib.sha256)
    sig = "sha256=" + mac.hexdigest()

    worker_http = httpx.AsyncClient(transport=transport, timeout=30.0)
    try:
        with TestClient(app) as client:
            st = client.app.state
            holder["ctx"] = {
                "settings": st.settings,
                "app_config": st.app_config,
                "engine": st.engine,
                "session_factory": st.session_factory,
                "http_client": worker_http,
                "llm": None,
                "redis": _FakeRedis(),
            }
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

            factory = client.app.state.session_factory
            async with session_scope(factory) as session:
                runs = (await session.execute(select(WorkflowRun))).scalars().all()
                assert len(runs) == 1
                assert runs[0].workflow_type == WorkflowKind.advisory
                assert runs[0].state == AdvisoryWorkflowState.done.value
                assert runs[0].finding_id is not None

                findings = (await session.execute(select(Finding))).scalars().all()
                assert len(findings) == 1
                assert _GHSA in findings[0].source_ref.upper()
    finally:
        await worker_http.aclose()

    assert any(f"/repos/acme/app/security-advisories/{_GHSA}" in p for p in http_paths)
    assert any(p.endswith("/chat.postMessage") for p in http_paths)

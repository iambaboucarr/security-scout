from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from worker import WorkerSettings, process_advisory_workflow_job, shutdown, startup


def test_worker_settings_registers_advisory_job() -> None:
    assert process_advisory_workflow_job in WorkerSettings.functions
    assert WorkerSettings.on_startup is not None
    assert WorkerSettings.on_shutdown is not None


@pytest.mark.asyncio
async def test_startup_shutdown_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    ctx: dict[str, Any] = {}
    await startup(ctx)
    assert "settings" in ctx
    assert "session_factory" in ctx
    assert ctx["llm"] is None
    assert isinstance(ctx["http_client"], httpx.AsyncClient)

    await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_advisory_workflow_job_unknown_repo_logs_and_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        await process_advisory_workflow_job(ctx, repo_name="nope", ghsa_id="GHSA-AAAA-BBBB-CCCC")
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_advisory_workflow_job_invalid_source_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        await process_advisory_workflow_job(
            ctx, repo_name="demo", ghsa_id="GHSA-AAAA-BBBB-CCCC", advisory_source="other"
        )
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_startup_sets_llm_provider_when_api_key_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        assert ctx["llm"] is not None
    finally:
        await shutdown(ctx)

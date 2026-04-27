# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from datetime import UTC
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from redis.exceptions import RedisError

from db import session_scope
from models import (
    Base,
    Finding,
    FindingStatus,
    Severity,
    WorkflowKind,
)
from tools.slack import SlackAPIError
from worker import (
    WorkerSettings,
    _advisory_sync_cron_jobs,
    _patch_oracle_failure_reply_text,
    configure_worker_cron_jobs,
    process_advisory_workflow_job,
    process_patch_oracle_job,
    shutdown,
    startup,
)


def test_patch_oracle_failure_reply_text_includes_exception_detail() -> None:
    text = _patch_oracle_failure_reply_text(RuntimeError("clone failed"))
    assert "Patch oracle failed" in text
    assert "RuntimeError" in text
    assert "clone failed" in text


def test_patch_oracle_failure_reply_text_empty_message_uses_type_only() -> None:
    text = _patch_oracle_failure_reply_text(RuntimeError())
    assert text == "Patch oracle failed: RuntimeError"


def test_worker_settings_registers_advisory_job() -> None:
    assert process_advisory_workflow_job in WorkerSettings.functions
    assert process_patch_oracle_job in WorkerSettings.functions
    assert WorkerSettings.on_startup is not None
    assert WorkerSettings.on_shutdown is not None


@pytest.mark.filterwarnings("ignore:.*iscoroutinefunction.*:DeprecationWarning")
def test_worker_settings_utc_timezone_and_cron_built_for_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert WorkerSettings.timezone is UTC
    assert WorkerSettings.cron_jobs == ()
    prev_cron = WorkerSettings.cron_jobs
    try:
        monkeypatch.setenv("ADVISORY_POLL_INTERVAL", "hourly")
        configure_worker_cron_jobs()
        assert len(WorkerSettings.cron_jobs) == 1
        cj = WorkerSettings.cron_jobs[0]
        assert cj.name == "sync_repository_advisories"
        assert cj.minute == 0
        assert cj.hour is None
        assert cj.unique is True
        assert cj.max_tries == 1
        jobs_direct = _advisory_sync_cron_jobs()
        assert len(jobs_direct) == 1
        assert jobs_direct[0].name == cj.name
    finally:
        WorkerSettings.cron_jobs = prev_cron


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
    assert ctx["advisory_polling_enabled"] is False

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
async def test_process_advisory_workflow_job_skips_on_dedupe_finding_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghsa = "GHSA-ABAA-ABAA-ABAA"
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
    db_path = tmp_path / "dd.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        engine = ctx["engine"]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_scope(ctx["session_factory"]) as s:
            s.add(
                Finding(
                    id=uuid.uuid4(),
                    workflow=WorkflowKind.advisory,
                    repo_name="acme/app",
                    source_ref="https://github.com/advisories/x",
                    severity=Severity.high,
                    title="t",
                    status=FindingStatus.unconfirmed,
                    evidence={"ghsa_id": ghsa},
                ),
            )
            await s.commit()
        with patch("worker.run_advisory_workflow", new_callable=AsyncMock) as m_run:
            await process_advisory_workflow_job(
                ctx,
                repo_name="demo",
                ghsa_id=ghsa,
                advisory_source="repository",
            )
        m_run.assert_not_awaited()
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_advisory_workflow_job_runs_when_prior_finding_is_false_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghsa = "GHSA-ACAA-ACAA-ACAA"
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
    db_path = tmp_path / "dd2.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        engine = ctx["engine"]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_scope(ctx["session_factory"]) as s:
            s.add(
                Finding(
                    id=uuid.uuid4(),
                    workflow=WorkflowKind.advisory,
                    repo_name="acme/app",
                    source_ref="https://github.com/advisories/x",
                    severity=Severity.high,
                    title="t",
                    status=FindingStatus.false_positive,
                    evidence={"ghsa_id": ghsa},
                ),
            )
            await s.commit()
        with patch("worker.run_advisory_workflow", new_callable=AsyncMock) as m_run:
            await process_advisory_workflow_job(
                ctx,
                repo_name="demo",
                ghsa_id=ghsa,
                advisory_source="repository",
            )
        m_run.assert_awaited_once()
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
async def test_process_patch_oracle_job_unknown_repo_returns(
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'po.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        await process_patch_oracle_job(
            ctx,
            repo_name="missing",
            finding_id="00000000-0000-0000-0000-000000000001",
            workflow_run_id="00000000-0000-0000-0000-000000000002",
            slack_channel_id="C1",
            slack_message_ts="1.0",
        )
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_patch_oracle_job_success_posts_thread_reply(
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'po2.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        mock_slack = AsyncMock()
        mock_slack.post_thread_reply = AsyncMock()
        mock_slack.__aenter__ = AsyncMock(return_value=mock_slack)
        mock_slack.__aexit__ = AsyncMock(return_value=None)
        mock_scm = AsyncMock()
        mock_scm.__aenter__ = AsyncMock(return_value=mock_scm)
        mock_scm.__aexit__ = AsyncMock(return_value=None)
        with (
            patch("worker.run_patch_oracle_job", new_callable=AsyncMock) as mock_oracle,
            patch("worker.SlackClient", return_value=mock_slack),
            patch("worker.GitHubSCMProvider", return_value=mock_scm),
        ):
            mock_oracle.return_value = (FindingStatus.confirmed_high, "Patch oracle complete.")
            await process_patch_oracle_job(
                ctx,
                repo_name="demo",
                finding_id="00000000-0000-0000-0000-000000000011",
                workflow_run_id="00000000-0000-0000-0000-000000000022",
                slack_channel_id="C1",
                slack_message_ts="1.0",
            )
        mock_oracle.assert_awaited_once()
        assert mock_oracle.call_args.kwargs.get("default_git_ref") == "main"
        mock_slack.post_thread_reply.assert_awaited_once()
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_patch_oracle_job_passes_custom_default_git_ref(
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
        "    require_approval_for: [critical]\n"
        "    default_git_ref: master\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'po2b.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        mock_slack = AsyncMock()
        mock_slack.post_thread_reply = AsyncMock()
        mock_slack.__aenter__ = AsyncMock(return_value=mock_slack)
        mock_slack.__aexit__ = AsyncMock(return_value=None)
        mock_scm = AsyncMock()
        mock_scm.__aenter__ = AsyncMock(return_value=mock_scm)
        mock_scm.__aexit__ = AsyncMock(return_value=None)
        with (
            patch("worker.run_patch_oracle_job", new_callable=AsyncMock) as mock_oracle,
            patch("worker.SlackClient", return_value=mock_slack),
            patch("worker.GitHubSCMProvider", return_value=mock_scm),
        ):
            mock_oracle.return_value = (FindingStatus.confirmed_high, "done")
            await process_patch_oracle_job(
                ctx,
                repo_name="demo",
                finding_id="00000000-0000-0000-0000-000000000011",
                workflow_run_id="00000000-0000-0000-0000-000000000022",
                slack_channel_id="C1",
                slack_message_ts="1.0",
            )
        assert mock_oracle.call_args.kwargs.get("default_git_ref") == "master"
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_patch_oracle_job_failure_posts_error_reply(
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'po3.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        mock_slack = AsyncMock()
        mock_slack.post_thread_reply = AsyncMock()
        mock_slack.__aenter__ = AsyncMock(return_value=mock_slack)
        mock_slack.__aexit__ = AsyncMock(return_value=None)
        mock_scm = AsyncMock()
        mock_scm.__aenter__ = AsyncMock(return_value=mock_scm)
        mock_scm.__aexit__ = AsyncMock(return_value=None)
        with (
            patch("worker.run_patch_oracle_job", new_callable=AsyncMock) as mock_oracle,
            patch("worker.SlackClient", return_value=mock_slack),
            patch("worker.GitHubSCMProvider", return_value=mock_scm),
        ):
            mock_oracle.side_effect = RuntimeError("clone failed")
            await process_patch_oracle_job(
                ctx,
                repo_name="demo",
                finding_id="00000000-0000-0000-0000-000000000033",
                workflow_run_id="00000000-0000-0000-0000-000000000044",
                slack_channel_id="C1",
                slack_message_ts="1.0",
            )
        mock_slack.post_thread_reply.assert_awaited_once()
        err_text = mock_slack.post_thread_reply.call_args.kwargs["text"]
        assert "Patch oracle failed" in err_text
        assert "clone failed" in err_text
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_patch_oracle_job_success_reply_slack_error_is_logged(
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'po4.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        mock_slack = AsyncMock()
        mock_slack.post_thread_reply = AsyncMock(
            side_effect=SlackAPIError("rate limited", is_transient=True, slack_error_code="rate_limited"),
        )
        mock_slack.__aenter__ = AsyncMock(return_value=mock_slack)
        mock_slack.__aexit__ = AsyncMock(return_value=None)
        mock_scm = AsyncMock()
        mock_scm.__aenter__ = AsyncMock(return_value=mock_scm)
        mock_scm.__aexit__ = AsyncMock(return_value=None)
        with (
            patch("worker.run_patch_oracle_job", new_callable=AsyncMock) as mock_oracle,
            patch("worker.SlackClient", return_value=mock_slack),
            patch("worker.GitHubSCMProvider", return_value=mock_scm),
        ):
            mock_oracle.return_value = (FindingStatus.confirmed_high, "done")
            await process_patch_oracle_job(
                ctx,
                repo_name="demo",
                finding_id="00000000-0000-0000-0000-000000000055",
                workflow_run_id="00000000-0000-0000-0000-000000000066",
                slack_channel_id="C1",
                slack_message_ts="1.0",
            )
        mock_oracle.assert_awaited_once()
        mock_slack.post_thread_reply.assert_awaited_once()
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_process_patch_oracle_job_failure_reply_slack_error_is_logged(
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'po5.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    ctx: dict[str, Any] = {}
    await startup(ctx)
    try:
        mock_slack = AsyncMock()
        mock_slack.post_thread_reply = AsyncMock(
            side_effect=SlackAPIError("rate limited", is_transient=True, slack_error_code="rate_limited"),
        )
        mock_slack.__aenter__ = AsyncMock(return_value=mock_slack)
        mock_slack.__aexit__ = AsyncMock(return_value=None)
        mock_scm = AsyncMock()
        mock_scm.__aenter__ = AsyncMock(return_value=mock_scm)
        mock_scm.__aexit__ = AsyncMock(return_value=None)
        with (
            patch("worker.run_patch_oracle_job", new_callable=AsyncMock) as mock_oracle,
            patch("worker.SlackClient", return_value=mock_slack),
            patch("worker.GitHubSCMProvider", return_value=mock_scm),
        ):
            mock_oracle.side_effect = RuntimeError("clone failed")
            await process_patch_oracle_job(
                ctx,
                repo_name="demo",
                finding_id="00000000-0000-0000-0000-000000000077",
                workflow_run_id="00000000-0000-0000-0000-000000000088",
                slack_channel_id="C1",
                slack_message_ts="1.0",
            )
        mock_oracle.assert_awaited_once()
        mock_slack.post_thread_reply.assert_awaited_once()
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


@pytest.mark.asyncio
async def test_startup_advisory_polling_enabled_when_redis_pings(
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
    monkeypatch.setenv("ADVISORY_POLL_INTERVAL", "hourly")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    ctx: dict[str, Any] = {"redis": redis}
    await startup(ctx)
    try:
        assert ctx["advisory_polling_enabled"] is True
        redis.ping.assert_awaited_once()
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_startup_advisory_polling_disabled_when_redis_ping_fails(
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
    monkeypatch.setenv("ADVISORY_POLL_INTERVAL", "hourly")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    redis = AsyncMock()
    redis.ping = AsyncMock(side_effect=RedisError("nop"))
    ctx: dict[str, Any] = {"redis": redis}
    await startup(ctx)
    try:
        assert ctx["advisory_polling_enabled"] is False
    finally:
        await shutdown(ctx)


@pytest.mark.asyncio
async def test_startup_advisory_polling_disabled_when_no_poll_states_but_interval_on(
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
        "    require_approval_for: [critical]\n"
        "    advisory_poll_states: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'w.db'}")
    monkeypatch.setenv("REPOS_CONFIG_PATH", str(manifest))
    monkeypatch.setenv("GITHUB_PAT", "pat")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("ADVISORY_POLL_INTERVAL", "hourly")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    ctx: dict[str, Any] = {"redis": redis}
    await startup(ctx)
    try:
        assert ctx["advisory_polling_enabled"] is False
        redis.ping.assert_not_called()
    finally:
        await shutdown(ctx)

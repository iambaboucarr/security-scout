from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from config import AppConfig, Settings, configure_logging, load_repos_manifest
from db import log_and_persist_config_loaded, persist_config_loaded_audit
from models import (
    AgentActionLog,
    Finding,
    FindingStatus,
    Severity,
    SSVCAction,
    WorkflowKind,
    WorkflowRun,
)


def _minimal_settings(repos_path: Path) -> Settings:
    return Settings(
        github_webhook_secret="test-whsec",
        github_pat="test-pat",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-signing",
        repos_config_path=repos_path,
    )


async def test_finding_round_trip(db_session) -> None:
    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="GHSA-xxxx-yyyy-zzzz",
        severity=Severity.high,
        ssvc_action=SSVCAction.act,
        status=FindingStatus.unconfirmed,
        triage_confidence=0.82,
        title="Test advisory",
        cve_id="CVE-2024-9999",
    )
    db_session.add(finding)
    await db_session.commit()

    row = await db_session.get(Finding, fid)
    assert row is not None
    assert row.severity == Severity.high
    assert row.triage_confidence == pytest.approx(0.82)
    assert row.cve_id == "CVE-2024-9999"


async def test_workflow_run_links_optional_finding(db_session) -> None:
    fid = uuid.uuid4()
    finding = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="GHSA-abcd",
        severity=Severity.medium,
        title="Linked",
    )
    run = WorkflowRun(
        finding_id=fid,
        workflow_type=WorkflowKind.advisory,
        state="triaging",
        retry_count=0,
    )
    db_session.add_all([finding, run])
    await db_session.commit()

    loaded = await db_session.get(WorkflowRun, run.id)
    assert loaded is not None
    assert loaded.finding_id == fid


async def test_agent_action_log_append_only(db_session) -> None:
    log = AgentActionLog(
        agent="triage",
        tool_name="read_advisory",
        tool_inputs={"advisory_id": "GHSA-test"},
        tool_output="truncated…",
        workflow_run_id=None,
    )
    db_session.add(log)
    await db_session.commit()

    result = await db_session.execute(select(AgentActionLog))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].tool_name == "read_advisory"


async def test_persist_config_loaded_audit_stores_checksum(db_session, tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    p.write_bytes(b"repos: []\n")
    manifest, _digest = load_repos_manifest(p)
    app = AppConfig(
        settings=_minimal_settings(p),
        repos=manifest,
        repos_yaml_sha256="a" * 64,
        repos_yaml_path=p.resolve(),
    )
    await persist_config_loaded_audit(db_session, app)
    await db_session.commit()

    result = await db_session.execute(select(AgentActionLog))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].tool_name == "config_loaded"
    assert rows[0].tool_inputs is not None
    assert rows[0].tool_inputs["repos_yaml_sha256"] == "a" * 64
    assert rows[0].workflow_run_id is None


async def test_log_and_persist_config_loaded_writes_log_and_row(
    db_session,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    p = tmp_path / "repos.yaml"
    p.write_bytes(b"repos: []\n")
    manifest, digest = load_repos_manifest(p)
    app = AppConfig(
        settings=_minimal_settings(p),
        repos=manifest,
        repos_yaml_sha256=digest,
        repos_yaml_path=p.resolve(),
    )
    configure_logging("INFO")
    await log_and_persist_config_loaded(db_session, app)
    await db_session.commit()

    out = capsys.readouterr().out
    assert "config_loaded" in out
    assert digest in out

    result = await db_session.execute(select(AgentActionLog).where(AgentActionLog.tool_name == "config_loaded"))
    row = result.scalar_one()
    assert row.tool_inputs is not None
    assert row.tool_inputs["repos_yaml_sha256"] == digest

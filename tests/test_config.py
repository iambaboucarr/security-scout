from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    AppConfig,
    GovernanceConfig,
    GovernanceRule,
    RepoConfig,
    RepoMode,
    ReposManifest,
    Settings,
    compute_repos_yaml_sha256,
    configure_logging,
    load_app_config,
    load_repos_manifest,
    log_config_loaded,
)
from models import Severity, SSVCAction


def _minimal_settings(repos_path: Path) -> Settings:
    return Settings(
        github_webhook_secret="test-whsec",
        github_pat="test-pat",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-signing",
        repos_config_path=repos_path,
    )


def _write_manifest(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_compute_repos_yaml_sha256_is_hex_and_stable() -> None:
    raw = b"hello\n"
    a = compute_repos_yaml_sha256(raw)
    b = compute_repos_yaml_sha256(raw)
    assert a == b
    assert len(a) == 64


def test_load_repos_manifest_rejects_empty_document(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty or not a mapping"):
        load_repos_manifest(p)


def test_load_repos_manifest_rejects_malformed_yaml(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    p.write_bytes(b"repos:\n  - name: bad\n  unindented: [\n")
    with pytest.raises(ValueError, match=str(p)) as exc_info:
        load_repos_manifest(p)
    assert "invalid YAML" in str(exc_info.value)
    assert "line" in str(exc_info.value)


def test_load_repos_manifest_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    p.write_bytes(b"[]\n")
    with pytest.raises(TypeError, match="mapping"):
        load_repos_manifest(p)


def test_load_repos_manifest_parses_and_checksums(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    body = """repos: []
"""
    p.write_bytes(body.encode())
    manifest, digest = load_repos_manifest(p)
    assert isinstance(manifest, ReposManifest)
    assert manifest.repos == []
    assert digest == compute_repos_yaml_sha256(body.encode())


def test_load_repos_manifest_rejects_duplicate_repo_name(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: dup
    github_org: a
    github_repo: r1
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
  - name: dup
    github_org: b
    github_repo: r2
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
""",
    )
    with pytest.raises(ValueError, match="duplicate repo name"):
        load_repos_manifest(p)


def test_load_repos_manifest_rejects_duplicate_org_repo(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: one
    github_org: acme
    github_repo: api
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
  - name: two
    github_org: acme
    github_repo: api
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
""",
    )
    with pytest.raises(ValueError, match="duplicate github_org"):
        load_repos_manifest(p)


def test_repo_config_defaults_mode_to_observe(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: svc
    github_org: o
    github_repo: r
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
""",
    )
    manifest, _ = load_repos_manifest(p)
    assert manifest.repos[0].mode is RepoMode.observe


def test_load_app_config_returns_app_config(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: svc
    github_org: o
    github_repo: r
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
""",
    )
    cfg = load_app_config(_minimal_settings(p))
    assert isinstance(cfg, AppConfig)
    assert cfg.repos_yaml_path == p.resolve()
    assert len(cfg.repos_yaml_sha256) == 64
    assert isinstance(cfg.repos, ReposManifest)
    assert isinstance(cfg.repos.repos[0], RepoConfig)


def test_load_app_config_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_app_config(_minimal_settings(missing))


def test_log_config_loaded_emits_structlog_event(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos: []
""",
    )
    configure_logging("INFO")
    app = load_app_config(_minimal_settings(p))
    log_config_loaded(app)
    out = capsys.readouterr().out
    assert "config_loaded" in out
    assert app.repos_yaml_sha256 in out


def test_settings_model_names_from_adr017() -> None:
    s = Settings(
        github_webhook_secret="a",
        github_pat="b",
        slack_bot_token="c",
        slack_signing_secret="d",
    )
    assert s.mechanical_model == "claude-haiku-4-5"
    assert s.reasoning_model == "claude-sonnet-4-6"
    assert s.high_stakes_model == "claude-opus-4-6"


def test_settings_secret_fields_have_local_dev_defaults() -> None:
    assert Settings.model_fields["github_webhook_secret"].default == "dev-local-github-webhook-secret"
    assert Settings.model_fields["github_pat"].default == "dev-local-github-pat"
    assert Settings.model_fields["slack_bot_token"].default == "xoxb-dev-local-placeholder"
    assert Settings.model_fields["slack_signing_secret"].default == "dev-local-slack-signing-secret"


def test_governance_defaults_to_none_when_block_absent(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: svc
    github_org: o
    github_repo: r
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
""",
    )
    manifest, _ = load_repos_manifest(p)
    assert manifest.repos[0].governance is None
    assert manifest.repos[0].approvers == []


def test_governance_block_parses_rules_and_approvers(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: svc
    github_org: o
    github_repo: r
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
    governance:
      auto_resolve:
        - severity: [informational, low]
        - duplicate: true
      notify:
        - severity: [medium]
        - ssvc_action: [attend, track]
      approve:
        - severity: [critical, high]
        - ssvc_action: [immediate]
        - poc_execution: true
    approvers:
      - slack_user: U12345AB
      - slack_user: U67890CD
""",
    )
    manifest, _ = load_repos_manifest(p)
    gov = manifest.repos[0].governance
    assert isinstance(gov, GovernanceConfig)
    assert gov.auto_resolve[0].severity == [Severity.informational, Severity.low]
    assert gov.auto_resolve[1].duplicate is True
    assert gov.notify[1].ssvc_action == [SSVCAction.attend, SSVCAction.track]
    assert gov.approve[2].poc_execution is True
    assert [a.slack_user for a in manifest.repos[0].approvers] == ["U12345AB", "U67890CD"]


def test_governance_rejects_empty_rule() -> None:
    with pytest.raises(ValidationError, match="at least one criterion"):
        GovernanceRule()


def test_governance_rejects_bad_slack_user(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: svc
    github_org: o
    github_repo: r
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
    approvers:
      - slack_user: alice
""",
    )
    with pytest.raises(ValueError, match="slack_user"):
        load_repos_manifest(p)


def test_governance_rejects_unknown_severity_value(tmp_path: Path) -> None:
    p = tmp_path / "repos.yaml"
    _write_manifest(
        p,
        """
repos:
  - name: svc
    github_org: o
    github_repo: r
    slack_channel: "#c"
    allowed_workflows: []
    notify_on_severity: [high]
    require_approval_for: [high]
    governance:
      approve:
        - severity: [armageddon]
""",
    )
    with pytest.raises(ValueError, match="armageddon"):
        load_repos_manifest(p)


def test_repo_config_accepted_risk_ttl_defaults_to_90() -> None:
    repo = RepoConfig(
        name="svc",
        github_org="o",
        github_repo="r",
        slack_channel="#c",
        allowed_workflows=[],
        notify_on_severity=["high"],
        require_approval_for=["high"],
    )
    assert repo.accepted_risk_ttl_days == 90


def test_repo_config_accepted_risk_ttl_rejects_negative() -> None:
    with pytest.raises(ValidationError, match="accepted_risk_ttl_days"):
        RepoConfig(
            name="svc",
            github_org="o",
            github_repo="r",
            slack_channel="#c",
            allowed_workflows=[],
            notify_on_severity=["high"],
            require_approval_for=["high"],
            accepted_risk_ttl_days=-1,
        )


def test_settings_tracker_credentials_default_none() -> None:
    s = Settings()
    assert s.jira_api_email is None
    assert s.jira_api_token is None
    assert s.linear_api_key is None

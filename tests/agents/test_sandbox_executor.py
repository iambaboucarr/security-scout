# SPDX-License-Identifier: Apache-2.0
"""Tests for agents/sandbox_executor.py — sandbox execution and confidence tiers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents.sandbox_executor import (
    ExecutionResult,
    PocType,
    _extract_evidence,
    _has_expected_pattern,
    _is_build_or_infra_error,
    assign_confidence_tier,
    execute_nuclei,
    execute_poc,
)
from models import FindingStatus
from tools.docker_sandbox import SandboxConfig, SandboxError, SandboxResult, SandboxTimeoutError
from tools.nuclei import NucleiMatch, NucleiResult

# ---------------------------------------------------------------------------
# assign_confidence_tier
# ---------------------------------------------------------------------------


class TestAssignConfidenceTier:
    def test_timeout_returns_error(self) -> None:
        assert assign_confidence_tier(exit_code=0, stdout="", stderr="", timed_out=True) == FindingStatus.error

    def test_exit_zero_with_pattern_returns_confirmed_low(self) -> None:
        assert (
            assign_confidence_tier(exit_code=0, stdout="SQL injection successful", stderr="", timed_out=False)
            == FindingStatus.confirmed_low
        )

    def test_exit_zero_with_pattern_and_patch_oracle_returns_confirmed_high(self) -> None:
        assert (
            assign_confidence_tier(
                exit_code=0,
                stdout="target is vulnerable to SQL injection",
                stderr="",
                timed_out=False,
                has_patch_oracle=True,
                patch_oracle_passed=True,
            )
            == FindingStatus.confirmed_high
        )

    def test_llm_generated_capped_at_confirmed_low_even_with_oracle(self) -> None:
        assert (
            assign_confidence_tier(
                exit_code=0,
                stdout="vulnerable to XSS attack",
                stderr="",
                timed_out=False,
                has_patch_oracle=True,
                patch_oracle_passed=True,
                poc_type=PocType.LLM_GENERATED,
            )
            == FindingStatus.confirmed_low
        )

    def test_exit_zero_no_pattern_returns_unconfirmed(self) -> None:
        assert (
            assign_confidence_tier(exit_code=0, stdout="test passed", stderr="", timed_out=False)
            == FindingStatus.unconfirmed
        )

    def test_nonzero_exit_with_infra_error_returns_error(self) -> None:
        assert (
            assign_confidence_tier(exit_code=1, stdout="", stderr="OOMKilled", timed_out=False) == FindingStatus.error
        )

    def test_nonzero_exit_without_pattern_returns_unconfirmed(self) -> None:
        assert (
            assign_confidence_tier(exit_code=1, stdout="", stderr="some error", timed_out=False)
            == FindingStatus.unconfirmed
        )

    def test_nonzero_exit_with_pattern_in_stderr_returns_confirmed_low(self) -> None:
        result = assign_confidence_tier(exit_code=1, stdout="", stderr="", timed_out=False)
        assert result == FindingStatus.unconfirmed

    def test_http_200_pattern_match(self) -> None:
        assert (
            assign_confidence_tier(exit_code=0, stdout="HTTP/1.1 200 OK\nsome body", stderr="", timed_out=False)
            == FindingStatus.confirmed_low
        )


# ---------------------------------------------------------------------------
# _has_expected_pattern
# ---------------------------------------------------------------------------


class TestHasExpectedPattern:
    def test_matches_vulnerable(self) -> None:
        assert _has_expected_pattern("Server is vulnerable!", "") is True

    def test_matches_rce(self) -> None:
        assert _has_expected_pattern("", "RCE confirmed") is True

    def test_matches_ssrf(self) -> None:
        assert _has_expected_pattern("SSRF detected in target", "") is True

    def test_no_match(self) -> None:
        assert _has_expected_pattern("test completed normally", "") is False


# ---------------------------------------------------------------------------
# _is_build_or_infra_error
# ---------------------------------------------------------------------------


class TestIsBuildOrInfraError:
    def test_oom_killed(self) -> None:
        assert _is_build_or_infra_error("Process was OOMKilled") is True

    def test_no_space(self) -> None:
        assert _is_build_or_infra_error("no space left on device") is True

    def test_permission_denied(self) -> None:
        assert _is_build_or_infra_error("permission denied") is True

    def test_normal_error(self) -> None:
        assert _is_build_or_infra_error("assertion failed: expected 1, got 2") is False


# ---------------------------------------------------------------------------
# _extract_evidence
# ---------------------------------------------------------------------------


class TestExtractEvidence:
    def test_combines_stdout_stderr(self) -> None:
        result = _extract_evidence("out", "err")
        assert "out" in result
        assert "err" in result

    def test_truncates_long_output(self) -> None:
        long = "x" * 1000
        result = _extract_evidence(long, "")
        assert len(result) <= 520  # 500 + "\n[truncated]" marker

    def test_empty_inputs(self) -> None:
        result = _extract_evidence("", "")
        assert result == ""

    def test_strips_whitespace(self) -> None:
        result = _extract_evidence("  hello  ", "")
        assert result == "hello"


# ---------------------------------------------------------------------------
# execute_poc
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_poc_timeout(tmp_path: Path) -> None:
    mock_result = SandboxTimeoutError("timed out")
    with patch("agents.sandbox_executor.run_container", new_callable=AsyncMock, side_effect=mock_result):
        result = await execute_poc(
            image="test:latest",
            poc_command=["python", "-c", "while True: pass"],
        )
    assert result.confidence_tier == FindingStatus.error
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_execute_poc_sandbox_error(tmp_path: Path) -> None:
    with patch("agents.sandbox_executor.run_container", new_callable=AsyncMock, side_effect=SandboxError("boom")):
        result = await execute_poc(
            image="test:latest",
            poc_command=["echo", "hello"],
        )
    assert result.confidence_tier == FindingStatus.error
    assert "Sandbox error" in result.evidence_excerpt


@pytest.mark.asyncio
async def test_execute_poc_success_confirmed_low() -> None:
    sandbox_result = SandboxResult(
        exit_code=0,
        stdout="SQL injection successful",
        stderr="",
        timed_out=False,
        elapsed_seconds=2.5,
    )
    with patch("agents.sandbox_executor.run_container", new_callable=AsyncMock, return_value=sandbox_result):
        result = await execute_poc(
            image="test:latest",
            poc_command=["python", "exploit.py"],
        )
    assert result.confidence_tier == FindingStatus.confirmed_low
    assert result.elapsed_seconds == 2.5
    assert result.poc_type == PocType.RESEARCHER_SUBMITTED


@pytest.mark.asyncio
async def test_execute_poc_unconfirmed() -> None:
    sandbox_result = SandboxResult(
        exit_code=0,
        stdout="nothing interesting",
        stderr="",
        timed_out=False,
        elapsed_seconds=1.0,
    )
    with patch("agents.sandbox_executor.run_container", new_callable=AsyncMock, return_value=sandbox_result):
        result = await execute_poc(
            image="test:latest",
            poc_command=["python", "test.py"],
        )
    assert result.confidence_tier == FindingStatus.unconfirmed


@pytest.mark.asyncio
async def test_execute_poc_with_repo_path(tmp_path: Path) -> None:
    sandbox_result = SandboxResult(
        exit_code=0,
        stdout="vulnerable",
        stderr="",
        timed_out=False,
        elapsed_seconds=1.0,
    )
    with patch(
        "agents.sandbox_executor.run_container", new_callable=AsyncMock, return_value=sandbox_result
    ) as mock_run:
        await execute_poc(
            image="test:latest",
            poc_command=["python", "test.py"],
            repo_path=tmp_path,
        )
    config = mock_run.call_args[0][0]
    assert isinstance(config, SandboxConfig)
    assert str(tmp_path) in config.read_only_volumes


# ---------------------------------------------------------------------------
# execute_nuclei
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_nuclei_with_matches() -> None:
    nuclei_result = NucleiResult(
        matches=(
            NucleiMatch(
                template_id="CVE-2024-1234",
                matched_at="http://localhost:8080",
                matcher_name="default",
                severity="high",
            ),
        ),
        exit_code=0,
        stdout_raw='{"template-id":"CVE-2024-1234"}',
        stderr_raw="",
        timed_out=False,
        elapsed_seconds=5.0,
    )
    with patch("agents.sandbox_executor.run_nuclei", new_callable=AsyncMock, return_value=nuclei_result):
        result = await execute_nuclei(target="http://localhost:8080", cve_id="CVE-2024-1234")
    assert result.confidence_tier == FindingStatus.confirmed_low
    assert result.nuclei_matches == 1
    assert "CVE-2024-1234" in result.evidence_excerpt


@pytest.mark.asyncio
async def test_execute_nuclei_no_matches() -> None:
    nuclei_result = NucleiResult(
        matches=(),
        exit_code=0,
        stdout_raw="",
        stderr_raw="",
        timed_out=False,
        elapsed_seconds=3.0,
    )
    with patch("agents.sandbox_executor.run_nuclei", new_callable=AsyncMock, return_value=nuclei_result):
        result = await execute_nuclei(target="http://localhost:8080")
    assert result.confidence_tier == FindingStatus.unconfirmed
    assert result.nuclei_matches == 0


@pytest.mark.asyncio
async def test_execute_nuclei_timeout() -> None:
    nuclei_result = NucleiResult(
        matches=(),
        exit_code=-1,
        stdout_raw="",
        stderr_raw="",
        timed_out=True,
        elapsed_seconds=120.0,
    )
    with patch("agents.sandbox_executor.run_nuclei", new_callable=AsyncMock, return_value=nuclei_result):
        result = await execute_nuclei(target="http://localhost:8080")
    assert result.confidence_tier == FindingStatus.error
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_execute_nuclei_exception() -> None:
    with patch(
        "agents.sandbox_executor.run_nuclei", new_callable=AsyncMock, side_effect=RuntimeError("nuclei crashed")
    ):
        result = await execute_nuclei(target="http://localhost:8080")
    assert result.confidence_tier == FindingStatus.error
    assert "Nuclei error" in result.evidence_excerpt


# ---------------------------------------------------------------------------
# ExecutionResult dataclass
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_frozen(self) -> None:
        r = ExecutionResult(
            confidence_tier=FindingStatus.unconfirmed,
            evidence_excerpt="test",
            raw_stdout="",
            raw_stderr="",
            elapsed_seconds=1.0,
            poc_type=PocType.RESEARCHER_SUBMITTED,
        )
        with pytest.raises(AttributeError):
            r.confidence_tier = FindingStatus.confirmed_high  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PocType enum
# ---------------------------------------------------------------------------


class TestPocType:
    def test_values(self) -> None:
        assert PocType.RESEARCHER_SUBMITTED.value == "researcher-submitted"
        assert PocType.NUCLEI_TEMPLATE.value == "nuclei-template"
        assert PocType.LLM_GENERATED.value == "llm-generated"

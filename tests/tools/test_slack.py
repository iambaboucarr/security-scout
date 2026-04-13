from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import httpx
import pytest

from models import Finding, FindingStatus, KnownStatus, Severity, WorkflowKind
from tools.slack import (
    ACTION_ID_APPROVE,
    ACTION_ID_DEDUP_CONFIRM,
    ACTION_ID_DEDUP_NEW_INSTANCE,
    ACTION_ID_DEDUP_REOPEN,
    ACTION_ID_DEDUP_RESOLVED,
    ACTION_ID_ESCALATE,
    ACTION_ID_REJECT,
    ACTION_ID_RISK_REEVALUATE,
    ACTION_ID_RISK_STILL_ACCEPTED,
    ApprovalButtonContext,
    DedupMatchInfo,
    FindingReportPayload,
    SlackAPIError,
    SlackClient,
    SlackMalformedResponseError,
    build_finding_blocks,
    escape_slack_mrkdwn,
    fallback_notification_text,
    finding_to_report_payload,
)


def _sample_report(**kwargs: object) -> FindingReportPayload:
    base = {
        "finding_id": uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        "title": "Test vulnerability in auth",
        "severity": "high",
        "ssvc_action": "act",
        "confidence": 0.91,
        "source_url": "https://github.com/o/r/security/advisories/GHSA-ABCD-EFGH-IJKL",
        "affected_versions": "1.0.0 - 1.2.3",
        "cve_ids": ("CVE-2024-9999",),
        "cwe_ids": ("CWE-89",),
        "cvss_score": 8.2,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "description_excerpt": "SQL injection in login handler.",
    }
    base.update(kwargs)
    return FindingReportPayload.model_validate(base)


def test_escape_slack_mrkdwn_escapes_special_chars() -> None:
    assert escape_slack_mrkdwn("a & b < c >") == "a &amp; b &lt; c &gt;"


def test_fallback_notification_text_includes_severity_and_url() -> None:
    r = _sample_report()
    text = fallback_notification_text(r)
    assert "[HIGH]" in text.upper()
    assert "Test vulnerability" in text
    assert "https://github.com/o/r/security/advisories/" in text


def test_fallback_notification_text_is_plain_text_not_mrkdwn_entities() -> None:
    """``text`` is plain — must not use HTML entity escaping (would show ``&amp;`` literally)."""

    r = _sample_report(title="Tom & Jerry <script>")
    text = fallback_notification_text(r)
    assert "Tom & Jerry" in text
    assert "&amp;" not in text
    assert "&lt;" not in text


def test_fallback_notification_text_collapses_multiline_title() -> None:
    r = _sample_report(title="One\nTwo  \nThree")
    text = fallback_notification_text(r)
    assert "\n" not in text
    assert "One Two Three" in text


def test_build_finding_blocks_header_collapses_newlines() -> None:
    r = _sample_report(title="A\nB")
    blocks = build_finding_blocks(r)
    header = next(b for b in blocks if b.get("type") == "header")
    assert "\n" not in header["text"]["text"]
    assert "A B" in header["text"]["text"]


def test_build_finding_blocks_encodes_pipe_in_source_url_for_slack_links() -> None:
    r = _sample_report(source_url="https://example.com/path?a=1|2")
    blocks = build_finding_blocks(r)
    dumped = json.dumps(blocks)
    assert "%7C" in dumped


def test_build_finding_blocks_confidence_nan_shows_dash() -> None:
    r = _sample_report(confidence=float("nan"))
    blocks = build_finding_blocks(r)
    fields_block = next(b for b in blocks if b.get("type") == "section" and b.get("fields"))
    texts = [f["text"] for f in fields_block["fields"]]
    conf_field = next(t for t in texts if t.startswith("*Confidence*"))
    assert "nan" not in conf_field.lower()
    assert "—" in conf_field


def test_build_finding_blocks_has_header_and_core_fields() -> None:
    r = _sample_report()
    blocks = build_finding_blocks(r)
    assert len(blocks) >= 5
    types = [b.get("type") for b in blocks]
    assert "header" in types
    assert types.count("divider") >= 2
    header = next(b for b in blocks if b.get("type") == "header")
    assert "HIGH" in header["text"]["text"]
    assert "Test vulnerability" in header["text"]["text"]


def test_build_finding_blocks_truncates_long_evidence() -> None:
    long_ev = "e" * 600
    r = _sample_report(evidence_excerpt=long_ev)
    blocks = build_finding_blocks(r)
    section_texts = [
        b["text"]["text"]
        for b in blocks
        if b.get("type") == "section"
        and isinstance(b.get("text"), dict)
        and "Evidence excerpt" in b["text"].get("text", "")
    ]
    assert section_texts
    assert len(section_texts[0]) < 600
    assert section_texts[0].endswith("…")


def test_finding_to_report_payload_maps_core_fields() -> None:
    fid = uuid.uuid4()
    f = Finding(
        id=fid,
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-TEST",
        severity=Severity.high,
        status=FindingStatus.unconfirmed,
        title="Test title",
        cve_id="CVE-2024-1",
        cwe_ids=["CWE-79"],
        evidence={"ok": True},
    )
    p = finding_to_report_payload(f)
    assert p.finding_id == fid
    assert p.title == "Test title"
    assert p.severity == "high"
    assert p.cve_ids == ("CVE-2024-1",)
    assert p.cwe_ids == ("CWE-79",)
    assert p.evidence_excerpt is not None
    assert '"ok": true' in p.evidence_excerpt


def test_finding_to_report_payload_omits_evidence_when_not_serializable() -> None:
    f = Finding(
        id=uuid.uuid4(),
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-TEST",
        severity=Severity.high,
        status=FindingStatus.unconfirmed,
        title="T",
        evidence={"x": object()},
    )
    p = finding_to_report_payload(f)
    assert p.evidence_excerpt is None


def test_build_finding_blocks_includes_dedup_section() -> None:
    r = _sample_report(
        dedup=DedupMatchInfo(
            tier=1,
            tracker_name="github_issues",
            match_url="https://github.com/o/r/issues/42",
            duplicate_of="GHSA-ABCD-EFGH-IJKL",
        ),
    )
    blocks = build_finding_blocks(r)
    dumped = json.dumps(blocks)
    assert "Known vulnerability match" in dumped
    assert "tier 1" in dumped
    assert "github_issues" in dumped


def test_build_finding_blocks_includes_workflow_run_in_context() -> None:
    wid = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    r = _sample_report()
    blocks = build_finding_blocks(r, workflow_run_id=wid)
    ctx = next(b for b in blocks if b.get("type") == "context")
    inner = ctx["elements"][0]["text"]
    assert str(wid) in inner


def _slack_transport(ok: bool) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url).endswith("/chat.postMessage")
        body = json.loads(request.content.decode())
        assert body["channel"] == "C01234567"
        assert "blocks" in body
        assert body["text"]
        if ok:
            return httpx.Response(
                200,
                json={"ok": True, "channel": "C01234567", "ts": "1234.5678"},
            )
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_send_finding_success() -> None:
    r = _sample_report()
    async with (
        httpx.AsyncClient(
            base_url="https://slack.com/api",
            transport=_slack_transport(True),
        ) as client,
        SlackClient("xoxb-test-token", client=client) as slack,
    ):
        out = await slack.send_finding("C01234567", r, workflow_run_id=uuid.uuid4())

    assert out.channel == "C01234567"
    assert out.message_ts == "1234.5678"


@pytest.mark.asyncio
async def test_send_finding_slack_error_raises() -> None:
    r = _sample_report()
    async with (
        httpx.AsyncClient(
            base_url="https://slack.com/api",
            transport=_slack_transport(False),
        ) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        with pytest.raises(SlackAPIError, match="channel_not_found"):
            await slack.send_finding("C01234567", r)


@pytest.mark.asyncio
async def test_send_finding_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    r = _sample_report()
    async with (
        httpx.AsyncClient(
            base_url="https://slack.com/api",
            transport=httpx.MockTransport(handler),
        ) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        with pytest.raises(SlackAPIError) as ei:
            await slack.send_finding("C01234567", r)
        assert ei.value.http_status == 503


@pytest.mark.asyncio
async def test_send_finding_missing_ts_raises_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "channel": "C01"})

    r = _sample_report()
    async with (
        httpx.AsyncClient(
            base_url="https://slack.com/api",
            transport=httpx.MockTransport(handler),
        ) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        with pytest.raises(SlackMalformedResponseError, match="missing"):
            await slack.send_finding("C01234567", r)


@pytest.mark.asyncio
async def test_send_finding_logs_metric_on_success() -> None:
    r = _sample_report()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": True, "channel": "C01234567", "ts": "99.99"},
        )

    async with (
        httpx.AsyncClient(
            base_url="https://slack.com/api",
            transport=httpx.MockTransport(handler),
        ) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        with patch("tools.slack._LOG") as mock_log:
            await slack.send_finding("C01234567", r)
        mock_log.info.assert_called_once()
        call_kw = mock_log.info.call_args[1]
        assert call_kw.get("metric_name") == "slack_finding_posted"


def test_slack_client_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="token"):
        SlackClient("")


def test_build_finding_blocks_with_approval_context_adds_three_buttons() -> None:
    r = _sample_report()
    ctx = ApprovalButtonContext(
        finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
        repo_name="demo",
    )
    blocks = build_finding_blocks(r, approval_context=ctx)
    actions = [b for b in blocks if b.get("type") == "actions"]
    assert len(actions) == 1
    elements = actions[0]["elements"]
    action_ids = [e["action_id"] for e in elements]
    assert action_ids == [ACTION_ID_APPROVE, ACTION_ID_REJECT, ACTION_ID_ESCALATE]
    encoded = ctx.encode()
    for element in elements:
        assert element["value"] == encoded


def test_build_finding_blocks_without_approval_context_has_no_buttons() -> None:
    r = _sample_report()
    blocks = build_finding_blocks(r)
    assert not any(b.get("type") == "actions" for b in blocks)


def test_build_finding_blocks_with_dedup_and_approval_context_adds_dedup_buttons() -> None:
    r = _sample_report(
        dedup=DedupMatchInfo(
            tier=1,
            tracker_name="jira",
            match_url="https://acme.atlassian.net/browse/SEC-1",
            duplicate_of="SEC-1",
        ),
    )
    ctx = ApprovalButtonContext(
        finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
        repo_name="demo",
    )
    blocks = build_finding_blocks(r, approval_context=ctx)
    actions = [b for b in blocks if b.get("type") == "actions"]
    # Two action blocks: dedup buttons (4) + approval buttons (3).
    assert len(actions) == 2
    dedup_block = next(b for b in actions if b.get("block_id") == "security_scout_dedup_actions")
    approval_block = next(b for b in actions if b.get("block_id") == "security_scout_actions")
    assert [e["action_id"] for e in dedup_block["elements"]] == [
        ACTION_ID_DEDUP_CONFIRM,
        ACTION_ID_DEDUP_NEW_INSTANCE,
        ACTION_ID_DEDUP_REOPEN,
        ACTION_ID_DEDUP_RESOLVED,
    ]
    assert [e["action_id"] for e in approval_block["elements"]] == [
        ACTION_ID_APPROVE,
        ACTION_ID_REJECT,
        ACTION_ID_ESCALATE,
    ]
    encoded = ctx.encode()
    for element in dedup_block["elements"]:
        assert element["value"] == encoded


def test_build_finding_blocks_with_dedup_but_no_approval_context_has_no_dedup_buttons() -> None:
    r = _sample_report(
        dedup=DedupMatchInfo(
            tier=1,
            tracker_name="jira",
            duplicate_of="SEC-1",
        ),
    )
    blocks = build_finding_blocks(r)
    assert not any(b.get("type") == "actions" for b in blocks)


def test_build_finding_blocks_accepted_risk_replaces_approval_buttons_with_risk_buttons() -> None:
    r = _sample_report(
        dedup=DedupMatchInfo(
            tier=1,
            tracker_name="scout_history",
            duplicate_of="prior-finding-uuid",
            is_accepted_risk=True,
        ),
    )
    ctx = ApprovalButtonContext(
        finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
        repo_name="demo",
    )
    blocks = build_finding_blocks(r, approval_context=ctx)
    actions = [b for b in blocks if b.get("type") == "actions"]
    assert len(actions) == 1
    assert actions[0].get("block_id") == "security_scout_risk_actions"
    assert [e["action_id"] for e in actions[0]["elements"]] == [
        ACTION_ID_RISK_STILL_ACCEPTED,
        ACTION_ID_RISK_REEVALUATE,
    ]


def test_build_finding_blocks_accepted_risk_section_renders_label() -> None:
    r = _sample_report(
        dedup=DedupMatchInfo(
            tier=1,
            tracker_name="scout_history",
            duplicate_of="prior",
            is_accepted_risk=True,
        ),
    )
    blocks = build_finding_blocks(r)
    dumped = json.dumps(blocks)
    assert "Previously Accepted Risk" in dumped
    assert "Known vulnerability match" not in dumped


def test_finding_to_report_payload_marks_accepted_risk_when_known_status_is_accepted_risk() -> None:
    f = Finding(
        id=uuid.uuid4(),
        workflow=WorkflowKind.advisory,
        source_ref="https://github.com/advisories/GHSA-TEST",
        severity=Severity.high,
        status=FindingStatus.unconfirmed,
        title="t",
        duplicate_of="prior-finding-uuid",
        duplicate_tracker="scout_history",
        known_status=KnownStatus.known_accepted_risk,
    )
    p = finding_to_report_payload(f)
    assert p.dedup is not None
    assert p.dedup.is_accepted_risk is True


def test_build_finding_blocks_informational_adds_badge() -> None:
    r = _sample_report()
    blocks = build_finding_blocks(r, informational=True)
    dumped = json.dumps(blocks)
    assert "Informational" in dumped
    assert not any(b.get("type") == "actions" for b in blocks)


def test_approval_button_context_roundtrip() -> None:
    ctx = ApprovalButtonContext(
        finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
        repo_name="demo",
    )
    restored = ApprovalButtonContext.decode(ctx.encode())
    assert restored == ctx


def test_approval_button_context_rejects_pipe_in_repo_name() -> None:
    with pytest.raises(ValueError, match="\\|"):
        ApprovalButtonContext(
            finding_id=uuid.uuid4(),
            workflow_run_id=uuid.uuid4(),
            repo_name="bad|name",
        ).encode()


def test_approval_button_context_decode_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3 pipe-separated"):
        ApprovalButtonContext.decode("only-two|fields")


@pytest.mark.asyncio
async def test_send_finding_for_approval_posts_buttons() -> None:
    r = _sample_report()
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"ok": True, "channel": "C01234567", "ts": "9.9"})

    async with (
        httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(handler)) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        ctx = ApprovalButtonContext(
            finding_id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
            workflow_run_id=uuid.UUID("66666666-7777-8888-9999-aaaaaaaaaaaa"),
            repo_name="demo",
        )
        await slack.send_finding_for_approval(
            "C01234567",
            r,
            workflow_run_id=ctx.workflow_run_id,
            approval_context=ctx,
        )

    assert len(captured) == 1
    body = captured[0]
    blocks = body["blocks"]
    assert any(b.get("type") == "actions" for b in blocks)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_post_thread_reply_sends_thread_ts() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"ok": True, "channel": "C01", "ts": "2.0"})

    async with (
        httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(handler)) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        await slack.post_thread_reply("C01", thread_ts="1.0", text="hi")

    assert captured[0]["thread_ts"] == "1.0"
    assert captured[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_send_dm_uses_user_id_as_channel() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"ok": True, "channel": "D01", "ts": "3.0"})

    async with (
        httpx.AsyncClient(base_url="https://slack.com/api", transport=httpx.MockTransport(handler)) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        await slack.send_dm("U123456", text="escalated")

    assert captured[0]["channel"] == "U123456"
    assert captured[0]["text"] == "escalated"


@pytest.mark.asyncio
async def test_send_finding_rejects_empty_channel() -> None:
    r = _sample_report()
    async with (
        httpx.AsyncClient(
            base_url="https://slack.com/api",
            transport=httpx.MockTransport(lambda _req: httpx.Response(500)),
        ) as client,
        SlackClient("xoxb-test", client=client) as slack,
    ):
        with pytest.raises(ValueError, match="channel"):
            await slack.send_finding("", r)

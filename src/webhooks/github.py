from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from config import AppConfig, RepoConfig, Settings

_LOG = structlog.get_logger(__name__)

_REPLAY_WINDOW_SEC = 300
_MAX_FUTURE_SKEW_SEC = 60


def verify_github_hub_signature_256(raw_body: bytes, secret: str, signature_header: str | None) -> None:
    """Validate ``X-Hub-Signature-256`` per GitHub Docs (HMAC-SHA256 over raw UTF-8 body)."""
    if not signature_header:
        raise HTTPException(status_code=401, detail="missing X-Hub-Signature-256")
    mac = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="invalid webhook signature")


def assert_delivery_fresh_http_date(
    date_header: str | None,
    *,
    now: datetime,
    replay_window_sec: int = _REPLAY_WINDOW_SEC,
) -> None:
    """Reject replays when ``Date`` is present; if absent, log and continue"""
    if not date_header:
        _LOG.info(
            "webhook_delivery_date_freshness_skipped",
            metric_name="webhook_delivery_date_freshness_skipped",
            reason="no_http_date_header",
        )
        return
    try:
        parsed = parsedate_to_datetime(date_header)
    except TypeError, ValueError:
        raise HTTPException(status_code=401, detail="invalid HTTP Date header") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    skew = (now - parsed.astimezone(UTC)).total_seconds()
    if skew < -_MAX_FUTURE_SKEW_SEC:
        raise HTTPException(status_code=401, detail="webhook Date too far in the future")
    if skew > replay_window_sec:
        raise HTTPException(status_code=401, detail="stale webhook delivery")


def find_repo_config(app: AppConfig, *, owner: str, repo: str) -> RepoConfig | None:
    for r in app.repos.repos:
        if r.github_org == owner and r.github_repo == repo:
            return r
    return None


def _split_full_name(full_name: str | None) -> tuple[str, str] | None:
    if not full_name or "/" not in full_name:
        return None
    owner, name = full_name.split("/", 1)
    if not owner or not name:
        return None
    return owner, name


def _ghsa_from_payload(event: str, payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(ghsa_id, skip_reason)``. ``skip_reason`` set when we intentionally no-op."""
    if event == "repository_advisory":
        ra = payload.get("repository_advisory")
        if not isinstance(ra, dict):
            return None, "missing_repository_advisory"
        ghsa = ra.get("ghsa_id")
        if isinstance(ghsa, str) and ghsa:
            return ghsa.upper(), None
        return None, "missing_ghsa_id"

    if event == "dependabot_alert":
        alert = payload.get("alert")
        if not isinstance(alert, dict):
            return None, "missing_alert"
        sa = alert.get("security_advisory")
        if not isinstance(sa, dict):
            return None, "missing_security_advisory"
        ghsa = sa.get("ghsa_id")
        if isinstance(ghsa, str) and ghsa:
            return ghsa.upper(), None
        return None, "missing_ghsa_id"

    if event == "security_advisory":
        return None, "global_security_advisory_deferred"

    return None, f"event_not_handled:{event}"


def _owner_repo_from_repository_dict(repo_obj: dict[str, Any]) -> tuple[str, str] | None:
    full_name = repo_obj.get("full_name")
    parsed = _split_full_name(full_name) if isinstance(full_name, str) else None
    if parsed is not None:
        return parsed
    owner_login = (repo_obj.get("owner") or {}).get("login") if isinstance(repo_obj.get("owner"), dict) else None
    name = repo_obj.get("name")
    if isinstance(owner_login, str) and isinstance(name, str):
        return (owner_login, name)
    return None


def _github_webhook_special_event_response(
    event: str,
    payload: dict[str, Any],
    log: structlog.BoundLogger,
) -> Response | None:
    if event == "ping":
        log.info("github_webhook_ping", metric_name="github_webhook_received_total")
        return Response(status_code=200)
    if event == "pull_request":
        log.info(
            "github_webhook_pr_deferred",
            metric_name="github_webhook_pr_deferred_total",
            action=payload.get("action"),
        )
        return Response(status_code=202)
    if event == "security_advisory":
        log.info(
            "github_webhook_global_advisory_skipped",
            metric_name="github_webhook_noop_total",
            action=payload.get("action"),
        )
        return Response(status_code=202)
    return None


async def _github_webhook_advisory_response(
    event: str,
    payload: dict[str, Any],
    log: structlog.BoundLogger,
    app_config: AppConfig,
    enqueue_advisory: Callable[..., Awaitable[str | None]],
) -> Response:
    ghsa, skip_reason = _ghsa_from_payload(event, payload)
    if skip_reason:
        log.info(
            "github_webhook_noop",
            metric_name="github_webhook_noop_total",
            reason=skip_reason,
        )
        return Response(status_code=202)

    if ghsa is None:
        log.warning("github_webhook_missing_ghsa", metric_name="github_webhook_noop_total")
        return Response(status_code=202)

    repo_obj = payload.get("repository")
    if not isinstance(repo_obj, dict):
        log.warning("github_webhook_missing_repository", metric_name="github_webhook_noop_total")
        return Response(status_code=202)

    parsed = _owner_repo_from_repository_dict(repo_obj)
    if parsed is None:
        log.warning("github_webhook_repo_unparsed", metric_name="github_webhook_noop_total")
        return Response(status_code=202)

    owner, repo_name = parsed
    cfg = find_repo_config(app_config, owner=owner, repo=repo_name)
    if cfg is None:
        log.info(
            "github_webhook_unknown_repo",
            metric_name="github_webhook_unknown_repo_total",
            github_org=owner,
            github_repo=repo_name,
        )
        return Response(status_code=202)

    advisory_source: Literal["repository", "global"] = "repository"
    job_id = await enqueue_advisory(
        repo_name=cfg.name,
        ghsa_id=ghsa,
        advisory_source=advisory_source,
    )
    log.info(
        "github_webhook_advisory_enqueued",
        metric_name="github_webhook_advisory_enqueued_total",
        repo=cfg.name,
        ghsa_id=ghsa,
        arq_job_id=job_id,
    )
    return Response(status_code=202)


async def github_webhook(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    app_config: AppConfig = request.app.state.app_config
    enqueue_advisory = request.app.state.enqueue_advisory

    raw = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    verify_github_hub_signature_256(raw, settings.github_webhook_secret, sig)
    assert_delivery_fresh_http_date(request.headers.get("Date"), now=datetime.now(UTC))

    event = request.headers.get("X-GitHub-Event", "")
    delivery = request.headers.get("X-GitHub-Delivery", "")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "invalid JSON body"})

    log = _LOG.bind(
        github_event=event,
        github_delivery_id=delivery or None,
    )

    special = _github_webhook_special_event_response(event, payload, log)
    if special is not None:
        return special
    return await _github_webhook_advisory_response(
        event,
        payload,
        log,
        app_config,
        enqueue_advisory,
    )


def create_github_webhook_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/webhooks/github", github_webhook, methods=["POST"])
    return router

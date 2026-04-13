from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Request, Response

from config import AppConfig, RepoConfig, Settings
from webhooks.scm.github import GitHubWebhookProvider
from webhooks.scm.protocol import WebhookEvent, WebhookVerificationError

_LOG = structlog.get_logger(__name__)

_WEBHOOK_PROVIDER = GitHubWebhookProvider()


def find_repo_config(app: AppConfig, *, owner: str, repo: str) -> RepoConfig | None:
    for r in app.repos.repos:
        if r.github_org == owner and r.github_repo == repo:
            return r
    return None


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
    webhook_event: WebhookEvent,
    log: structlog.BoundLogger,
    app_config: AppConfig,
    enqueue_advisory: Callable[..., Awaitable[str | None]],
) -> Response:
    ghsa, skip_reason = _ghsa_from_payload(webhook_event.event_type, webhook_event.payload)
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

    owner = webhook_event.repo_owner
    repo_name = webhook_event.repo_name
    if owner is None or repo_name is None:
        log.warning("github_webhook_repo_unparsed", metric_name="github_webhook_noop_total")
        return Response(status_code=202)

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
    headers = dict(request.headers)
    try:
        _WEBHOOK_PROVIDER.verify_signature(raw, headers, settings.github_webhook_secret)
        _WEBHOOK_PROVIDER.assert_delivery_fresh(headers, now=datetime.now(UTC))
    except WebhookVerificationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        webhook_event = _WEBHOOK_PROVIDER.parse_event(raw, headers)
    except WebhookVerificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log = _LOG.bind(
        github_event=webhook_event.event_type,
        github_delivery_id=webhook_event.delivery_id or None,
    )

    special = _github_webhook_special_event_response(webhook_event.event_type, webhook_event.payload, log)
    if special is not None:
        return special
    return await _github_webhook_advisory_response(
        webhook_event,
        log,
        app_config,
        enqueue_advisory,
    )


def create_github_webhook_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/webhooks/github", github_webhook, methods=["POST"])
    return router

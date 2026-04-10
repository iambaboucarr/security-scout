"""Inbound webhooks (GitHub, Slack Phase 2)."""

from webhooks.github import create_github_webhook_router

__all__ = ["create_github_webhook_router"]

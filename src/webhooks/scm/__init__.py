"""SCM webhook provider abstraction (ADR-027)."""

from webhooks.scm.protocol import WebhookEvent, WebhookProvider

__all__ = ["WebhookEvent", "WebhookProvider"]

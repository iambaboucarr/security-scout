from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from exceptions import SecurityScoutError


class WebhookVerificationError(SecurityScoutError):
    """Webhook signature or freshness check failed."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message, is_transient=False, is_resource_error=False)


@dataclass(frozen=True, slots=True)
class WebhookEvent:
    """Parsed inbound webhook event, platform-agnostic."""

    event_type: str
    delivery_id: str
    action: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    repo_owner: str | None = None
    repo_name: str | None = None


@runtime_checkable
class WebhookProvider(Protocol):
    """Platform-agnostic webhook verification and parsing."""

    def verify_signature(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> None:
        """Validate the webhook signature.  Raises ``WebhookVerificationError`` on failure."""
        ...

    def assert_delivery_fresh(
        self,
        headers: Mapping[str, str],
        *,
        now: datetime,
    ) -> None:
        """Reject replayed deliveries.  Raises ``WebhookVerificationError`` if stale."""
        ...

    def parse_event(
        self,
        raw_body: bytes,
        headers: Mapping[str, str],
    ) -> WebhookEvent:
        """Extract a ``WebhookEvent`` from raw bytes and headers."""
        ...


__all__ = [
    "WebhookEvent",
    "WebhookProvider",
    "WebhookVerificationError",
]

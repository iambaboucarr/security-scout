# SPDX-License-Identifier: Apache-2.0
"""SCM webhook provider abstraction."""

from webhooks.scm.protocol import WebhookEvent, WebhookProvider

__all__ = ["WebhookEvent", "WebhookProvider"]

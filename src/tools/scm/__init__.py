# SPDX-License-Identifier: Apache-2.0
"""SCM provider abstraction.

Agents import from this package, not from ``tools.github`` directly.
"""

from tools.scm.models import (
    AdvisoryData,
    IssueSearchItem,
    PullRequestFileInfo,
    PullRequestInfo,
    RepositoryMetadata,
    normalise_ghsa_id,
)
from tools.scm.protocol import DiffData, SCMProvider

__all__ = [
    "AdvisoryData",
    "DiffData",
    "IssueSearchItem",
    "PullRequestFileInfo",
    "PullRequestInfo",
    "RepositoryMetadata",
    "SCMProvider",
    "normalise_ghsa_id",
]

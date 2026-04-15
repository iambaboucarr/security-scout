# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

_GHSA_ID_PATTERN = re.compile(r"^GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}$", re.IGNORECASE)


def normalise_ghsa_id(raw: str) -> str:
    """Return canonical ``GHSA-xxxx-xxxx-xxxx`` (raises if shape is invalid)."""
    s = raw.strip()
    if not _GHSA_ID_PATTERN.match(s):
        msg = f"invalid GHSA id: {raw!r}"
        raise ValueError(msg)
    parts = s.split("-")
    return f"GHSA-{'-'.join(p.upper() for p in parts[1:])}"


class AdvisoryData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ghsa_id: str
    source: Literal["repository", "global"]
    summary: str
    description: str
    severity: str | None = None
    cve_ids: tuple[str, ...] = ()
    cwe_ids: tuple[str, ...] = ()
    html_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    cvss_vector: str | None = None
    cvss_score_api: float | None = None
    affected_package_name: str | None = None
    affected_package_ecosystem: str | None = None


class PullRequestInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    state: str
    head_sha: str
    base_sha: str
    user_login: str | None = None
    html_url: str
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


class PullRequestFileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    status: str
    patch: str | None = None
    additions: int = 0
    deletions: int = 0
    sha: str | None = None


class IssueSearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    html_url: str
    state: str
    updated_at: datetime | None = None
    body: str | None = None


class RepositoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    full_name: str
    description: str | None = None
    default_branch: str
    private: bool
    html_url: str
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    language: str | None = None
    pushed_at: datetime | None = None


__all__ = [
    "AdvisoryData",
    "IssueSearchItem",
    "PullRequestFileInfo",
    "PullRequestInfo",
    "RepositoryMetadata",
    "normalise_ghsa_id",
]

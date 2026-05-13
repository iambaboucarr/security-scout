# SPDX-License-Identifier: Apache-2.0
"""Version 1 of the authenticated HTTP API.

Routers live under ``/api/v1``. The mount point and version segment are
fixed here so future API versions can be added as sibling packages
without forcing every caller to rediscover the prefix.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.v1.findings import create_findings_router

API_V1_PREFIX = "/api/v1"


def create_v1_router() -> APIRouter:
    router = APIRouter(prefix=API_V1_PREFIX)
    router.include_router(create_findings_router())
    return router


__all__ = ["API_V1_PREFIX", "create_v1_router"]

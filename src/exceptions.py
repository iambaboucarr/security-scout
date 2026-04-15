# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid


class SecurityScoutError(Exception):
    def __init__(
        self,
        message: str | None = None,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
        is_transient: bool = False,
        is_resource_error: bool = False,
    ) -> None:
        super().__init__(message if message is not None else "")
        self.finding_id = finding_id
        self.workflow_run_id = workflow_run_id
        self._is_transient = is_transient
        self._is_resource_error = is_resource_error

    @property
    def is_transient(self) -> bool:
        return self._is_transient

    @property
    def is_resource_error(self) -> bool:
        return self._is_resource_error


class TransientError(SecurityScoutError):
    def __init__(
        self,
        message: str | None = None,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=True,
            is_resource_error=False,
        )


class PermanentError(SecurityScoutError):
    def __init__(
        self,
        message: str | None = None,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=False,
            is_resource_error=False,
        )


class ResourceError(SecurityScoutError):
    """Pipeline may continue; used when a step is inconclusive (e.g. sandbox ERROR tier)."""

    def __init__(
        self,
        message: str | None = None,
        *,
        finding_id: str | None = None,
        workflow_run_id: uuid.UUID | str | None = None,
    ) -> None:
        super().__init__(
            message,
            finding_id=finding_id,
            workflow_run_id=workflow_run_id,
            is_transient=False,
            is_resource_error=True,
        )

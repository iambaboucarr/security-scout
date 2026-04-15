# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid

from exceptions import (
    PermanentError,
    ResourceError,
    SecurityScoutError,
    TransientError,
)


def test_security_scout_error_default_flags() -> None:
    err = SecurityScoutError("x")
    assert str(err) == "x"
    assert err.is_transient is False
    assert err.is_resource_error is False
    assert err.finding_id is None
    assert err.workflow_run_id is None


def test_security_scout_error_optional_fields() -> None:
    wid = uuid.uuid4()
    err = SecurityScoutError(
        "m",
        finding_id="f1",
        workflow_run_id=wid,
        is_transient=True,
    )
    assert err.finding_id == "f1"
    assert err.workflow_run_id == wid
    assert err.is_transient is True


def test_security_scout_error_empty_message() -> None:
    err = SecurityScoutError(None)
    assert str(err) == ""


def test_transient_error_is_transient() -> None:
    err = TransientError("retry")
    assert err.is_transient is True
    assert err.is_resource_error is False


def test_permanent_error_is_not_transient() -> None:
    err = PermanentError("stop")
    assert err.is_transient is False
    assert err.is_resource_error is False


def test_resource_error_flags() -> None:
    err = ResourceError("inconclusive")
    assert err.is_transient is False
    assert err.is_resource_error is True


def test_exception_chaining() -> None:
    cause = ValueError("root")
    err = SecurityScoutError("wrapped")
    err.__cause__ = cause
    assert err.__cause__ is cause


def test_subclass_relationships() -> None:
    assert issubclass(TransientError, SecurityScoutError)
    assert issubclass(PermanentError, SecurityScoutError)
    assert issubclass(ResourceError, SecurityScoutError)
    assert not issubclass(TransientError, PermanentError)

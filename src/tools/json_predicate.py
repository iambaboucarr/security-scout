# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from typing import Any, NoReturn, cast

from sqlalchemy import String, func
from sqlalchemy.exc import CompileError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import ColumnElement
from sqlalchemy.sql.functions import FunctionElement

_PATH_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")


class _JsonTextAt(FunctionElement[Any]):
    """JSON text extraction at a dot path, compiled per dialect (SQLite vs PostgreSQL)."""

    name = "json_text_at"
    type = String()
    inherit_cache = False

    def __init__(self, column: Any, path_keys: tuple[str, ...]) -> None:
        self.column = column
        self.path_keys = path_keys
        super().__init__(column)


def json_text_at(column: Any, *path_keys: str) -> ColumnElement[Any]:
    """Return a SQL expression for the JSON text value at *path_keys* (object keys only).

    SQLite compiles to ``json_extract(column, '$.a.b')``. PostgreSQL chains
    subscript operators and compiles to the text form (``->>``) via
    :meth:`sqlalchemy.sql.elements.BinaryExpression.as_string`.

    Each key must match ``[A-Za-z0-9_]+`` so path segments cannot be confused
    with JSON path injection.
    """
    if not path_keys:
        msg = "json_text_at requires at least one path key"
        raise ValueError(msg)
    for k in path_keys:
        if not _PATH_KEY_RE.fullmatch(k):
            msg = f"invalid json path key: {k!r}"
            raise ValueError(msg)
    return _JsonTextAt(column, path_keys)


def json_text_at_upper_trimmed(column: Any, *path_keys: str) -> ColumnElement[Any]:
    """``upper(trim(json_text_at(...)))`` for comparing stored JSON strings to normalised identifiers."""
    return func.upper(func.trim(json_text_at(column, *path_keys)))


@compiles(_JsonTextAt)
def _compile_json_text_unsupported(_element: _JsonTextAt, _compiler: Any, **_kw: Any) -> NoReturn:
    msg = "json_text_at is only supported for sqlite and postgresql"
    raise CompileError(msg)


@compiles(_JsonTextAt, "sqlite")
def _compile_json_text_sqlite(element: _JsonTextAt, compiler: Any, **kw: Any) -> str:
    path = "$" + "".join(f".{k}" for k in element.path_keys)
    return cast(str, compiler.process(func.json_extract(element.column, path), **kw))


@compiles(_JsonTextAt, "postgresql")
def _compile_json_text_postgresql(element: _JsonTextAt, compiler: Any, **kw: Any) -> str:
    expr: Any = element.column
    for k in element.path_keys:
        expr = expr[k]
    return cast(str, compiler.process(expr.as_string(), **kw))

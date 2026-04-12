from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


class Capability(Enum):
    EXTENDED_THINKING = auto()
    PROMPT_CACHING = auto()
    TOOL_USE_STREAMING = auto()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResult: ...

    def capabilities(self) -> frozenset[Capability]: ...

    async def close(self) -> None: ...


__all__ = [
    "Capability",
    "CompletionResult",
    "LLMProvider",
    "TokenUsage",
    "ToolCall",
]

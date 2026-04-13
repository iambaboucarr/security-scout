"""Anthropic-native LLMProvider implementation.

This is the *only* module in the codebase that imports ``anthropic``.
All other code accesses LLM capabilities through ``LLMProvider``.
"""

from __future__ import annotations

from typing import Any

import anthropic
import structlog

from ai.provider import Capability, CompletionResult, LLMProvider, TokenUsage, ToolCall

_LOG = structlog.get_logger(__name__)

_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.EXTENDED_THINKING,
        Capability.PROMPT_CACHING,
        Capability.TOOL_USE_STREAMING,
    },
)


class AnthropicProvider:
    """``LLMProvider`` backed by the Anthropic Messages API."""

    def __init__(self, *, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            api_kwargs["system"] = system
        if tools is not None:
            api_kwargs["tools"] = tools
        api_kwargs.update(kwargs)

        response = await self._client.messages.create(**api_kwargs)
        return _parse_response(response)

    def capabilities(self) -> frozenset[Capability]:
        return _CAPABILITIES

    async def close(self) -> None:
        await self._client.close()


def _parse_response(response: anthropic.types.Message) -> CompletionResult:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input if isinstance(block.input, dict) else {},
                ),
            )

    usage = TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    return CompletionResult(
        text="".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        raw=response,
    )


def create_provider(api_key: str) -> LLMProvider:
    """Factory that returns an ``AnthropicProvider`` typed as ``LLMProvider``."""
    return AnthropicProvider(api_key=api_key)


__all__ = ["AnthropicProvider", "create_provider"]

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

from ai.provider import Capability, CompletionResult, LLMProvider, TokenUsage, ToolCall


class _FakeLLMProvider:
    """Minimal LLMProvider for testing — returns canned responses."""

    def __init__(self, text: str = "", tool_calls: list[ToolCall] | None = None) -> None:
        self._text = text
        self._tool_calls = tool_calls or []
        self.last_call_kwargs: dict[str, Any] = {}

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
        self.last_call_kwargs = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "tools": tools,
            **kwargs,
        }
        return CompletionResult(text=self._text, tool_calls=self._tool_calls)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset()

    async def close(self) -> None:
        pass


def test_fake_provider_satisfies_protocol() -> None:
    provider = _FakeLLMProvider()
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_fake_provider_returns_canned_response() -> None:
    provider = _FakeLLMProvider(text="hello")
    result = await provider.complete(
        [{"role": "user", "content": "hi"}],
        model="test-model",
        max_tokens=100,
    )
    assert result.text == "hello"
    assert result.tool_calls == []
    assert result.usage == TokenUsage()


def test_completion_result_defaults() -> None:
    r = CompletionResult(text="x")
    assert r.tool_calls == []
    assert r.usage.input_tokens == 0
    assert r.usage.output_tokens == 0
    assert r.raw is None


def test_tool_call_defaults() -> None:
    tc = ToolCall(id="tc_1", name="search")
    assert tc.input == {}


def test_tool_call_with_input() -> None:
    tc = ToolCall(id="tc_1", name="search", input={"query": "test"})
    assert tc.input == {"query": "test"}


def test_capability_enum_members() -> None:
    assert Capability.EXTENDED_THINKING is not None
    assert Capability.PROMPT_CACHING is not None
    assert Capability.TOOL_USE_STREAMING is not None
    assert len(Capability) == 3


def test_token_usage_frozen() -> None:
    u = TokenUsage(input_tokens=10, output_tokens=20)
    assert u.input_tokens == 10
    assert u.output_tokens == 20
    with pytest.raises(AttributeError):
        u.input_tokens = 5  # type: ignore[misc]

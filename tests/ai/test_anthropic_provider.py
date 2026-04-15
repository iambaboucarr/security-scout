# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.anthropic_provider import AnthropicProvider, _parse_response, create_provider
from ai.provider import Capability, CompletionResult, LLMProvider


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(*, id: str, name: str, input: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


def _make_response(*blocks: MagicMock, input_tokens: int = 10, output_tokens: int = 20) -> MagicMock:
    resp = MagicMock()
    resp.content = list(blocks)
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def test_parse_response_text_only() -> None:
    resp = _make_response(_make_text_block("hello world"))
    result = _parse_response(resp)
    assert result.text == "hello world"
    assert result.tool_calls == []
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20


def test_parse_response_multiple_text_blocks() -> None:
    resp = _make_response(_make_text_block("foo"), _make_text_block("bar"))
    result = _parse_response(resp)
    assert result.text == "foobar"


def test_parse_response_tool_use() -> None:
    resp = _make_response(
        _make_text_block("thinking"),
        _make_tool_use_block(id="tc_1", name="search", input={"q": "test"}),
    )
    result = _parse_response(resp)
    assert result.text == "thinking"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tc_1"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].input == {"q": "test"}


def test_parse_response_tool_use_non_dict_input() -> None:
    resp = _make_response(
        _make_tool_use_block(id="tc_2", name="run", input="not_a_dict"),
    )
    result = _parse_response(resp)
    assert result.tool_calls[0].input == {}


def test_parse_response_preserves_raw() -> None:
    resp = _make_response(_make_text_block("x"))
    result = _parse_response(resp)
    assert result.raw is resp


def test_capabilities_includes_all() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    caps = provider.capabilities()
    assert Capability.EXTENDED_THINKING in caps
    assert Capability.PROMPT_CACHING in caps
    assert Capability.TOOL_USE_STREAMING in caps


def test_create_provider_returns_llm_provider() -> None:
    provider = create_provider("sk-ant-test")
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_complete_delegates_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    mock_client = MagicMock()
    resp = _make_response(_make_text_block("result"))
    mock_client.messages.create = AsyncMock(return_value=resp)
    provider._client = mock_client

    result = await provider.complete(
        [{"role": "user", "content": "hello"}],
        model="claude-sonnet-4-6",
        max_tokens=100,
        system="Be helpful.",
    )

    assert isinstance(result, CompletionResult)
    assert result.text == "result"
    call_kw = mock_client.messages.create.call_args.kwargs
    assert call_kw["model"] == "claude-sonnet-4-6"
    assert call_kw["system"] == "Be helpful."
    assert call_kw["max_tokens"] == 100


@pytest.mark.asyncio
async def test_complete_passes_kwargs_through(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    mock_client = MagicMock()
    resp = _make_response(_make_text_block("ok"))
    mock_client.messages.create = AsyncMock(return_value=resp)
    provider._client = mock_client

    await provider.complete(
        [{"role": "user", "content": "hi"}],
        model="claude-haiku-4-5",
        max_tokens=50,
        temperature=0.5,
    )

    call_kw = mock_client.messages.create.call_args.kwargs
    assert call_kw["temperature"] == 0.5


@pytest.mark.asyncio
async def test_close_delegates_to_sdk() -> None:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    await provider.close()
    mock_client.close.assert_awaited_once()

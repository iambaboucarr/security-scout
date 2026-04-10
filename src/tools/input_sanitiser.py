"""Semantic firewall: sanitise untrusted strings before LLM prompts (ADR-010).

Regex-based redaction targets common jailbreak phrasing; it may also alter benign vendor
wording (e.g. lines that look like ``new instructions:``). Treat sanitised text as safe to
embed, not as a lossless copy of the source.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Final

# Size limits (Guardrail 7: MCP raw response size; general cap avoids prompt bombs)

DEFAULT_MAX_CHARS: Final[int] = 512_000
MCP_MAX_RESPONSE_BYTES: Final[int] = 50 * 1024

EXTERNAL_CONTENT_TAG: Final[str] = "external_content"


class ExternalContentKind(StrEnum):
    """Semantic label for framed blocks; maps to inner XML-style tags."""

    GENERIC = "generic"
    ADVISORY = "advisory"
    POC_CODE = "poc_code"
    PR_DIFF = "pr_diff"
    NUCLEI_OUTPUT = "nuclei_output"
    MCP_RESPONSE = "mcp_response"
    SARIF_SNIPPET = "sarif_snippet"


_INNER_TAG: dict[ExternalContentKind, str] = {
    ExternalContentKind.GENERIC: "untrusted_text",
    ExternalContentKind.ADVISORY: "advisory_text",
    ExternalContentKind.POC_CODE: "poc_code",
    ExternalContentKind.PR_DIFF: "pr_diff",
    ExternalContentKind.NUCLEI_OUTPUT: "nuclei_output",
    ExternalContentKind.MCP_RESPONSE: "mcp_response",
    ExternalContentKind.SARIF_SNIPPET: "sarif_snippet",
}


def inner_tags_for_prompt_contract() -> str:
    """Comma-separated inner tag names from ``ExternalContentKind`` / ``_INNER_TAG`` for the system prompt."""
    names = sorted(frozenset(_INNER_TAG.values()))
    return ", ".join(f"<{name}>" for name in names)


# Characters that break markdown fences, XML boundaries, or hide payload text (bidi / zero-width).
_DISRUPTIVE_CODEPOINTS: Final[frozenset[str]] = frozenset(
    "".join(
        [
            "\u200b",
            "\u200c",
            "\u200d",
            "\u2060",
            "\ufeff",
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        ]
    )
)

_REDACTED_INJECTION_MARKER: Final[str] = "[redacted-injection-pattern]"
_REDACTED_INJECTION_MARKER_WITH_COLON: Final[str] = f"{_REDACTED_INJECTION_MARKER}: "

# High-signal prompt-injection phrases (case-insensitive). Curated for common jailbreak / override text.
_INJECTION_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (
        re.compile(r"(?is)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|context)\b"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(r"(?is)\bdisregard\s+(all\s+)?(previous|prior|above)\b"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(r"(?is)\bforget\s+(everything|all\s+)?(above|before|prior)\b"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(
            r"(?is)\b(new|updated?)\s+instructions?\s*:\s*",
        ),
        _REDACTED_INJECTION_MARKER_WITH_COLON,
    ),
    (
        re.compile(r"(?is)\b(system|developer)\s+message\s*:\s*"),
        _REDACTED_INJECTION_MARKER_WITH_COLON,
    ),
    (
        re.compile(r"(?is)\b(you\s+are\s+now|enter|enable)\s+(DAN|jailbreak|developer)\s+mode\b"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(r"(?is)\boverride\s+(prior|previous|system)\s+(instructions?|rules?)\b"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(r"(?is)\[\s*/?\s*(INST|SYSTEM)\s*\]"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(r"(?is)<\s*/?\s*(system|user|assistant)\s*>"),
        _REDACTED_INJECTION_MARKER,
    ),
    (
        re.compile(
            r"(?is)\b(end\s+of|terminate)\s+(system|user)\s+(message|prompt)\b",
        ),
        _REDACTED_INJECTION_MARKER,
    ),
]

# U+02CB modifier letter grave — visually distinct from markdown backticks / code fences.
_BACKTICK_REPLACEMENT: Final[str] = "\u02cb"
_TRIPLE_BACKTICK_PLACEHOLDER: Final[str] = "⟨code_fence_block⟩"

_DEFAULT_TRUNCATE_SUFFIX: Final[str] = "\n\n[truncated: exceeded maximum length for untrusted content]"


def _strip_disruptive_unicode(text: str) -> str:
    return "".join(ch for ch in text if ch not in _DISRUPTIVE_CODEPOINTS)


def _neutralize_injection_patterns(text: str) -> str:
    out = text
    for pattern, repl in _INJECTION_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _neutralize_backticks(text: str) -> str:
    # Longest sequence first so ``` is not turned into three single replacements incorrectly.
    return text.replace("```", _TRIPLE_BACKTICK_PLACEHOLDER).replace("`", _BACKTICK_REPLACEMENT)


def _escape_xml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate_utf8_to_byte_budget(text: str, max_bytes: int, suffix: str) -> str:
    """Shorten ``text`` so the UTF-8 encoding of ``result`` is at most ``max_bytes`` octets."""
    if max_bytes < 1:
        msg = "max_bytes must be >= 1"
        raise ValueError(msg)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    suffix_b = suffix.encode("utf-8")
    budget = max_bytes - len(suffix_b)
    if budget <= 0:
        return suffix_b[:max_bytes].decode("utf-8", errors="replace")
    prefix_b = encoded[:budget]
    while prefix_b:
        try:
            return prefix_b.decode("utf-8") + suffix
        except UnicodeDecodeError:
            prefix_b = prefix_b[:-1]
    return suffix


def sanitize_text(
    text: str,
    *,
    max_chars: int | None = None,
    truncate_suffix: str = _DEFAULT_TRUNCATE_SUFFIX,
) -> str:
    """Return text safe to embed inside framed external-content delimiters.

    Escapes XML/markdown-sensitive characters, removes disruptive Unicode, neutralises common
    injection phrases, and replaces backticks that could break fenced code regions in prompts.
    """
    if not isinstance(text, str):
        msg = "sanitize_text expects str"
        raise TypeError(msg)

    limit = DEFAULT_MAX_CHARS if max_chars is None else max_chars
    if limit < 1:
        msg = "max_chars must be >= 1"
        raise ValueError(msg)

    if len(text) > limit:
        text = text[:limit] + truncate_suffix

    normalized = unicodedata.normalize("NFKC", text)
    stripped = _strip_disruptive_unicode(normalized)
    injected = _neutralize_injection_patterns(stripped)
    tick_safe = _neutralize_backticks(injected)
    return _escape_xml_text(tick_safe)


def frame_external_content(kind: ExternalContentKind, sanitized_body: str) -> str:
    """Wrap already-sanitised text in explicit external-content delimiters (ADR-010).

    Do not pass raw advisory/PoC/MCP text here; run :func:`sanitize_text` first, or use
    :func:`prepare_for_llm`.
    """
    inner = _INNER_TAG[kind]
    return (
        f'<{EXTERNAL_CONTENT_TAG} kind="{kind.value}">\n'
        f"<{inner}>\n{sanitized_body}\n</{inner}>\n"
        f"</{EXTERNAL_CONTENT_TAG}>"
    )


def prepare_for_llm(
    kind: ExternalContentKind,
    raw_text: str,
    *,
    max_chars: int | None = None,
    truncate_suffix: str = _DEFAULT_TRUNCATE_SUFFIX,
) -> str:
    cleaned = sanitize_text(raw_text, max_chars=max_chars, truncate_suffix=truncate_suffix)
    return frame_external_content(kind, cleaned)


def prepare_mcp_response_for_llm(raw_text: str, *, truncate_suffix: str = _DEFAULT_TRUNCATE_SUFFIX) -> str:
    """MCP responses: 50 KiB raw cap and ``mcp_response`` framing.

    The limit applies to the raw response before sanitisation. The framed string sent to the
    model may exceed 50 KiB because of escaping and wrapper markup.
    """
    truncated = _truncate_utf8_to_byte_budget(raw_text, MCP_MAX_RESPONSE_BYTES, truncate_suffix)
    cleaned = sanitize_text(truncated, max_chars=None, truncate_suffix=truncate_suffix)
    return frame_external_content(ExternalContentKind.MCP_RESPONSE, cleaned)

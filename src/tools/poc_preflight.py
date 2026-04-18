# SPDX-License-Identifier: Apache-2.0
"""Deterministic PoC pre-flight validator.

Analyses PoC artifacts for hostile indicators before sandbox execution.
Pure pattern matching and static analysis — no LLM calls.

Called by the Orchestrator between triage and sandbox execution.
No access to Docker, Slack, or SCM write APIs.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import structlog

_LOG = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PreflightVerdict(StrEnum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


@dataclass(frozen=True, slots=True)
class PreflightIndicator:
    category: str
    pattern: str
    severity_weight: float
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightThresholds:
    clean_max: float = 0.3
    malicious_min: float = 0.7


@dataclass(frozen=True, slots=True)
class PreflightResult:
    verdict: PreflightVerdict
    score: float
    indicators: tuple[PreflightIndicator, ...]
    thresholds: PreflightThresholds


_DEFAULT_THRESHOLDS = PreflightThresholds()

# Scoring: max indicator weight per category + bonus per additional category.
_CROSS_CATEGORY_BONUS = 0.1


# ---------------------------------------------------------------------------
# CWE context sets — lower weights for expected PoC behaviour
# ---------------------------------------------------------------------------

_NETWORK_EXPECTED_CWES: frozenset[str] = frozenset(
    {
        "CWE-918",  # SSRF
        "CWE-352",  # CSRF
        "CWE-601",  # Open redirect
        "CWE-116",  # Improper output encoding
        "CWE-79",  # XSS (may call back to attacker server)
    }
)

_CODE_EXEC_EXPECTED_CWES: frozenset[str] = frozenset(
    {
        "CWE-94",  # Code injection
        "CWE-95",  # Eval injection
        "CWE-502",  # Deserialization of untrusted data
    }
)


# Known-malicious package lists (static tripwire — NOT a comprehensive scanner)
_KNOWN_MALICIOUS_PYPI: frozenset[str] = frozenset(
    {
        "python3-dateutil",
        "python-binance",
        "python-mongo",
        "crossenv",
        "colourama",
        "requessts",
        "python-mysql",
        "libpeshnern",
        "beautifulsup4",
        "numpyy",
        "pipsqlite3",
        "python-openssl",
        "djang0",
        "flasck",
        "urlib3",
        "requets",
    }
)

_KNOWN_MALICIOUS_NPM: frozenset[str] = frozenset(
    {
        "event-stream",
        "flatmap-stream",
        "ua-parser-is",
        "coa",
        "rc",
        "eslint-scope",
        "conventional-changelog-angular",
        "colors-hijack",
        "faker-hijack",
        "node-ipc-malware",
    }
)

_KNOWN_MALICIOUS_ALL: frozenset[str] = _KNOWN_MALICIOUS_PYPI | _KNOWN_MALICIOUS_NPM


# ---------------------------------------------------------------------------
# Check category 1: Network indicators
# ---------------------------------------------------------------------------

# (pattern, base_weight, detail, hostile)
# hostile=True means CWE context cannot reduce weight below _HOSTILE_FLOOR.
_NETWORK_PATTERNS: list[tuple[str, float, str, bool]] = [
    (r"\bsocket\.connect\b", 0.4, "socket.connect call", False),
    (r"\burllib\.request\b", 0.2, "urllib.request usage", False),
    (r"\brequests\.(get|post|put|delete|patch|head)\s*\(", 0.2, "requests HTTP call", False),
    (r"\bhttpx\.(get|post|put|delete|patch|head|AsyncClient|Client)\b", 0.15, "httpx usage", False),
    (r"\bcurl\s+", 0.3, "curl command", False),
    (r"\bwget\s+", 0.3, "wget command", False),
    (r"\bnc\s+-e\b", 0.9, "netcat reverse shell", True),
    (r"bash\s+-i\s+>&\s*/dev/tcp/", 0.95, "bash reverse shell", True),
    (r"\breverse_tcp\b", 0.9, "reverse TCP payload reference", True),
    (r"\bmeterpreter\b", 0.9, "Meterpreter payload reference", True),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", 0.15, "IP address literal", False),
    (r"dns.*exfil|exfil.*dns", 0.8, "DNS exfiltration pattern", True),
    (r"\bnslookup\b.*\$|\bdig\b.*\$", 0.7, "DNS lookup with variable interpolation", False),
    (r"/dev/tcp/", 0.85, "/dev/tcp device redirection", True),
]

# CWE weight multiplier when network calls are expected
_NETWORK_CWE_MULTIPLIER = 0.3

# Floor weight for clearly hostile patterns — CWE context cannot reduce below this
_HOSTILE_FLOOR = 0.7


def _check_network(content: str, cwe_context: frozenset[str]) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []
    multiplier = _NETWORK_CWE_MULTIPLIER if cwe_context & _NETWORK_EXPECTED_CWES else 1.0

    for pattern, base_weight, detail, hostile in _NETWORK_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            adjusted = base_weight * multiplier
            if hostile:
                adjusted = max(adjusted, _HOSTILE_FLOOR)
            indicators.append(
                PreflightIndicator(
                    category="network",
                    pattern=pattern,
                    severity_weight=round(adjusted, 3),
                    detail=detail,
                )
            )
    return indicators


# ---------------------------------------------------------------------------
# Check category 2: Privilege escalation
# ---------------------------------------------------------------------------

_PRIVESC_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bsetuid\b", 0.7, "setuid call"),
    (r"\bsetgid\b", 0.7, "setgid call"),
    (r"\bmount\s*\(", 0.7, "mount syscall"),
    (r"\bchroot\b", 0.6, "chroot call"),
    (r"\bnsenter\b", 0.8, "nsenter — namespace entry"),
    (r"\bunshare\b", 0.8, "unshare — namespace creation"),
    (r"/proc/self/exe", 0.7, "/proc/self/exe reference"),
    (r"\bCAP_SYS_ADMIN\b", 0.8, "CAP_SYS_ADMIN capability"),
    (r"\bptrace\b", 0.8, "ptrace — process tracing"),
    (r"\bprctl\b", 0.5, "prctl call"),
    (r"\bdocker\.sock\b|/var/run/docker", 0.85, "Docker socket access"),
    (r"\bpodman\.sock\b|/run/podman/podman", 0.85, "Podman socket access"),
]


def _check_privesc(content: str) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []
    for pattern, weight, detail in _PRIVESC_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            indicators.append(
                PreflightIndicator(
                    category="privesc",
                    pattern=pattern,
                    severity_weight=weight,
                    detail=detail,
                )
            )
    return indicators


# ---------------------------------------------------------------------------
# Check category 3: Resource abuse
# ---------------------------------------------------------------------------

_RESOURCE_PATTERNS: list[tuple[str, float, str]] = [
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:", 0.9, "fork bomb"),
    (r"\bfork\s*\(\s*\)\s*.*\bfork\s*\(\s*\)", 0.8, "recursive fork"),
    (r"\bwhile\s+(true|1)\b", 0.25, "infinite loop"),
    (r"\bdd\s+if=/dev/zero\b", 0.7, "dd write from /dev/zero"),
    (r"/dev/urandom.*>|>\s*/dev/urandom", 0.5, "/dev/urandom write"),
    (r"\bmalloc\b.*\bwhile\b|\bwhile\b.*\bmalloc\b", 0.5, "unbounded allocation in loop"),
    (r"\b(yes|cat\s+/dev/zero)\s*\|", 0.6, "pipe-based resource flood"),
]


def _check_resource_abuse(content: str) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []
    for pattern, weight, detail in _RESOURCE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            indicators.append(
                PreflightIndicator(
                    category="resource_abuse",
                    pattern=pattern,
                    severity_weight=weight,
                    detail=detail,
                )
            )
    return indicators


# ---------------------------------------------------------------------------
# Check category 4: Dependency analysis
# ---------------------------------------------------------------------------

_REQUIREMENTS_LINE = re.compile(
    r"^([a-zA-Z0-9][\w.\-]*)",
    re.MULTILINE,
)


def _parse_requirements_txt(content: str) -> list[str]:
    return [m.group(1).lower() for m in _REQUIREMENTS_LINE.finditer(content) if not m.group(1).startswith(("-", "#"))]


def _parse_package_json(content: str) -> list[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError, ValueError:
        return []
    packages: list[str] = []
    for key in ("dependencies", "devDependencies"):
        deps = data.get(key)
        if isinstance(deps, dict):
            packages.extend(deps.keys())
    return [p.lower() for p in packages]


def _check_dependencies(
    content: str,
    dependency_contents: dict[str, str],
) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []
    packages: list[str] = []

    for filename, file_content in dependency_contents.items():
        lower = filename.lower()
        if lower.endswith(("requirements.txt", "requirements.in")):
            packages.extend(_parse_requirements_txt(file_content))
        elif lower.endswith("package.json"):
            packages.extend(_parse_package_json(file_content))

    # Also scan main content for inline pip install / npm install
    inline_pip = re.findall(r"pip\s+install\s+([\w.\-]+)", content, re.IGNORECASE)
    inline_npm = re.findall(r"npm\s+install\s+([\w.\-@/]+)", content, re.IGNORECASE)
    packages.extend(p.lower() for p in inline_pip)
    packages.extend(p.lower() for p in inline_npm)

    seen: set[str] = set()
    for pkg in packages:
        if pkg in seen:
            continue
        seen.add(pkg)
        if pkg in _KNOWN_MALICIOUS_ALL:
            indicators.append(
                PreflightIndicator(
                    category="dependency",
                    pattern=f"known_malicious:{pkg}",
                    severity_weight=0.9,
                    detail=f"Known malicious package: {pkg}",
                )
            )
    return indicators


# ---------------------------------------------------------------------------
# Check category 5: Obfuscation detection
# ---------------------------------------------------------------------------

_OBFUSCATION_PATTERNS: list[tuple[str, float, str]] = [
    (r"\beval\s*\(", 0.4, "eval() call"),
    (r"\bexec\s*\(", 0.4, "exec() call"),
    (r"\bcompile\s*\(", 0.3, "compile() call"),
    (r"\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){7,}", 0.5, "bulk hex escape sequences"),
]

# CWE weight multiplier when code execution patterns are expected
_OBFUSCATION_CWE_MULTIPLIER = 0.5

# Base64 blob detection: 256+ bytes of valid base64 characters
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/=]{256,}")

_ENTROPY_THRESHOLD = 4.5
_MIN_ENTROPY_STRING_LEN = 64


def _shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    freq: dict[str, int] = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _check_obfuscation(content: str, cwe_context: frozenset[str]) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []
    multiplier = _OBFUSCATION_CWE_MULTIPLIER if cwe_context & _CODE_EXEC_EXPECTED_CWES else 1.0

    for pattern, base_weight, detail in _OBFUSCATION_PATTERNS:
        if re.search(pattern, content):
            indicators.append(
                PreflightIndicator(
                    category="obfuscation",
                    pattern=pattern,
                    severity_weight=round(base_weight * multiplier, 3),
                    detail=detail,
                )
            )

    if _BASE64_BLOB.search(content):
        indicators.append(
            PreflightIndicator(
                category="obfuscation",
                pattern="base64_blob>256",
                severity_weight=round(0.5 * multiplier, 3),
                detail="Base64-encoded blob > 256 bytes",
            )
        )

    for match in re.finditer(r'["\']([^"\']{64,})["\']', content):
        candidate = match.group(1)
        if len(candidate) >= _MIN_ENTROPY_STRING_LEN:
            entropy = _shannon_entropy(candidate)
            if entropy > _ENTROPY_THRESHOLD:
                indicators.append(
                    PreflightIndicator(
                        category="obfuscation",
                        pattern="high_entropy_string",
                        severity_weight=round(0.4 * multiplier, 3),
                        detail=f"High-entropy string (Shannon={entropy:.2f})",
                    )
                )
                break  # One indicator is enough
    return indicators


# ---------------------------------------------------------------------------
# Check category 6: File scope
# ---------------------------------------------------------------------------

_FILE_COUNT_THRESHOLDS: list[tuple[int, float, str]] = [
    (100, 0.6, "PoC contains > 100 files"),
    (50, 0.3, "PoC contains > 50 files"),
]

_SIZE_THRESHOLDS: list[tuple[int, float, str]] = [
    (10 * 1024 * 1024, 0.5, "PoC total size > 10 MB"),
    (5 * 1024 * 1024, 0.25, "PoC total size > 5 MB"),
]


def _check_file_scope(
    total_file_count: int,
    total_size_bytes: int,
) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []

    for threshold, weight, detail in _FILE_COUNT_THRESHOLDS:
        if total_file_count > threshold:
            indicators.append(
                PreflightIndicator(
                    category="file_scope",
                    pattern=f"file_count>{threshold}",
                    severity_weight=weight,
                    detail=detail,
                )
            )
            break  # Highest matching threshold only

    for threshold, weight, detail in _SIZE_THRESHOLDS:
        if total_size_bytes > threshold:
            indicators.append(
                PreflightIndicator(
                    category="file_scope",
                    pattern=f"total_size>{threshold}",
                    severity_weight=weight,
                    detail=detail,
                )
            )
            break

    return indicators


# ---------------------------------------------------------------------------
# Check category 7: Prompt injection markers
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, float, str]] = [
    (r"IGNORE\s+PREVIOUS\s+INSTRUCTIONS?", 0.8, "prompt injection: IGNORE PREVIOUS"),
    (r"SYSTEM\s*:", 0.5, "prompt injection: SYSTEM: prefix"),
    (r"<\|im_start\|>", 0.8, "prompt injection: ChatML delimiter"),
    (r"<\|im_end\|>", 0.7, "prompt injection: ChatML end delimiter"),
    (r"\[INST\]", 0.7, "prompt injection: Llama instruction tag"),
    (r"</tool_call>", 0.7, "prompt injection: tool_call close tag"),
    (r"<system>", 0.7, "prompt injection: <system> XML tag"),
    (r"</system>", 0.7, "prompt injection: </system> XML tag"),
    (r"<\|system\|>", 0.8, "prompt injection: system delimiter"),
    (r"Human:\s|Assistant:\s", 0.5, "prompt injection: role prefix"),
    (r"FORGET\s+(ALL|EVERYTHING|PRIOR)", 0.8, "prompt injection: FORGET directive"),
    (r"NEW\s+INSTRUCTIONS?\s*:", 0.7, "prompt injection: NEW INSTRUCTIONS"),
]


def _check_prompt_injection(content: str) -> list[PreflightIndicator]:
    indicators: list[PreflightIndicator] = []
    for pattern, weight, detail in _INJECTION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            indicators.append(
                PreflightIndicator(
                    category="prompt_injection",
                    pattern=pattern,
                    severity_weight=weight,
                    detail=detail,
                )
            )
    return indicators


# ---------------------------------------------------------------------------
# Scoring and classification
# ---------------------------------------------------------------------------


def compute_score(indicators: Sequence[PreflightIndicator]) -> float:
    """Aggregate score: max weight per category + cross-category bonus.

    Single high-weight indicator in one category → dominates score.
    Moderate indicators across multiple categories → escalates via bonus.
    """
    if not indicators:
        return 0.0

    max_per_category: dict[str, float] = {}
    for ind in indicators:
        current = max_per_category.get(ind.category, 0.0)
        max_per_category[ind.category] = max(current, ind.severity_weight)

    peak = max(max_per_category.values())
    additional_categories = len(max_per_category) - 1
    return min(1.0, peak + _CROSS_CATEGORY_BONUS * additional_categories)


def classify(score: float, thresholds: PreflightThresholds) -> PreflightVerdict:
    if score > thresholds.malicious_min:
        return PreflightVerdict.MALICIOUS
    if score < thresholds.clean_max:
        return PreflightVerdict.CLEAN
    return PreflightVerdict.SUSPICIOUS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def validate(
    poc_content: str,
    *,
    cwe_ids: list[str] | None = None,
    dependency_contents: dict[str, str] | None = None,
    total_file_count: int = 1,
    total_size_bytes: int = 0,
    thresholds: PreflightThresholds | None = None,
) -> PreflightResult:
    """Run all pre-flight checks and return a verdict with indicators.

    Parameters
    ----------
    poc_content:
        Main PoC script/template content.
    cwe_ids:
        CWE identifiers for context-aware weight adjustment.
    dependency_contents:
        Map of dependency filenames to their content (e.g. requirements.txt).
    total_file_count:
        Number of files in the PoC artifact.
    total_size_bytes:
        Total size of all PoC files in bytes.  Falls back to encoded
        ``poc_content`` length when 0.
    thresholds:
        Override default clean/malicious score boundaries.
    """
    effective_thresholds = thresholds or _DEFAULT_THRESHOLDS
    cwe_context = frozenset(cwe_ids) if cwe_ids else frozenset()
    dep_contents = dependency_contents or {}
    effective_size = total_size_bytes or len(poc_content.encode("utf-8"))

    indicators: list[PreflightIndicator] = []
    indicators.extend(_check_network(poc_content, cwe_context))
    indicators.extend(_check_privesc(poc_content))
    indicators.extend(_check_resource_abuse(poc_content))
    indicators.extend(_check_dependencies(poc_content, dep_contents))
    indicators.extend(_check_obfuscation(poc_content, cwe_context))
    indicators.extend(_check_file_scope(total_file_count, effective_size))
    indicators.extend(_check_prompt_injection(poc_content))

    score = compute_score(indicators)
    verdict = classify(score, effective_thresholds)
    frozen_indicators = tuple(indicators)

    _LOG.info(
        "preflight_validation_complete",
        metric_name="preflight_verdict_total",
        verdict=verdict.value,
        score=round(score, 3),
        indicator_count=len(frozen_indicators),
        categories=sorted({i.category for i in frozen_indicators}),
    )

    return PreflightResult(
        verdict=verdict,
        score=score,
        indicators=frozen_indicators,
        thresholds=effective_thresholds,
    )


__all__ = [
    "PreflightIndicator",
    "PreflightResult",
    "PreflightThresholds",
    "PreflightVerdict",
    "classify",
    "compute_score",
    "validate",
]

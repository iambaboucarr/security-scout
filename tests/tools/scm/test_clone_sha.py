# SPDX-License-Identifier: Apache-2.0
"""Tests for SHA-aware clone_repo in tools/scm/github.py."""

from __future__ import annotations

from tools.scm.github import is_sha_ref


class TestIsShaRef:
    def test_full_40_char_sha(self) -> None:
        assert is_sha_ref("a" * 40) is True

    def test_full_sha_mixed_case(self) -> None:
        assert is_sha_ref("AbCdEf0123456789" * 2 + "AbCdEf01") is True

    def test_short_7_char_sha(self) -> None:
        assert is_sha_ref("abc1234") is True

    def test_short_12_char_sha(self) -> None:
        assert is_sha_ref("abc123def456") is True

    def test_branch_name_not_sha(self) -> None:
        assert is_sha_ref("main") is False

    def test_tag_not_sha(self) -> None:
        assert is_sha_ref("v1.0.0") is False

    def test_too_short_6_chars(self) -> None:
        assert is_sha_ref("abc123") is False

    def test_41_chars_not_sha(self) -> None:
        assert is_sha_ref("a" * 41) is False

    def test_empty_string_not_sha(self) -> None:
        assert is_sha_ref("") is False

    def test_non_hex_chars_rejected(self) -> None:
        assert is_sha_ref("abcdefg") is False  # g is not hex

    def test_branch_with_slash(self) -> None:
        assert is_sha_ref("feature/foo") is False

    def test_refs_head_main(self) -> None:
        assert is_sha_ref("refs/heads/main") is False

# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def test_project_structure_exists():
    root = Path(__file__).resolve().parent.parent
    assert (root / "src" / "agents").is_dir()
    assert (root / "src" / "tools").is_dir()
    assert (root / "src" / "webhooks").is_dir()
    assert (root / "src" / "py.typed").is_file()
    assert (root / "sandbox").is_dir()

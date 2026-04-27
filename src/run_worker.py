# SPDX-License-Identifier: Apache-2.0
"""Wrapper to run the ARQ worker on Python 3.14+.

arq 0.27 calls asyncio.get_event_loop() during __init__ which raises on
Python 3.12+ when no loop exists.  This script ensures one is available.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load ``.env`` before importing ``worker`` so ``os.environ`` matches pydantic-settings when the
# worker process starts (``ADVISORY_POLL_INTERVAL`` and other vars used by :class:`Settings` at runtime).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from arq.worker import run_worker  # noqa: E402

from worker import WorkerSettings, configure_worker_cron_jobs  # noqa: E402


def main() -> None:
    asyncio.set_event_loop(asyncio.new_event_loop())
    configure_worker_cron_jobs()
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()

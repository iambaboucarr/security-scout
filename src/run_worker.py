# SPDX-License-Identifier: Apache-2.0
"""Wrapper to run the ARQ worker on Python 3.14+.

arq 0.27 calls asyncio.get_event_loop() during __init__ which raises on
Python 3.12+ when no loop exists.  This script ensures one is available.
"""

from __future__ import annotations

import asyncio

from arq.worker import run_worker

from worker import WorkerSettings


def main() -> None:
    asyncio.set_event_loop(asyncio.new_event_loop())
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()

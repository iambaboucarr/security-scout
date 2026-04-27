# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import patch

import pytest

from run_worker import main
from worker import WorkerSettings


def test_main_sets_event_loop_and_calls_arq_run_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVISORY_POLL_INTERVAL", "disabled")
    prev_cron = WorkerSettings.cron_jobs
    try:
        with patch("run_worker.run_worker") as mock_arq_run:
            main()
        mock_arq_run.assert_called_once_with(WorkerSettings)
    finally:
        WorkerSettings.cron_jobs = prev_cron
